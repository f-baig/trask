"""Deterministic, parameterized 2D vehicle dynamics.

The model deliberately sits between arcade point-mass motion and a full tire
simulator. Longitudinal force balance is physical (engine, braking, rolling
resistance, aerodynamic drag, and traction limits); lateral motion is a stable
transient bicycle model with steering slew, yaw inertia, axle stiffness,
friction-circle limits, and load-transfer grip loss. A control decision spans
multiple fixed physics substeps, so model-call cadence and integration quality
are independent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .models import DynamicsSpec, RoadDynamicsSpec


@dataclass
class VehiclePhysicsState:
    x: float
    y: float
    heading_radians: float
    longitudinal_velocity_mps: float = 0.0
    lateral_velocity_mps: float = 0.0
    yaw_rate_radians_per_second: float = 0.0
    steering_angle_radians: float = 0.0
    longitudinal_acceleration_mps2: float = 0.0
    lateral_acceleration_mps2: float = 0.0
    slip_angle_radians: float = 0.0
    aerodynamic_drag_n: float = 0.0
    rolling_resistance_n: float = 0.0
    lateral_load_transfer_n: float = 0.0


def surface_road_dynamics(surface: str) -> RoadDynamicsSpec:
    if surface == "clay":
        return RoadDynamicsSpec(
            friction_coefficient=.72, lateral_grip_multiplier=.78,
            rolling_resistance_multiplier=1.8,
            off_track_friction_coefficient=.48,
        )
    if surface == "ice":
        return RoadDynamicsSpec(
            friction_coefficient=.32, lateral_grip_multiplier=.68,
            rolling_resistance_multiplier=.65,
            off_track_friction_coefficient=.24,
            off_track_rolling_resistance_multiplier=2.2,
        )
    return RoadDynamicsSpec()


def apply_surface_grip(road: RoadDynamicsSpec, grip: float) -> RoadDynamicsSpec:
    """Scale a surface preset by a continuous grip multiplier.

    Slipperiness is a condition, not a separate surface: "a slippery asphalt
    circuit" keeps the asphalt palette and physics family while scaling the
    friction that both the player and every opponent must respect.
    """
    if abs(grip - 1.0) < 1e-9:
        return road.model_copy(deep=True)
    return road.model_copy(update={
        "friction_coefficient": _clamp(road.friction_coefficient * grip, .05, 2.0),
        "lateral_grip_multiplier": _clamp(road.lateral_grip_multiplier * grip, .1, 2.0),
        "off_track_friction_coefficient": _clamp(
            road.off_track_friction_coefficient * grip, .05, 1.5,
        ),
    })


def apply_dynamics_preset(dynamics: DynamicsSpec, preset: str) -> DynamicsSpec:
    """Return an explicit condition variant without mutating the source scene."""
    vehicle = dynamics.vehicle
    road = dynamics.road
    if preset in {"normal", "balanced", "fog", "action_delay", "obstacle_shift"}:
        return dynamics.model_copy(deep=True)
    if preset == "low_grip":
        road = road.model_copy(update={
            "friction_coefficient": road.friction_coefficient * .58,
            "lateral_grip_multiplier": road.lateral_grip_multiplier * .72,
        })
    elif preset == "worn_tires":
        vehicle = vehicle.model_copy(update={
            "tire_friction_multiplier": vehicle.tire_friction_multiplier * .62,
            "front_cornering_stiffness_n_per_rad": vehicle.front_cornering_stiffness_n_per_rad * .7,
            "rear_cornering_stiffness_n_per_rad": vehicle.rear_cornering_stiffness_n_per_rad * .7,
        })
    elif preset == "heavy_car":
        vehicle = vehicle.model_copy(update={
            "mass_kg": vehicle.mass_kg * 1.35,
            "yaw_inertia_kg_m2": vehicle.yaw_inertia_kg_m2 * 1.55,
        })
    elif preset == "rear_bias":
        vehicle = vehicle.model_copy(update={
            "front_weight_fraction": .43,
            "rear_cornering_stiffness_n_per_rad": vehicle.rear_cornering_stiffness_n_per_rad * .78,
        })
    elif preset == "high_drag":
        vehicle = vehicle.model_copy(update={
            "drag_coefficient": min(1.5, vehicle.drag_coefficient * 2.6),
            "frontal_area_m2": min(4.0, vehicle.frontal_area_m2 * 1.18),
        })
    elif preset == "high_downforce":
        vehicle = vehicle.model_copy(update={
            "lift_coefficient": -1.15,
            "drag_coefficient": min(1.5, vehicle.drag_coefficient * 1.5),
        })
    else:
        raise ValueError(f"Unknown dynamics condition: {preset}")
    return dynamics.model_copy(update={"vehicle": vehicle, "road": road})


def integrate_vehicle_substep(
    state: VehiclePhysicsState,
    dynamics: DynamicsSpec,
    *,
    throttle: float,
    brake: float,
    steering: float,
    nitro: bool,
    on_track: bool,
    grade_radians: float = 0.0,
    bank_radians: float = 0.0,
) -> VehiclePhysicsState:
    """Advance one fixed substep using only deterministic scalar operations.

    `grade_radians` is the uphill slope of the road under the car and
    `bank_radians` its cross-slope. Both default to zero, which reproduces the
    planar model exactly, so the 2D engine is bit-for-bit unaffected by the
    existence of the 3D one.
    """
    vehicle = dynamics.vehicle
    road = dynamics.road
    dt = 1.0 / dynamics.physics_hz
    gravity = dynamics.gravity_mps2
    mass = vehicle.mass_kg
    velocity = max(0.0, state.longitudinal_velocity_mps)

    target_steering = math.radians(vehicle.max_steering_angle_degrees) * _clamp(steering, -1.0, 1.0)
    steering_step = math.radians(vehicle.steering_rate_degrees_per_second) * dt
    steering_angle = state.steering_angle_radians + _clamp(
        target_steering - state.steering_angle_radians, -steering_step, steering_step,
    )

    road_mu = road.friction_coefficient if on_track else road.off_track_friction_coefficient
    lateral_mu = road_mu * road.lateral_grip_multiplier * vehicle.tire_friction_multiplier
    rolling_multiplier = (
        road.rolling_resistance_multiplier
        if on_track else road.off_track_rolling_resistance_multiplier
    )
    dynamic_pressure = .5 * dynamics.air_density_kg_m3 * velocity * velocity
    aerodynamic_drag = dynamic_pressure * vehicle.drag_coefficient * vehicle.frontal_area_m2
    downforce = max(0.0, -dynamic_pressure * vehicle.lift_coefficient * vehicle.frontal_area_m2)
    # Only the component of weight perpendicular to the road presses the tires
    # down; the in-plane component becomes the slope force below.
    slope_normal_fraction = math.cos(grade_radians) * math.cos(bank_radians)
    normal_load = mass * gravity * slope_normal_fraction + downforce
    rolling_resistance = (
        vehicle.rolling_resistance_coefficient * rolling_multiplier * normal_load
        if velocity > .02 else 0.0
    )
    cornering_scrub = (
        normal_load * abs(math.sin(steering_angle)) * .95 * min(1.0, velocity / 5.0)
    )

    power_limited_force = vehicle.engine_power_w / max(velocity, 3.0)
    requested_drive = throttle * min(vehicle.engine_force_n, power_limited_force)
    if nitro:
        requested_drive += vehicle.nitro_force_n
    driven_axle_share = max(.35, 1.0 - vehicle.front_weight_fraction + .12)
    traction_limit = road_mu * vehicle.tire_friction_multiplier * normal_load * driven_axle_share
    drive_force = min(requested_drive, traction_limit)
    brake_force = brake * min(vehicle.brake_force_n, road_mu * vehicle.tire_friction_multiplier * normal_load)
    limiter_force = mass * max(0.0, velocity - vehicle.max_speed_mps * (1.35 if nitro else 1.0)) * 5.0
    slope_force = mass * gravity * math.sin(grade_radians)
    net_longitudinal_force = (
        drive_force - brake_force - aerodynamic_drag - rolling_resistance
        - cornering_scrub - limiter_force - slope_force
    )
    # The model has no reverse gear, so a stationary car holds its position on a
    # gradient rather than rolling backwards into negative velocity.
    if velocity <= .02 and net_longitudinal_force < 0:
        net_longitudinal_force = 0.0
    longitudinal_acceleration = net_longitudinal_force / mass
    next_velocity = max(0.0, velocity + longitudinal_acceleration * dt)

    wheelbase = vehicle.wheelbase_m
    cg_to_rear = vehicle.front_weight_fraction * wheelbase
    stiffness_total = (
        vehicle.front_cornering_stiffness_n_per_rad
        + vehicle.rear_cornering_stiffness_n_per_rad
    )
    stiffness_front_fraction = vehicle.front_cornering_stiffness_n_per_rad / stiffness_total
    understeer_gradient = _clamp(
        (vehicle.front_weight_fraction - stiffness_front_fraction) * .12,
        -.018, .035,
    )
    lateral_transfer = abs(state.lateral_acceleration_mps2) * mass * vehicle.center_of_mass_height_m / vehicle.width_m
    transfer_ratio = min(1.0, lateral_transfer / max(1.0, normal_load * .5))
    load_sensitivity = max(.55, 1.0 - vehicle.tire_load_sensitivity * transfer_ratio)
    # The lateral limit is friction times normal load per unit mass. Using gravity
    # alone made downforce raise longitudinal traction while leaving cornering
    # untouched, which inverts the entire point of an aero package. With no
    # downforce and a level road this reduces exactly to `lateral_mu * gravity`.
    lateral_load_per_mass = normal_load / mass
    max_lateral_acceleration = lateral_mu * lateral_load_per_mass * load_sensitivity * _banking_gain(
        bank_radians, lateral_mu,
    )

    if next_velocity < .08:
        desired_yaw_rate = 0.0
        desired_slip = 0.0
    else:
        desired_yaw_rate = (
            next_velocity / wheelbase * math.tan(steering_angle)
            / max(.35, 1.0 + understeer_gradient * next_velocity * next_velocity)
        )
        desired_yaw_rate = _clamp(
            desired_yaw_rate,
            -max_lateral_acceleration / next_velocity,
            max_lateral_acceleration / next_velocity,
        )
        desired_slip = math.atan((cg_to_rear / wheelbase) * math.tan(steering_angle))
        # A rear axle with less cornering authority develops more transient
        # rotation/slip, making weight distribution and axle tires observable.
        rear_stiffness_fraction = 1.0 - stiffness_front_fraction
        oversteer_bias = max(0.0, vehicle.front_weight_fraction - stiffness_front_fraction)
        oversteer_bias += max(0.0, .5 - rear_stiffness_fraction) * .35
        desired_slip += steering_angle * oversteer_bias

    yaw_response = _clamp(
        stiffness_total * lateral_mu / (vehicle.yaw_inertia_kg_m2 * max(next_velocity, 3.0)) * dt,
        0.0, 1.0,
    )
    yaw_rate = state.yaw_rate_radians_per_second + (
        desired_yaw_rate - state.yaw_rate_radians_per_second
    ) * yaw_response
    slip_response = _clamp(
        stiffness_total * lateral_mu / (mass * max(next_velocity, 3.0)) * dt,
        0.0, 1.0,
    )
    slip_angle = state.slip_angle_radians + (desired_slip - state.slip_angle_radians) * slip_response
    lateral_velocity = math.tan(slip_angle) * next_velocity
    # Centripetal acceleration is the stable force-budget quantity. The slip
    # state already has its own stiffness-limited transient, so differentiating
    # it again here would double-count the tire response and suppress yaw.
    lateral_acceleration = next_velocity * yaw_rate
    if abs(lateral_acceleration) > max_lateral_acceleration:
        scale = max_lateral_acceleration / abs(lateral_acceleration)
        yaw_rate *= scale
        lateral_velocity *= scale
        slip_angle = math.atan2(lateral_velocity, max(next_velocity, .01))
        lateral_acceleration = math.copysign(max_lateral_acceleration, lateral_acceleration)

    heading = (state.heading_radians + yaw_rate * dt) % math.tau
    forward_x, forward_y = math.cos(heading), math.sin(heading)
    side_x, side_y = -forward_y, forward_x
    x = state.x + (
        forward_x * next_velocity + side_x * lateral_velocity
    ) * dt * dynamics.pixels_per_meter
    y = state.y + (
        forward_y * next_velocity + side_y * lateral_velocity
    ) * dt * dynamics.pixels_per_meter
    return VehiclePhysicsState(
        x=x, y=y, heading_radians=heading,
        longitudinal_velocity_mps=next_velocity,
        lateral_velocity_mps=lateral_velocity,
        yaw_rate_radians_per_second=yaw_rate,
        steering_angle_radians=steering_angle,
        longitudinal_acceleration_mps2=longitudinal_acceleration,
        lateral_acceleration_mps2=lateral_acceleration,
        slip_angle_radians=slip_angle,
        aerodynamic_drag_n=aerodynamic_drag,
        rolling_resistance_n=rolling_resistance,
        lateral_load_transfer_n=lateral_transfer,
    )


def _banking_gain(bank_radians: float, lateral_mu: float) -> float:
    """How much extra cornering a banked road allows, as a grip multiplier.

    On a bank of angle t the limit is `g(mu cos t + sin t) / (cos t - mu sin t)`,
    so the gain over flat ground is that expression divided by `g mu`. Tracks are
    compiled banked in the direction their corner turns, so the magnitude is what
    matters; the result is clamped because the exact form diverges as the
    denominator approaches zero.
    """
    angle = abs(bank_radians)
    if angle < 1e-9:
        return 1.0
    cos_bank, sin_bank = math.cos(angle), math.sin(angle)
    denominator = cos_bank - lateral_mu * sin_bank
    if denominator <= .05:
        return 1.75
    limit = (lateral_mu * cos_bank + sin_bank) / denominator
    return _clamp(limit / max(1e-6, lateral_mu), 1.0, 1.75)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
