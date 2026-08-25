import { useRef, useState } from "react";
import { api } from "./api";
import { WorldView } from "./WorldView";
import { isElevated } from "./View3D";
import { Fidelity } from "./Fidelity";
import type { Environment } from "./types";

/** Look at a compiled circuit. Nothing else.
 *
 *  A 3D circuit gets a free-floating camera orbiting it — drag to swing round, scroll to pull
 *  back — because the question here is what shape this circuit is and where the hills sit, and
 *  no car-relative camera can answer that: a driving camera only looks where the car points.
 *  A planar circuit gets the top-down plan, which is already the whole picture.
 */
export function EnvironmentViewer({ environment }: { environment: Environment }) {
  if (!isElevated(environment.scene)) {
    return <div className="viewer">
      <WorldView scene={environment.scene} />
      <SceneFacts environment={environment} />
      <Fidelity environment={environment} />
    </div>;
  }
  return <OrbitViewer environment={environment} />;
}

/** Quantised so a drag does not fire a render per pixel: the pose the server is asked for
 *  moves in steps, while the pointer moves continuously. */
const YAW_STEP = 3;
const PITCH_STEP = 2;
const DISTANCE_STEP = 40;
const quantise = (value: number, step: number) => Math.round(value / step) * step;

function OrbitViewer({ environment }: { environment: Environment }) {
  const [yaw, setYaw] = useState(45);
  const [pitch, setPitch] = useState(32);
  const [distance, setDistance] = useState(1_150);
  const [focus, setFocus] = useState<"circuit" | "car">("circuit");
  const [plan, setPlan] = useState(false);
  const dragging = useRef<{ x: number; y: number } | null>(null);

  const src = api.environmentView3dUrl(environment.id, "free", {
    yaw: quantise(yaw, YAW_STEP),
    pitch: quantise(pitch, PITCH_STEP),
    distance: quantise(distance, DISTANCE_STEP),
    focus,
  });

  return <div className="viewer">
    <div className="viewer-bar">
      <div className="camera-picker" role="group" aria-label="View">
        <button type="button" className={plan ? "" : "active"} onClick={() => setPlan(false)}>3D</button>
        <button type="button" className={plan ? "active" : ""} onClick={() => setPlan(true)}>Plan</button>
      </div>
      {!plan && <div className="camera-picker" role="group" aria-label="Focus">
        <button type="button" className={focus === "circuit" ? "active" : ""} onClick={() => setFocus("circuit")}>Circuit</button>
        <button type="button" className={focus === "car" ? "active" : ""} onClick={() => setFocus("car")}>Car</button>
      </div>}
      {!plan && <span className="viewer-hint">drag to orbit · scroll to zoom</span>}
    </div>
    {plan ? <WorldView scene={environment.scene} /> : <img
      className="orbit-frame"
      src={src}
      alt={`3D view of ${environment.scene.name}`}
      draggable={false}
      onPointerDown={(event) => {
        dragging.current = { x: event.clientX, y: event.clientY };
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        const from = dragging.current;
        if (!from) return;
        setYaw((current) => current + (event.clientX - from.x) * 0.4);
        setPitch((current) => Math.max(4, Math.min(86, current + (event.clientY - from.y) * 0.3)));
        dragging.current = { x: event.clientX, y: event.clientY };
      }}
      onPointerUp={() => { dragging.current = null; }}
      onPointerCancel={() => { dragging.current = null; }}
      onWheel={(event) => {
        event.preventDefault();
        setDistance((current) => Math.max(160, Math.min(3_000, current + event.deltaY * 1.4)));
      }}
    />}
    <SceneFacts environment={environment} />
    <Fidelity environment={environment} />
  </div>;
}

function SceneFacts({ environment }: { environment: Environment }) {
  const scene = environment.scene;
  const certificate = environment.playability_certificate;
  const facts = [
    `${scene.surface} at ${scene.grip.toFixed(2)}× grip`,
    `${scene.laps} lap${scene.laps === 1 ? "" : "s"}`,
    `${scene.sector_count} gates`,
    `start ${scene.start_line_region && scene.start_line_region !== "auto" ? scene.start_line_region : "on the main straight"} · player P${scene.player_grid_position ?? 1}`,
    `corridor ${scene.track_width.toFixed(0)}px`,
    `${scene.npc_behaviors.length} opponent${scene.npc_behaviors.length === 1 ? "" : "s"}`,
    scene.elevation && scene.elevation.profile !== "flat"
      ? `${scene.elevation.profile}, ${scene.elevation.amplitude_m.toFixed(1)} m, banking to ${scene.elevation.banking_degrees.toFixed(0)}°`
      : "flat",
    certificate?.playable ? `lap verified in ${certificate.route_steps} ticks` : "not verified",
  ];
  return <p className="scene-facts">{facts.join(" · ")}</p>;
}
