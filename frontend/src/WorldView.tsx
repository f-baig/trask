import type { Entity, Frame, Scene } from "./types";

type Point = { x: number; y: number };
type RuntimeEntity = Frame["privileged_state"]["entities"][number];

const surfacePalette = {
  asphalt: { ground: "#3f6746", road: "#343a40" },
  clay: { ground: "#657343", road: "#9b6544" },
  ice: { ground: "#b8d3d8", road: "#a9c5cc" },
} as const;

/** Nudge a hex colour toward white or black.
 *
 *  The renderer needs a lighter and a darker tone of whatever the road and ground turn
 *  out to be — for texture speckle, run-off, and lane sheen. Deriving them means a
 *  recoloured circuit still reads as one material rather than as a flat fill, and it is
 *  why the visual plan carries one colour per surface instead of five. */
function tone(hex: string, factor: number) {
  const value = hex.replace("#", "");
  const channels = [0, 2, 4].map((offset) => parseInt(value.slice(offset, offset + 2), 16) || 0);
  const shifted = channels.map((channel) => Math.max(0, Math.min(255, Math.round(
    factor >= 1 ? channel + (255 - channel) * (factor - 1) : channel * factor,
  ))));
  return `#${shifted.map((channel) => channel.toString(16).padStart(2, "0")).join("")}`;
}

/** The scene's own palette, falling back to the surface default for anything the
 *  brief never mentioned. Unset stays exactly as it looked before visual plans existed. */
function palette(scene: Scene) {
  const base = surfacePalette[scene.surface ?? "asphalt"];
  const visual = scene.visual ?? {};
  const ground = visual.terrain || base.ground;
  const road = visual.road || base.road;
  return {
    ground,
    grassLight: tone(ground, 1.22),
    road,
    roadLight: tone(road, 1.28),
    runoff: tone(road, 0.62),
    barrier: visual.barrier || "#ef4c3f",
    playerCar: visual.player_car || "#f4f0e5",
    opponentCar: visual.opponent_car || "#32a6e6",
    kerbs: visual.kerbs !== false,
    kerbLight: visual.kerb_light || "#f4eee0",
    kerbDark: visual.kerb_dark || "#d7493f",
    scenery: visual.scenery ?? [],
  };
}

/** Where one of the nine named track regions sits, in scene coordinates. Mirrors the
 *  backend's `TrackRegion`, which is the same nine-cell grid the corner grammar uses. */
function regionCentre(region: string, bounds: { width: number; height: number }) {
  const columns: Record<string, number> = { left: 0.5, right: 2.5, center: 1.5 };
  const rows: Record<string, number> = { top: 0.5, bottom: 2.5, center: 1.5 };
  const [first, second] = region.split("-");
  const column = second ? columns[second] ?? 1.5 : columns[first] ?? 1.5;
  const row = second ? rows[first] ?? 1.5 : rows[first] ?? 1.5;
  return { x: (bounds.width / 3) * column, y: (bounds.height / 3) * row };
}

function closedCurve(points: Point[]) {
  if (points.length < 3) return points.length ? `M ${points[0].x} ${points[0].y}` : "";
  const commands = [`M ${points[0].x} ${points[0].y}`];
  for (let index = 0; index < points.length; index += 1) {
    const previous = points[(index - 1 + points.length) % points.length];
    const current = points[index];
    const next = points[(index + 1) % points.length];
    const after = points[(index + 2) % points.length];
    commands.push(`C ${current.x + (next.x - previous.x) / 6} ${current.y + (next.y - previous.y) / 6}, ${next.x - (after.x - current.x) / 6} ${next.y - (after.y - current.y) / 6}, ${next.x} ${next.y}`);
  }
  return `${commands.join(" ")} Z`;
}

function trackEdges(scene: Scene, offset = scene.track_width / 2 + 3) {
  const points = scene.track_centerline ?? [];
  const left: Point[] = [];
  const right: Point[] = [];
  points.forEach((current, index) => {
    const before = points[(index - 1 + points.length) % points.length];
    const after = points[(index + 1) % points.length];
    const tx = after.x - before.x;
    const ty = after.y - before.y;
    const length = Math.max(1e-9, Math.hypot(tx, ty));
    const nx = -ty / length;
    const ny = tx / length;
    left.push({ x: current.x + nx * offset, y: current.y + ny * offset });
    right.push({ x: current.x - nx * offset, y: current.y - ny * offset });
  });
  return { left, right };
}

function nearestTrackAngle(point: Point, track: Point[]) {
  if (track.length < 2) return 0;
  let nearest = 0;
  let distance = Number.POSITIVE_INFINITY;
  track.forEach((candidate, index) => {
    const candidateDistance = (candidate.x - point.x) ** 2 + (candidate.y - point.y) ** 2;
    if (candidateDistance < distance) {
      distance = candidateDistance;
      nearest = index;
    }
  });
  const before = track[(nearest - 1 + track.length) % track.length];
  const after = track[(nearest + 1) % track.length];
  return Math.atan2(after.y - before.y, after.x - before.x) * 180 / Math.PI;
}

function RaceCar({ x, y, heading, color, accent, number, player = false, speed = 0, nitro = false }: {
  x: number; y: number; heading: number; color: string; accent: string; number: string; player?: boolean; speed?: number; nitro?: boolean;
}) {
  return <g className={`race-car ${player ? "player-car" : "opponent-car"}`} transform={`translate(${x} ${y}) rotate(${heading})`} aria-label={`${player ? "Player" : "Opponent"} car pointing ${Math.round(heading)} degrees`}>
    {player && <circle className="player-marker" r="25" />}
    {nitro && <g className="nitro-flame">
      <path d="M -21 -5 C -36 -8 -43 -3 -53 0 C -43 3 -36 8 -21 5 Z" fill="#35d9ff" />
      <path d="M -22 -2 C -34 -3 -39 -1 -44 0 C -39 1 -34 3 -22 2 Z" fill="#f5fdff" />
    </g>}
    {speed > 3 && <g className="speed-streaks" opacity={Math.min(.7, speed / 12)}>
      <path d="M -23 -7 H -38" /><path d="M -25 0 H -44" /><path d="M -23 7 H -36" />
    </g>}
    <ellipse cx="-1" cy="4" rx="23" ry="13" fill="#101418" opacity=".35" filter="url(#carShadow)" />
    <g className="wheels" fill="#111416">
      <rect x="-15" y="-13" width="10" height="6" rx="2" /><rect x="-15" y="7" width="10" height="6" rx="2" />
      <rect x="9" y="-13" width="9" height="6" rx="2" /><rect x="9" y="7" width="9" height="6" rx="2" />
    </g>
    <path d="M -20 -9 Q -22 0 -20 9 L 8 10 Q 18 8 22 0 Q 18 -8 8 -10 Z" fill={color} stroke="#11171a" strokeWidth="2" />
    <path d="M -18 -7 L -7 -7 L -4 7 L -18 7 Q -20 0 -18 -7 Z" fill={accent} opacity=".88" />
    <path d="M -5 -7 L 7 -8 L 13 -4 L 13 4 L 7 8 L -5 7 Z" fill="#17252d" stroke="#b9d8df" strokeWidth="1.2" />
    <path d="M 8 -7 L 14 -4 L 14 4 L 8 7 L 10 0 Z" fill="#79a7b5" opacity=".85" />
    <path d="M -5 -7 L -2 -2 L -2 2 L -5 7" fill="none" stroke="#d8eef1" strokeWidth="1" opacity=".7" />
    <rect x="20" y="-9" width="3" height="18" rx="1.5" fill={accent} />
    <path d="M 18 -6 L 22 -5 M 18 6 L 22 5" stroke="#fff4bb" strokeWidth="2.4" strokeLinecap="round" />
    <circle cx="-11" cy="0" r="5.5" fill="#f6f0dc" stroke="#182126" strokeWidth="1" />
    <text x="-11" y="2.6" textAnchor="middle" fill="#182126" fontSize="7" fontWeight="800">{number}</text>
    {player && <path className="nose-pointer" d="M 28 0 L 35 -4 L 35 4 Z" />}
  </g>;
}

function Barrier({ entity, spec, color = "#ef4c3f" }: { entity: RuntimeEntity; spec?: Entity; color?: string }) {
  const x = entity.x + entity.width / 2;
  const y = entity.y + entity.height / 2;
  const shape = spec?.shape ?? "box";
  const angle = shape === "oriented-box" ? spec?.rotation_degrees ?? 0 : 0;
  const radius = Math.min(entity.width, entity.height) / 2;
  return <g className={`track-barrier ${shape}`} transform={`translate(${x} ${y}) rotate(${angle})`} aria-label={`${shape} barrier`}>
    {shape === "circle" ? <>
      <circle r={radius} fill={color} stroke="#161b1d" strokeWidth="2" />
      <circle r={radius * .57} fill="#202629" stroke="#f2eee2" strokeWidth="1.5" />
      <circle r={radius * .2} fill="#8f9899" />
    </> : <>
      <rect x={-entity.width / 2} y={-entity.height / 2} width={entity.width} height={entity.height} rx="1.5" fill={color} stroke="#161b1d" strokeWidth="2" />
      <path d={`M ${-entity.width / 2 + 4} ${entity.height / 2} L ${-entity.width / 2 + 12} ${-entity.height / 2} M ${entity.width / 2 - 12} ${entity.height / 2} L ${entity.width / 2 - 4} ${-entity.height / 2}`} stroke="#f2eee2" strokeWidth="3" opacity=".8" />
    </>}
  </g>;
}

export function WorldView({ scene, frame, trajectory = [], onMapClick, selectedPoint }: { scene: Scene; frame?: Frame; trajectory?: Frame[]; onMapClick?: (point: Point) => void; selectedPoint?: Point }) {
  const entities: RuntimeEntity[] = frame ? frame.privileged_state.entities : scene.entities.map((entity) => ({
    id: entity.id, kind: entity.kind, x: entity.rect.x, y: entity.rect.y, width: entity.rect.width, height: entity.rect.height, active: true, open: false,
  }));
  const player = frame?.privileged_state.player ?? scene.player_spawn;
  const heading = frame?.privileged_state.heading ?? nearestTrackAngle(player, scene.track_centerline);
  const speed = frame?.privileged_state.speed ?? 0;
  const nitro = frame?.privileged_state.nitro ?? 0;
  const nitroActive = frame?.privileged_state.nitro_active ?? false;
  const nitroReady = frame?.privileged_state.nitro_ready ?? false;
  const turning = frame?.privileged_state.turning ?? false;
  const track = scene.track_centerline ?? [];
  const trackPath = closedCurve(track);
  const guardrails = scene.edge_barriers ? trackEdges(scene) : null;
  const surface = palette(scene);
  const checkpoints = entities.filter((entity) => entity.active && entity.kind === "checkpoint");
  const opponents = entities.filter((entity) => entity.active && entity.kind === "npc");
  const obstacles = entities.filter((entity) => entity.active && entity.kind === "obstacle");
  const sceneEntities = new Map(scene.entities.map((entity) => [entity.id, entity]));
  const directionMarkers = track.filter((_, index) => index % Math.max(1, Math.floor(track.length / 7)) === 0).slice(0, 7);
  const countdownTicks = frame?.privileged_state.countdown_ticks_remaining ?? 0;
  const startSignal = countdownTicks > 0 ? `${Math.ceil(countdownTicks / 10)}` : frame?.events.includes("go") ? "GO" : null;
  const trajectoryPoints = trajectory.map((item) => `${item.privileged_state.player.x},${item.privileged_state.player.y}`).join(" ");

  return (
    <svg className={`world race-world${onMapClick ? " selectable-map" : ""}`} viewBox={`0 0 ${scene.bounds.width} ${scene.bounds.height}`} role="img" aria-label="Top-down racing circuit replay" onClick={(event) => {
      if (!onMapClick) return;
      const bounds = event.currentTarget.getBoundingClientRect();
      onMapClick({ x: (event.clientX - bounds.left) / bounds.width * scene.bounds.width, y: (event.clientY - bounds.top) / bounds.height * scene.bounds.height });
    }}>
      <defs>
        <pattern id="grassTexture" width="34" height="34" patternUnits="userSpaceOnUse" patternTransform="rotate(25)">
          <rect width="34" height="34" fill={surface.ground} /><path d="M 0 7 H 34 M 0 24 H 34" stroke={surface.grassLight} strokeWidth="1.2" opacity=".2" />
        </pattern>
        <pattern id="asphaltTexture" width="22" height="22" patternUnits="userSpaceOnUse">
          <rect width="22" height="22" fill={surface.road} /><circle cx="4" cy="7" r=".8" fill={surface.roadLight} opacity=".32" /><circle cx="16" cy="15" r=".6" fill="#11181c" opacity=".25" />
        </pattern>
        <filter id="trackShadow" x="-20%" y="-20%" width="140%" height="140%"><feGaussianBlur stdDeviation="7" /></filter>
        <filter id="carShadow" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="3" /></filter>
        <pattern id="finishPattern" width="12" height="12" patternUnits="userSpaceOnUse">
          <rect width="6" height="6" fill="#f7f1df" /><rect x="6" y="6" width="6" height="6" fill="#f7f1df" /><rect x="6" width="6" height="6" fill="#171d20" /><rect y="6" width="6" height="6" fill="#171d20" />
        </pattern>
      </defs>

      <rect width={scene.bounds.width} height={scene.bounds.height} fill="url(#grassTexture)" />
      {/* Ground features — a river, a sand trap — drawn before the road, so the circuit
          passes over them exactly as the description says. */}
      {surface.scenery.map((band, index) => {
        const horizontal = band.orientation !== "vertical";
        const centre = regionCentre(band.region, scene.bounds);
        return <rect
          key={`scenery-${index}`}
          x={horizontal ? 0 : centre.x - band.width_pixels / 2}
          y={horizontal ? centre.y - band.width_pixels / 2 : 0}
          width={horizontal ? scene.bounds.width : band.width_pixels}
          height={horizontal ? band.width_pixels : scene.bounds.height}
          fill={band.color} opacity=".85"
        />;
      })}
      <path d={trackPath} fill="none" stroke="#101719" strokeWidth={scene.track_width + 34} strokeLinecap="round" strokeLinejoin="round" opacity=".3" filter="url(#trackShadow)" />
      <path d={trackPath} fill="none" stroke={surface.runoff} strokeWidth={scene.track_width + 25} strokeLinecap="round" strokeLinejoin="round" />
      {surface.kerbs && <>
        <path d={trackPath} fill="none" stroke={surface.kerbLight} strokeWidth={scene.track_width + 15} strokeLinecap="round" strokeLinejoin="round" />
        <path d={trackPath} fill="none" stroke={surface.kerbDark} strokeWidth={scene.track_width + 15} strokeDasharray="18 18" strokeLinecap="butt" strokeLinejoin="round" />
      </>}
      <path d={trackPath} fill="none" stroke="url(#asphaltTexture)" strokeWidth={scene.track_width} strokeLinecap="round" strokeLinejoin="round" />
      <path d={trackPath} fill="none" stroke={surface.roadLight} strokeWidth={scene.track_width - 9} strokeLinecap="round" strokeLinejoin="round" opacity=".16" />
      <path d={trackPath} fill="none" stroke="#e8ece6" strokeWidth="2.4" strokeDasharray="11 17" strokeLinecap="round" opacity=".65" />
      {guardrails && [guardrails.left, guardrails.right].map((points, side) => {
        const path = closedCurve(points);
        const stride = Math.max(1, Math.floor(points.length / 30));
        return <g key={`edge-guardrail-${side}`} className="edge-guardrail" aria-label={`${side ? "Right" : "Left"} edge guardrail`}>
          <path d={path} fill="none" stroke="#11171a" strokeWidth="10" strokeLinecap="round" strokeLinejoin="round" />
          <path d={path} fill="none" stroke={surface.barrier} strokeWidth="6" strokeLinecap="round" strokeLinejoin="round" />
          {points.filter((_, index) => index % stride === 0).map((point, index) => <circle key={index} cx={point.x} cy={point.y} r="2.2" fill="#f6efde" />)}
        </g>;
      })}

      {trajectoryPoints && <>
        <polyline className="run-trajectory-shadow" points={trajectoryPoints} />
        <polyline className="run-trajectory" points={trajectoryPoints} />
      </>}
      {selectedPoint && <circle className="trajectory-selection" cx={selectedPoint.x} cy={selectedPoint.y} r="14" />}

      {directionMarkers.map((point, index) => <g key={`direction-${index}`} transform={`translate(${point.x} ${point.y}) rotate(${nearestTrackAngle(point, track)})`} opacity=".48">
        <path d="M -5 -7 L 5 0 L -5 7" fill="none" stroke="#f5f2df" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round" />
      </g>)}

      {checkpoints.map((entity, index) => {
        const center = { x: entity.x + entity.width / 2, y: entity.y + entity.height / 2 };
        const angle = nearestTrackAngle(center, track);
        const finish = entity.id === "finish-line";
        return <g key={entity.id} transform={`translate(${center.x} ${center.y}) rotate(${angle})`} opacity={entity.open ? .3 : 1}>
          <rect x="-8" y={-scene.track_width / 2} width="16" height={scene.track_width} fill={finish ? "url(#finishPattern)" : "#f4cc45"} opacity={finish ? .95 : .3} />
          {!finish && <path d={`M 0 ${-scene.track_width / 2 + 4} V ${scene.track_width / 2 - 4}`} stroke="#ffe785" strokeWidth="3" strokeDasharray="7 7" />}
          <g transform={`translate(0 ${-scene.track_width / 2 - 17}) rotate(${-angle})`}><circle r="12" fill="#182024" stroke="#f5ecd6" strokeWidth="2" /><text y="4" textAnchor="middle" fill="#f5ecd6" fontSize="10" fontWeight="800">{finish ? "F" : index + 1}</text></g>
        </g>;
      })}

      {obstacles.map((entity) => <Barrier key={entity.id} entity={entity} spec={sceneEntities.get(entity.id)} color={surface.barrier} />)}

      {opponents.map((entity, index) => {
        const center = { x: entity.x + entity.width / 2, y: entity.y + entity.height / 2 };
        return <RaceCar key={entity.id} x={center.x} y={center.y} heading={entity.heading ?? nearestTrackAngle(center, track)} color={surface.opponentCar} accent="#f4f0e5" number={`${index + 2}`} speed={entity.speed ?? 0} nitro={entity.nitro_active ?? false} />;
      })}
      <RaceCar x={player.x} y={player.y} heading={heading} color={surface.playerCar} accent="#ff5a36" number="1" player speed={speed} nitro={nitroActive} />
      {frame?.privileged_state.barrier_impact && <g className="barrier-impact" transform={`translate(${frame.privileged_state.barrier_impact.x} ${frame.privileged_state.barrier_impact.y})`} aria-label="Barrier impact">
        <circle r="18" fill="none" stroke="#ffd854" strokeWidth="3" />
        {[0, 72, 144, 216, 288].map((angle) => <path key={angle} d="M 11 0 L 25 0" transform={`rotate(${angle})`} stroke="#fff1a8" strokeWidth="4" strokeLinecap="round" />)}
      </g>}

      <g className="race-telemetry">
        <rect x="18" y="18" width="242" height="62" rx="8" fill="#101619" opacity=".93" />
        <rect x="18" y="18" width="5" height="62" rx="2.5" fill="#ff5a36" />
        <text x="38" y="42" className="telemetry-kicker">RACELAB / LIVE CIRCUIT</text>
        <text x="38" y="65" className="telemetry-value">{trajectory.length ? `${trajectory.length} TICK PATH` : frame ? `${(frame.privileged_state.longitudinal_velocity_mps * 3.6).toFixed(0)} KM/H` : `${scene.surface.toUpperCase()}  ·  ${scene.dynamics.physics_hz} HZ PHYSICS`}</text>
        {frame && <text x="245" y="65" textAnchor="end" className="telemetry-sector">L{Math.min(scene.laps, frame.privileged_state.lap + 1)}/{scene.laps} · S{frame.privileged_state.lap >= scene.laps ? scene.sector_count : frame.privileged_state.objective_index % scene.sector_count + 1}</text>}
        {frame && <g className={nitroActive ? "nitro-meter active" : "nitro-meter"}>
          <rect x="280" y="18" width="180" height="62" rx="8" fill="#101619" opacity=".93" />
          <text x="298" y="42" className="telemetry-kicker">{nitroActive ? "NITRO BURN" : nitroReady ? "NITRO READY" : turning ? `${(Math.abs(frame.privileged_state.lateral_acceleration_mps2) / scene.dynamics.gravity_mps2).toFixed(2)}G · ${Math.abs(frame.privileged_state.slip_angle_degrees).toFixed(1)}° SLIP` : "NITRO CHARGING"}</text>
          <rect x="298" y="55" width="144" height="10" rx="5" fill="#28363c" />
          <rect x="298" y="55" width={144 * nitro / 100} height="10" rx="5" fill={nitroActive ? "#f4fbff" : "#42cde8"} />
        </g>}
      </g>
      {startSignal && <g className="start-signal" transform={`translate(${scene.bounds.width / 2 - 112} 34)`}>
        <rect width="224" height="86" rx="14" fill="#0b1114" opacity=".94" stroke="#f4f0e5" strokeWidth="2" />
        {[0, 1, 2].map((index) => {
          const lit = startSignal === "GO" ? index === 2 : Number(startSignal) === 3 - index;
          const colors = ["#df4039", "#efad35", "#39d276"];
          return <circle key={index} cx={53 + index * 59} cy="35" r="17" fill={lit ? colors[index] : "#253035"} stroke="#080d0f" strokeWidth="5" />;
        })}
        <text x="112" y="76" textAnchor="middle" fill="#f4f0e5" fontSize="18" fontWeight="900">{startSignal}</text>
      </g>}
    </svg>
  );
}
