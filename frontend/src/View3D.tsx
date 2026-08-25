import { useEffect, useState } from "react";
import type { Scene } from "./types";

export const CAMERAS = ["third-person", "first-person", "hood", "third-person-far", "overhead-3d"] as const;
export type Camera = (typeof CAMERAS)[number] | "plan";

const LABELS: Record<Camera, string> = {
  "third-person": "Chase",
  "first-person": "Cockpit",
  hood: "Bumper",
  "third-person-far": "Chase far",
  "overhead-3d": "Overhead",
  plan: "Plan",
};

export function isElevated(scene?: Scene): boolean {
  return Boolean(scene?.elevation && scene.elevation.profile !== "flat" && scene.elevation.amplitude_m > 0);
}

/** Perspective view of an elevated circuit, rendered by the harness and served as an image.
 *
 *  Deliberately not a WebGL scene: `view3d` already renders these cameras for the desktop
 *  player, a camera is a pure function of world state, and a second renderer in the browser
 *  could only add a second set of bugs to keep in sync with the physics. The trade is a
 *  request per frame, which is a few kilobytes on a local harness.
 *
 *  `plan` falls through to the caller's top-down view, because the layout of a circuit is
 *  still easier to read from above than from inside the car. */
export function View3D({ src, camera, setCamera, elevation, children }: {
  src: string;
  camera: Camera;
  setCamera(camera: Camera): void;
  elevation: NonNullable<Scene["elevation"]>;
  children: React.ReactNode;
}) {
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState<string>();

  // The previous frame stays on screen while the next one decodes, so scrubbing does not
  // strobe between the image and an empty box.
  useEffect(() => { setFailed(undefined); }, [src]);

  return <div className="view3d">
    <div className="view3d-bar">
      <div className="camera-picker" role="group" aria-label="Camera">
        {(["plan", ...CAMERAS] as Camera[]).map((option) => <button
          key={option} type="button" className={camera === option ? "active" : ""}
          onClick={() => setCamera(option)}
        >{LABELS[option]}</button>)}
      </div>
      <span className="view3d-note">
        {elevation.profile} · {elevation.amplitude_m.toFixed(1)} m over {elevation.hill_count} crest
        {elevation.hill_count === 1 ? "" : "s"} · banking to {elevation.banking_degrees.toFixed(0)}°
      </span>
    </div>
    {camera === "plan" ? children : failed ? <div className="empty-preview">{failed}</div> : <img
      className={loaded ? "view3d-frame" : "view3d-frame is-loading"}
      src={src} alt={`${LABELS[camera]} view of the circuit`}
      onLoad={() => setLoaded(true)}
      onError={() => setFailed("The harness could not render this frame. Is the API running?")}
    />}
  </div>;
}
