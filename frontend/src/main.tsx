import { StrictMode } from "react";
import { createRoot, type Root } from "react-dom/client";
import App from "./App";

/**
 * Vite may re-evaluate the entry module while a researcher is leaving the UI
 * open. Reusing this root is important: creating a second root in #root leaves
 * the previous App alive with its own selected circuit and experiment controls.
 */
type RaceLabWindow = typeof globalThis & { __raceLabRoot?: Root };
const raceLabWindow = globalThis as RaceLabWindow;
const container = document.getElementById("root");

if (!container) throw new Error("RaceLab could not find its application root.");
const root = raceLabWindow.__raceLabRoot ?? createRoot(container);
raceLabWindow.__raceLabRoot = root;
root.render(<StrictMode><App /></StrictMode>);

if (import.meta.hot) {
  import.meta.hot.dispose(() => {
    root.unmount();
    delete raceLabWindow.__raceLabRoot;
  });
}
