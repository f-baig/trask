import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import type { TrackDrawing } from "./types";

type Point = { x: number; y: number };

function path(points: Point[], close = true) {
  if (!points.length) return "";
  return `M ${points.map((point) => `${point.x * 1000} ${point.y * 640}`).join(" L ")}${close ? " Z" : ""}`;
}

export function DrawView({ useDrawing }: { useDrawing(id: string): void }) {
  const [drawings, setDrawings] = useState<TrackDrawing[]>([]);
  const [points, setPoints] = useState<Point[]>([]);
  const [name, setName] = useState("Untitled circuit");
  const [drawing, setDrawing] = useState(false);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string>();
  const canvas = useRef<SVGSVGElement>(null);

  async function refresh() { setDrawings(await api.drawings()); }
  useEffect(() => { void refresh().catch((reason: Error) => setError(reason.message)); }, []);

  function pointFromEvent(event: React.PointerEvent<SVGSVGElement>): Point {
    const box = canvas.current!.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(1, (event.clientX - box.left) / box.width)),
      y: Math.max(0, Math.min(1, (event.clientY - box.top) / box.height)),
    };
  }

  function begin(event: React.PointerEvent<SVGSVGElement>) {
    event.currentTarget.setPointerCapture(event.pointerId);
    setDrawing(true);
    setPoints([pointFromEvent(event)]);
    setError(undefined);
  }

  function move(event: React.PointerEvent<SVGSVGElement>) {
    if (!drawing) return;
    const next = pointFromEvent(event);
    setPoints((current) => {
      const previous = current.at(-1);
      return previous && Math.hypot(next.x - previous.x, next.y - previous.y) < .006
        ? current : [...current, next];
    });
  }

  async function save() {
    if (points.length < 8) { setError("Draw one complete loop before saving."); return; }
    setSaving(true);
    setError(undefined);
    try {
      const saved = await api.createDrawing(name.trim() || "Untitled circuit", points);
      await refresh();
      setName(saved.name);
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : "Could not save drawing");
    } finally { setSaving(false); }
  }

  async function remove(item: TrackDrawing) {
    if (!window.confirm(`Delete /${item.id}? This does not delete circuits already compiled from it.`)) return;
    await api.deleteDrawing(item.id);
    await refresh();
  }

  return <section className="draw-workspace">
    <aside className="draw-library">
      <div className="eyebrow">Saved drawings</div>
      {drawings.length === 0 ? <p className="muted">No sketches yet.</p> : drawings.map((item) => <article key={item.id}>
        <svg viewBox="0 0 1000 640" aria-hidden="true"><path d={path(item.points)} /></svg>
        <div><b>{item.name}</b><code>/{item.id}</code></div>
        <div className="draw-item-actions">
          <button onClick={() => useDrawing(item.id)}>Use</button>
          <button onClick={() => void remove(item)}>Delete</button>
        </div>
      </article>)}
    </aside>
    <section className="draw-stage">
      <header>
        <div><div className="eyebrow">Centerline sketch</div><h1>Draw a closed loop</h1></div>
        <p>One stroke is enough. The compiler closes, smooths, and certifies it when you use the saved reference.</p>
      </header>
      <div className="draw-canvas-shell">
        <svg
          ref={canvas} className="draw-canvas" viewBox="0 0 1000 640"
          onPointerDown={begin} onPointerMove={move}
          onPointerUp={() => setDrawing(false)} onPointerCancel={() => setDrawing(false)}
          role="img" aria-label="Track drawing canvas"
        >
          <defs><pattern id="draw-grid" width="40" height="40" patternUnits="userSpaceOnUse"><path d="M 40 0 L 0 0 0 40" /></pattern></defs>
          <rect width="1000" height="640" className="draw-grid" />
          {points.length > 1 && <>
            <path className="draw-road-preview" d={path(points, !drawing)} />
            <path className="draw-stroke" d={path(points, !drawing)} />
            {drawing && <line className="draw-closure" x1={points[0].x * 1000} y1={points[0].y * 640} x2={points.at(-1)!.x * 1000} y2={points.at(-1)!.y * 640} />}
          </>}
          {points.length === 0 && <text x="500" y="320">PRESS + DRAG TO DRAW</text>}
        </svg>
        <span className="draw-canvas-index">1000 × 640 normalized canvas</span>
      </div>
      <div className="draw-controls">
        <label><span>Name</span><input value={name} maxLength={64} onChange={(event) => setName(event.target.value)} /></label>
        <span>{points.length} samples</span>
        <button onClick={() => setPoints((current) => current.slice(0, Math.max(0, current.length - 12)))} disabled={!points.length}>Undo tail</button>
        <button onClick={() => setPoints([])} disabled={!points.length}>Clear</button>
        <button className="primary" onClick={() => void save()} disabled={saving || points.length < 8}>{saving ? "Saving…" : "Save drawing"}</button>
      </div>
      {error && <p className="draw-error">{error}</p>}
      <p className="draw-instruction">After saving, send <code>use /drawing-id</code> in Coordinator. You can add normal options too, such as laps, surface, barriers, or 3D.</p>
    </section>
  </section>;
}
