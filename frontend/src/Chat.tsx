import { useEffect, useRef, useState } from "react";
import { Markdown } from "./Markdown";
import type { AgentMessage, ArtifactLink, Dimensions, StepState, StreamEvent } from "./types";

/** One stage of harness work behind a reply. Folded away by default: a reader wants the
 *  answer, and the log is for when the answer is not enough. */
type Step = { id: string; label: string; state: StepState; logs: string[] };
type Progress = {
  completed: number; total: number; elapsedMs: number; etaMs?: number | null;
  label: string; receivedAt: number;
};

type Turn = {
  key: string;
  speaker: "user" | "assistant";
  text: string;
  steps: Step[];
  artifacts: ArtifactLink[];
  streaming: boolean;
  progress?: Progress;
  error?: string;
};

export type ChatSend = (
  message: string,
  dimensions: Dimensions,
) => AsyncGenerator<StreamEvent>;

export function ChatView({ title, subtitle, history, send, openArtifact, showDimensions = false, showStartControls = false, emptyHint, placeholder, conversationId = title, loadingHistory = false, draftRequest }: {
  title: string;
  subtitle: string;
  history: AgentMessage[];
  send: ChatSend;
  openArtifact?: (artifact: ArtifactLink) => void;
  showDimensions?: boolean;
  /** Circuit creation exposes the race-start contract without making users phrase it. */
  showStartControls?: boolean;
  emptyHint: string;
  placeholder?: string;
  conversationId?: string;
  loadingHistory?: boolean;
  draftRequest?: { key: number; text: string };
}) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [draft, setDraft] = useState("");
  const [dimensions, setDimensions] = useState<Dimensions>("2d");
  const [startRegion, setStartRegion] = useState("auto");
  const [playerGridPosition, setPlayerGridPosition] = useState(1);
  const [sending, setSending] = useState(false);
  const scroller = useRef<HTMLDivElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);
  const seededConversation = useRef<string | undefined>(undefined);
  const appliedHistory = useRef<string>("");

  useEffect(() => {
    if (!draftRequest) return;
    setDraft(draftRequest.text);
    requestAnimationFrame(() => composer.current?.focus());
  }, [draftRequest?.key]);

  // Reconstruct the transcript whenever its durable record changes. A conversation switch
  // resets immediately even if the next history request is still in flight, which prevents
  // one circuit's chat flashing into another circuit. While a turn streams, the live copy
  // wins; the persisted copy replaces it once the completed message arrives.
  useEffect(() => {
    const signature = history.map((item) => item.id).join("|");
    const changedConversation = seededConversation.current !== conversationId;
    if (!changedConversation && (sending || appliedHistory.current === signature)) return;
    seededConversation.current = conversationId;
    appliedHistory.current = signature;
    const persisted = history.map((item, index) => ({
      key: `${item.id}-${index}`, speaker: item.speaker, text: item.content,
      steps: (item.actions ?? []).map((action) => ({
        id: action.id, label: action.label, state: action.state, logs: action.logs,
      })),
      artifacts: item.artifacts, streaming: false,
    }));
    setTurns((current) => {
      if (changedConversation) return persisted;
      const localCompletedReply = [...current].reverse().find(
        (turn) => turn.speaker === "assistant" && turn.key.startsWith("reply-")
          && !turn.streaming && Boolean(turn.text || turn.error || turn.steps.length),
      );
      const persistenceStillEndsAtUser = history.at(-1)?.speaker === "user";
      // The backend commits a long-running assistant turn at completion. A refresh can land
      // in the narrow interval where the local stream is complete but that durable write is
      // not visible yet. Never replace the real answer with the shorter user-only snapshot;
      // the next history signature will reconcile it once persistence catches up.
      if (localCompletedReply && persistenceStillEndsAtUser && persisted.length < current.length) {
        return current;
      }
      return persisted;
    });
  }, [conversationId, history, sending]);

  useEffect(() => {
    const node = scroller.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [turns]);

  function patch(key: string, update: (turn: Turn) => Turn) {
    setTurns((current) => current.map((turn) => (turn.key === key ? update(turn) : turn)));
  }

  async function submit() {
    const message = draft.trim();
    if (!message || sending) return;
    const startContract = showStartControls && (startRegion !== "auto" || playerGridPosition !== 1)
      ? `\n\nRace start: start/finish in the ${startRegion.replaceAll("-", " ")}; player starts P${playerGridPosition}.`
      : "";
    const configuredMessage = `${message}${startContract}`;
    const stamp = Date.now();
    const replyKey = `reply-${stamp}`;
    setDraft("");
    setSending(true);
    setTurns((current) => [
      ...current,
      { key: `you-${stamp}`, speaker: "user", text: configuredMessage, steps: [], artifacts: [], streaming: false },
      { key: replyKey, speaker: "assistant", text: "", steps: [], artifacts: [], streaming: true },
    ]);
    try {
      for await (const event of send(configuredMessage, dimensions)) {
        if (event.type === "token") {
          patch(replyKey, (turn) => ({ ...turn, text: turn.text + event.text }));
        } else if (event.type === "step") {
          patch(replyKey, (turn) => {
            const steps = turn.steps.some((step) => step.id === event.id)
              ? turn.steps.map((step) => step.id === event.id ? { ...step, label: event.label, state: event.state } : step)
              : [...turn.steps, { id: event.id, label: event.label, state: event.state, logs: [] }];
            const artifacts = event.artifact && !turn.artifacts.some((item) => item.kind === event.artifact?.kind && item.id === event.artifact?.id)
              ? [...turn.artifacts, event.artifact] : turn.artifacts;
            return { ...turn, steps, artifacts };
          });
        } else if (event.type === "progress") {
          patch(replyKey, (turn) => ({
            ...turn,
            progress: {
              completed: event.completed, total: event.total,
              elapsedMs: event.elapsed_ms, etaMs: event.eta_ms,
              label: event.label, receivedAt: Date.now(),
            },
          }));
        } else if (event.type === "log") {
          patch(replyKey, (turn) => {
            const log = `${event.stage}: ${event.detail}`;
            const steps = turn.steps.some((step) => step.id === event.id)
              ? turn.steps.map((step) => step.id === event.id ? { ...step, logs: [...step.logs, log] } : step)
              : [...turn.steps, { id: event.id, label: event.stage.replaceAll("-", " "), state: "done" as const, logs: [log] }];
            return { ...turn, steps };
          });
        } else if (event.type === "done") {
          patch(replyKey, (turn) => ({
            ...turn, streaming: false,
            text: turn.text || event.result?.summary || event.content || "",
            artifacts: event.artifacts?.length ? event.artifacts : turn.artifacts,
          }));
        } else if (event.type === "error") {
          patch(replyKey, (turn) => ({ ...turn, streaming: false, error: event.detail }));
        }
      }
      patch(replyKey, (turn) => ({ ...turn, streaming: false }));
    } catch (reason) {
      const detail = reason instanceof Error ? reason.message : "Stream failed";
      patch(replyKey, (turn) => ({ ...turn, streaming: false, error: detail }));
    } finally {
      setSending(false);
      composer.current?.focus();
    }
  }

  return <section className="chat-view">
    <div className="chat-scroll" ref={scroller}>
      <div className="chat-column">
        {turns.length === 0 && loadingHistory && <div className="chat-history-loading" role="status">
          <span className="step-dot running" /> Loading conversation…
        </div>}
        {turns.length === 0 && !loadingHistory && <div className="chat-empty">
          <h1>{title}</h1>
          <p>{subtitle}</p>
          <p className="chat-hint">{emptyHint}</p>
        </div>}
        {turns.map((turn) => <TurnView key={turn.key} turn={turn} openArtifact={openArtifact} />)}
      </div>
    </div>
    <div className="chat-dock">
      <div className="chat-column">
        <form className="composer" onSubmit={(event) => { event.preventDefault(); void submit(); }}>
          <textarea
            ref={composer}
            value={draft}
            rows={1}
            placeholder={sending ? "Working…" : placeholder ?? "Describe a circuit and the driver you want"}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); void submit(); }
            }}
          />
          <div className="composer-actions">
            {showDimensions && <div className="dimension-toggle" role="group" aria-label="Engine dimension">
              {(["2d", "3d"] as Dimensions[]).map((option) => <button
                key={option} type="button" className={dimensions === option ? "active" : ""}
                onClick={() => setDimensions(option)}
              >{option.toUpperCase()}</button>)}
            </div>}
            <button className="primary send" type="submit" disabled={sending || !draft.trim()} aria-label="Send">
              {sending ? "···" : "↑"}
            </button>
          </div>
        </form>
        {showStartControls && <div className="start-controls" aria-label="Race start settings">
          <label>Start / finish
            <select value={startRegion} onChange={(event) => setStartRegion(event.target.value)}>
              <option value="auto">Main straight (auto)</option>
              <option value="top-left">Top left</option>
              <option value="top-center">Top centre</option>
              <option value="top-right">Top right</option>
              <option value="left">Left</option>
              <option value="center">Centre</option>
              <option value="right">Right</option>
              <option value="bottom-left">Bottom left</option>
              <option value="bottom-center">Bottom centre</option>
              <option value="bottom-right">Bottom right</option>
            </select>
          </label>
          <label>Player grid
            <select value={playerGridPosition} onChange={(event) => setPlayerGridPosition(Number(event.target.value))}>
              {[1, 2, 3, 4, 5, 6].map((position) => <option key={position} value={position}>P{position}</option>)}
            </select>
          </label>
          <small>Grid cars line up behind the selected finish gate.</small>
        </div>}
        <p className="composer-note">
          Enter sends · Shift+Enter for a new line
          {showDimensions && dimensions === "3d" ? " · compiling over an elevation profile" : ""}
        </p>
      </div>
    </div>
  </section>;
}

function TurnView({ turn, openArtifact }: { turn: Turn; openArtifact?: (artifact: ArtifactLink) => void }) {
  if (turn.speaker === "user") {
    return <article className="turn user"><div className="bubble">{turn.text}</div></article>;
  }
  const waiting = turn.streaming && !turn.text && turn.steps.length === 0;
  return <article className="turn assistant">
    {turn.progress && <ExperimentProgress progress={turn.progress} streaming={turn.streaming} />}
    {turn.steps.length > 0 && <StepList steps={turn.steps} />}
    {waiting && <div className="typing" aria-label="Working"><i /><i /><i /></div>}
    {turn.text && <div className="reply"><Markdown text={turn.text} />{turn.streaming && <span className="caret" />}</div>}
    {turn.error && <div className="turn-error">{turn.error}</div>}
    {turn.artifacts.length > 0 && <div className="artifact-links">
      {turn.artifacts.map((artifact, index) => <button
        key={`${artifact.kind}-${artifact.id}-${index}`} onClick={() => openArtifact?.(artifact)}
      >{artifact.label} ↗</button>)}
    </div>}
  </article>;
}

function duration(milliseconds: number) {
  const seconds = Math.max(0, Math.round(milliseconds / 1_000));
  if (seconds < 60) return `${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const remainder = seconds % 60;
  return remainder ? `${minutes}m ${remainder}s` : `${minutes}m`;
}

function ExperimentProgress({ progress, streaming }: { progress: Progress; streaming: boolean }) {
  const [now, setNow] = useState(Date.now());
  const active = streaming && progress.completed < progress.total;
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => setNow(Date.now()), 1_000);
    return () => window.clearInterval(timer);
  }, [active]);
  const sinceUpdate = active ? Math.max(0, now - progress.receivedAt) : 0;
  const elapsed = progress.elapsedMs + sinceUpdate;
  const remaining = progress.etaMs == null ? null : Math.max(0, progress.etaMs - sinceUpdate);
  const percent = progress.total ? Math.round((progress.completed / progress.total) * 100) : 0;
  const timing = !streaming && progress.completed < progress.total
    ? "Stopped"
    : progress.completed >= progress.total
      ? `Finished in ${duration(progress.elapsedMs)}`
      : remaining == null
        ? `Estimating · ${duration(elapsed)} elapsed`
        : remaining > 0 ? `About ${duration(remaining)} remaining` : "Finishing current run";
  return <section className="experiment-progress" aria-label="Experiment progress" aria-live="polite">
    <div className="experiment-progress-copy">
      <span>{progress.label}</span>
      <b>{progress.completed}/{progress.total} runs</b>
    </div>
    <div className="experiment-progress-track" role="progressbar" aria-valuemin={0} aria-valuemax={progress.total} aria-valuenow={progress.completed}>
      <i style={{ width: `${percent}%` }} />
    </div>
    <div className="experiment-progress-time"><span>{percent}%</span><span>{timing}</span></div>
  </section>;
}

/** The work log. Collapsed, with a one-line summary of where the flow is, because the
 *  interesting question during a long run is "what is it doing now", not "show me everything". */
function StepList({ steps }: { steps: Step[] }) {
  const running = steps.find((step) => step.state === "running");
  const failed = steps.some((step) => step.state === "failed");
  const summary = running ? running.label : failed ? "Some steps failed" : `${steps.length} steps complete`;
  return <details className="steps" open={false}>
    <summary>
      <span className={`step-dot ${running ? "running" : failed ? "failed" : "done"}`} />
      <span className="step-summary">{summary}</span>
      <span className="step-count">{steps.filter((step) => step.state === "done").length}/{steps.length}</span>
    </summary>
    <ol>
      {steps.map((step) => <li key={step.id} className={step.state}>
        <span className={`step-dot ${step.state}`} />
        <div>
          <span className="step-label">{step.label}</span>
          {step.logs.length > 0 && <ul className="step-logs">
            {step.logs.map((log, index) => <li key={index}>{log}</li>)}
          </ul>}
        </div>
      </li>)}
    </ol>
  </details>;
}
