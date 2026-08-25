import { useEffect, useRef, useState } from "react";
import { api } from "./api";
import { ChatView } from "./Chat";
import { DrawView } from "./Draw";
import { EnvironmentViewer } from "./EnvironmentViewer";
import { View3D, isElevated } from "./View3D";
import type { Camera } from "./View3D";
import { WorldView } from "./WorldView";
import type { AgentMessage, ArtifactLink, ControllerWrite, Environment, Run, StreamEvent } from "./types";
import "./styles.css";

/** Five tabs, one flow: draw or describe a circuit, inspect it, then run experiments.
 *  one of them, and the runs land in the last tab. Anything that was not part of that path has
 *  been taken out of the interface — the endpoints behind it still exist for scripts. */
type View = "coordinator" | "draw" | "environments" | "experiments" | "runs";

const VIEWS: View[] = ["coordinator", "draw", "environments", "experiments", "runs"];
const LABELS: Record<View, string> = {
  coordinator: "Coordinator", draw: "Draw", environments: "Environments", experiments: "Experiments", runs: "Runs",
};

function viewFromHash(): View {
  const requested = window.location.hash.slice(1) as View;
  return VIEWS.includes(requested) ? requested : "coordinator";
}

export default function App() {
  const [view, setView] = useState<View>(viewFromHash());
  const [environments, setEnvironments] = useState<Environment[]>([]);
  const [allRuns, setAllRuns] = useState<Run[]>([]);
  const [environmentId, setEnvironmentId] = useState<string>();
  const [runId, setRunId] = useState<string>();
  const [coordinatorChat, setCoordinatorChat] = useState<AgentMessage[]>([]);
  const [coordinatorLoaded, setCoordinatorLoaded] = useState(false);
  const [environmentChats, setEnvironmentChats] = useState<Record<string, AgentMessage[]>>({});
  const [activity, setActivity] = useState<AgentMessage[]>([]);
  const [error, setError] = useState<string>();
  const [deletingEnvironmentId, setDeletingEnvironmentId] = useState<string>();
  const [coordinatorDraft, setCoordinatorDraft] = useState<{key: number; text: string}>();
  const refreshSequence = useRef(0);
  const environmentIdRef = useRef<string | undefined>(undefined);

  const environment = environments.find((item) => item.id === environmentId) ?? environments[0];
  const experimentEnvironment = sourceEnvironment(environment, environments);
  const experimentEnvironments = environments.filter((item) => !item.parent_environment_id);
  const runs = allRuns.filter((item) => item.environment_id === environment?.id);
  const run = runs.find((item) => item.id === runId) ?? runs[0];

  async function refresh(nextEnvironmentId?: string) {
    const requestId = ++refreshSequence.current;
    const [allEnvironments, chat, fetchedRuns, recentActivity] = await Promise.all([
      api.environments(), api.agentMessages("main"), api.runs(), api.agentActivity(),
    ]);
    if (requestId !== refreshSequence.current) return;
    const preferred = nextEnvironmentId ?? environmentIdRef.current;
    const active = allEnvironments.some((item) => item.id === preferred)
      ? preferred
      : allEnvironments[0]?.id;
    setEnvironments(allEnvironments);
    setCoordinatorChat(chat);
    setCoordinatorLoaded(true);
    setAllRuns(fetchedRuns);
    setActivity(recentActivity);
    setEnvironmentId(active);
    if (!active) { setRunId(undefined); return; }
    const environmentRuns = fetchedRuns.filter((item) => item.environment_id === active);
    setRunId((current) => environmentRuns.some((item) => item.id === current) ? current : environmentRuns[0]?.id);
    const scopedChat = await api.agentMessages("environment", active);
    if (requestId !== refreshSequence.current) return;
    setEnvironmentChats((current) => ({ ...current, [active]: scopedChat }));
  }

  useEffect(() => { void refresh().catch((reason: Error) => setError(reason.message)); }, []);
  useEffect(() => { environmentIdRef.current = environment?.id; }, [environment?.id]);
  useEffect(() => {
    const reload = () => { void refresh(environmentIdRef.current).catch((reason: Error) => setError(reason.message)); };
    window.addEventListener("focus", reload);
    return () => window.removeEventListener("focus", reload);
  }, []);
  useEffect(() => {
    if (window.location.hash.slice(1) !== view) window.history.replaceState(null, "", `#${view}`);
  }, [view]);
  useEffect(() => {
    const onHash = () => setView(viewFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  function select(id: string) {
    setEnvironmentId(id);
    environmentIdRef.current = id;
    void refresh(id).catch((reason: Error) => setError(reason.message));
  }

  /** Pass a stream through untouched and reload once it finishes, so whatever it created is
   *  selected and visible in the tab that owns it. */
  async function* withRefresh(stream: AsyncGenerator<StreamEvent>, land?: (event: StreamEvent) => void) {
    for await (const event of stream) {
      yield event;
      if (event.type === "done") {
        land?.(event);
        // A conversational turn builds nothing, so there is no circuit to select — the
        // reload still happens, to pick up the persisted transcript.
        await refresh(event.result?.environment_id ?? environmentIdRef.current).catch((reason: Error) => setError(reason.message));
      } else if (event.type === "error") {
        await refresh(environmentIdRef.current).catch((reason: Error) => setError(reason.message));
      }
    }
  }

  async function openArtifact(artifact: { kind: string; id: string }, environmentDestination: View = "environments") {
    if (artifact.kind === "environment") {
      select(artifact.id);
      setView(environmentDestination);
      return;
    }
    if (artifact.kind === "run") {
      let target = allRuns.find((item) => item.id === artifact.id);
      if (!target) {
        const latestRuns = await api.runs();
        setAllRuns(latestRuns);
        target = latestRuns.find((item) => item.id === artifact.id);
      }
      if (target) {
        setEnvironmentId(target.environment_id);
        environmentIdRef.current = target.environment_id;
        void refresh(target.environment_id).catch((reason: Error) => setError(reason.message));
      }
      setRunId(artifact.id);
      setView("runs");
    }
  }

  async function deleteExperiment(experimentKey: string, scopedEnvironmentId: string) {
    await api.deleteExperiment(experimentKey, scopedEnvironmentId);
    await refresh(scopedEnvironmentId);
  }

  function deleteEnvironment(target: Environment) {
    const ids = new Set([target.id]);
    let changed = true;
    while (changed) {
      changed = false;
      for (const candidate of environments) {
        if (!ids.has(candidate.id) && candidate.parent_environment_id && ids.has(candidate.parent_environment_id)) {
          ids.add(candidate.id);
          changed = true;
        }
      }
    }
    const relatedRuns = allRuns.filter((candidate) => ids.has(candidate.environment_id));
    const variants = ids.size - 1;
    const scope = [
      variants ? `${variants} derived variant${variants === 1 ? "" : "s"}` : "",
      relatedRuns.length ? `${relatedRuns.length} run${relatedRuns.length === 1 ? "" : "s"} and their forks` : "",
      "its experiment chat",
    ].filter(Boolean).join(", ");
    if (!window.confirm(`Delete “${target.scene.name}”? This also removes ${scope}. This cannot be undone.`)) return;
    setDeletingEnvironmentId(target.id);
    setError(undefined);
    void api.deleteEnvironment(target.id).then(async (result) => {
      setEnvironmentChats((current) => {
        const next = { ...current };
        for (const id of result.deleted_environment_ids) delete next[id];
        return next;
      });
      environmentIdRef.current = undefined;
      setEnvironmentId(undefined);
      setRunId(undefined);
      await refresh();
    }).catch((reason: Error) => setError(reason.message))
      .finally(() => setDeletingEnvironmentId(undefined));
  }

  return <main className="app-shell">
    <header className="topbar">
      <div className="brand"><span className="brand-mark" aria-hidden="true">◒</span><span>RaceLab</span></div>
      <nav>{VIEWS.map((item) => <button
        key={item} className={view === item ? "nav active" : "nav"} onClick={() => setView(item)}
      >{LABELS[item]}{item === "environments" || item === "experiments"
        ? <span className="nav-count">{item === "experiments" ? experimentEnvironments.length : environments.length}</span>
        : item === "runs" ? <span className="nav-count">{allRuns.length}</span> : null}</button>)}</nav>
      <span className="topbar-note">{environments.length} circuit{environments.length === 1 ? "" : "s"}</span>
    </header>
    {error && <div className="notice">{error}<button onClick={() => setError(undefined)}>Dismiss</button></div>}

    <section className="persistent-chat-view" hidden={view !== "coordinator"}>
      <ChatView
        title="Race director"
        subtitle="Describe a circuit, choose its race start below, and I’ll compile and replay-verify it before it exists."
        emptyHint="Try: “slippery curvy track with a 90 degree bend in the top right and three aggressive npcs”"
        history={coordinatorChat}
        conversationId="coordinator"
        loadingHistory={!coordinatorLoaded}
        draftRequest={coordinatorDraft}
        showDimensions
        showStartControls
        openArtifact={(artifact) => {
          void openArtifact(artifact);
        }}
        send={(message, dimensions) => withRefresh(api.streamCoordinator(message, dimensions))}
      />
    </section>

    {view === "draw" && <DrawView useDrawing={(id) => {
      setCoordinatorDraft({ key: Date.now(), text: `use /${id}` });
      setView("coordinator");
    }} />}

    {view === "environments" && <section className="workspace">
      <CircuitList
        environments={environments} selectedId={environment?.id} select={select}
        remove={deleteEnvironment} deletingId={deletingEnvironmentId}
      />
      <section className="stage">
        {environment ? <>
          <h1>{environment.scene.name}</h1>
          <EnvironmentViewer environment={environment} />
        </> : <div className="empty-preview">Ask the coordinator for a circuit.</div>}
      </section>
      <ActivityPanel messages={activity} openArtifact={(artifact) => { void openArtifact(artifact); }} />
    </section>}

    <section className="workspace persistent-chat-view" hidden={view !== "experiments"}>
      <CircuitList
        environments={experimentEnvironments} selectedId={experimentEnvironment?.id} select={select}
        remove={deleteEnvironment} deletingId={deletingEnvironmentId} title="Source circuits"
      />
      {experimentEnvironment ? <section className="experiment-console">
        <ExperimentSuggestion environment={experimentEnvironment} />
        <ExperimentOutcomeSummary
          history={environmentChats[experimentEnvironment.id] ?? []} runs={allRuns} environmentId={experimentEnvironment.id}
          deleteExperiment={deleteExperiment}
        />
        <ChatView
          key={experimentEnvironment.id}
          title={experimentEnvironment.scene.name}
          subtitle="Describe the conditions, seeds, and pace settings you want to compare."
          emptyHint="Try: “compare the predictive controller and vision action agent under low grip”"
          placeholder="Describe a small ad-hoc comparison…"
          history={environmentChats[experimentEnvironment.id] ?? []}
          conversationId={`environment:${experimentEnvironment.id}`}
          loadingHistory={!Object.prototype.hasOwnProperty.call(environmentChats, experimentEnvironment.id)}
          openArtifact={(artifact) => { void openArtifact(artifact); }}
          send={(message) => withRefresh(
            api.streamExperiment(experimentEnvironment.id, message),
            (event) => { if (event.type === "done" && event.run_ids?.length) setRunId(event.run_ids[0]); },
          )}
        />
      </section> : <section className="stage"><div className="empty-preview">Select a circuit to run experiments on.</div></section>}
      <ActivityPanel messages={activity} openArtifact={(artifact) => { void openArtifact(artifact); }} />
    </section>

    {view === "runs" && <RunsView
      environment={environment} environments={environments} runs={runs} run={run}
      select={setRunId} selectEnvironment={select} refresh={refresh}
      deleteExperiment={deleteExperiment}
      activity={activity} openArtifact={(artifact) => { void openArtifact(artifact); }}
    />}
  </main>;
}

/** Seed variants are evidence inside an experiment, never new matrix sources. */
function sourceEnvironment(selected: Environment | undefined, environments: Environment[]) {
  let current = selected;
  const visited = new Set<string>();
  while (current?.parent_environment_id && !visited.has(current.id)) {
    visited.add(current.id);
    current = environments.find((item) => item.id === current!.parent_environment_id);
  }
  return current;
}

function ExperimentSuggestion({ environment }: { environment: Environment }) {
  const elevated = isElevated(environment.scene);
  return <section className="experiment-suggestion" aria-label="Experiment suggestion">
    <span className="eyebrow">Things to vary</span>
    <ul>
      <li>Track conditions</li><li>Deterministic seeds</li><li>Player aggression</li><li>Tick budget</li>
    </ul>
    <small>Describe any combination in the chat below · player fixed: predictive skills ({elevated ? "3D camera + speed + elevation" : "2D camera + speed"})</small>
  </section>;
}

function CircuitList({ environments, selectedId, select, remove, deletingId, title = "Circuits" }: {
  environments: Environment[]; selectedId?: string; select(id: string): void;
  remove(item: Environment): void; deletingId?: string; title?: string;
}) {
  return <aside className="sidebar">
    <div className="eyebrow">{title}</div>
    {environments.length === 0 ? <p className="muted">None yet.</p> : environments.map((item) => {
      const active = item.id === selectedId;
      return <div key={item.id} className={active ? "circuit-row active" : "circuit-row"}>
        <button className={active ? "tree-item active" : "tree-item"} onClick={() => select(item.id)}>
          <span className="tree-item-title">{item.scene.name}</span>
          <small>{item.scene.surface} · {isElevated(item.scene) ? "3D" : "2D"} · seed {item.scene.seed}</small>
        </button>
        {active && <button className="circuit-delete" disabled={Boolean(deletingId)} onClick={() => remove(item)}>
          {deletingId === item.id ? "Deleting…" : "Delete circuit"}
        </button>}
      </div>;
    })}
  </aside>;
}

function ExperimentOutcomeSummary({ history, runs, environmentId, deleteExperiment }: {
  history: AgentMessage[]; runs: Run[]; environmentId: string;
  deleteExperiment(experimentKey: string, environmentId: string): Promise<void>;
}) {
  const [deleting, setDeleting] = useState<string>();
  const byId = new Map(runs.map((run) => [run.id, run]));
  const experiments = history.flatMap((message, index) => {
    if (message.speaker !== "assistant") return [];
    const runArtifacts = message.artifacts.filter((artifact) => artifact.kind === "run");
    const attemptedRuns = message.actions.filter((action) => action.id.startsWith("run:"));
    if (!runArtifacts.length && !attemptedRuns.length) return [];
    const request = [...history.slice(0, index)].reverse().find((candidate) => candidate.speaker === "user");
    const records = runArtifacts.map((artifact) => byId.get(artifact.id)).filter((run): run is Run => Boolean(run));
    return [{
      id: message.id,
      title: request?.content.split("\n")[0] || "Experiment run",
      successful: records.filter((run) => run.status === "succeeded").length,
      total: Math.max(attemptedRuns.length, runArtifacts.length),
      groupKey: records[0]?.address?.experiment
        ? `experiment-${records[0].address.experiment}`
        : records[0] ? `legacy-${records[0].study_name ?? "ad-hoc"}` : undefined,
    }];
  }).reverse();

  return <section className="experiment-outcomes" aria-label="Experiment run outcomes">
    <div className="experiment-outcomes-heading">
      <span>Experiment outcomes</span>
      <small>{experiments.length} total</small>
    </div>
    {experiments.length === 0
      ? <p>No experiments launched for this circuit yet.</p>
      : <div className="experiment-outcome-list">{experiments.map((experiment) => <article key={experiment.id}>
        <div><b>{experiment.title}</b><small>{experiment.successful} successful runs out of {experiment.total}</small></div>
        <strong>{experiment.successful}/{experiment.total}</strong>
        {experiment.groupKey && <button className="danger-link" disabled={deleting === experiment.groupKey} onClick={() => {
          const label = experiment.groupKey!.startsWith("experiment-")
            ? `EXP-${experiment.groupKey!.replace("experiment-", "").padStart(3, "0")}`
            : experiment.title;
          if (!window.confirm(`Delete ${label} and every run and perturbation it contains? The circuit will be kept. This cannot be undone.`)) return;
          setDeleting(experiment.groupKey);
          void deleteExperiment(experiment.groupKey!, environmentId)
            .catch((reason: Error) => window.alert(reason.message))
            .finally(() => setDeleting(undefined));
        }}>{deleting === experiment.groupKey ? "Deleting…" : "Delete experiment"}</button>}
      </article>)}</div>}
  </section>;
}

function ActivityPanel({ messages, openArtifact }: {
  messages: AgentMessage[];
  openArtifact(artifact: ArtifactLink): void;
}) {
  const visible = messages.filter((message) => message.actions.length || message.artifacts.length).slice(0, 8);
  return <aside className="inspector activity-panel">
    <div className="eyebrow">Pit wall · recent actions</div>
    {visible.length === 0 ? <p className="activity-empty">Completed work will stay visible here across every tab.</p> : <div className="activity-ledger">
      {visible.map((message) => {
        const artifacts = [...message.artifacts];
        for (const action of message.actions) {
          if (action.artifact && !artifacts.some((item) => item.kind === action.artifact?.kind && item.id === action.artifact?.id)) artifacts.push(action.artifact);
        }
        return <article className="activity-entry" key={message.id}>
          <div className="activity-meta">
            <span>{message.agent_role === "main" ? "Coordinator" : "Experiment"}</span>
            <time>{new Date(message.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time>
          </div>
          {message.actions.length > 0 ? <ul className="activity-actions">
            {message.actions.slice(0, 4).map((action) => <li key={action.id}>
              <span className={`step-dot ${action.state}`} />
              <span>{action.label}</span>
            </li>)}
            {message.actions.length > 4 && <li className="activity-more">+{message.actions.length - 4} more actions</li>}
          </ul> : <p className="activity-summary">{message.content.split("\n")[0]}</p>}
          {artifacts.length > 0 && <div className="activity-artifacts">
            {artifacts.map((artifact) => <button key={`${artifact.kind}-${artifact.id}`} onClick={() => openArtifact(artifact)}>
              {artifact.label} <span aria-hidden="true">↗</span>
            </button>)}
          </div>}
        </article>;
      })}
    </div>}
  </aside>;
}

function ControllerScriptViewer({ writes, onJump }: {
  writes: ControllerWrite[];
  onJump(frameStep: number): void;
}) {
  if (!writes.length) return null;
  return <details className="controller-viewer">
    <summary>
      <span><i aria-hidden="true">⌁</i> Controller scripts</span>
      <span>{writes.length} authored · click to inspect</span>
    </summary>
    <div className="controller-list">
      {writes.map((write, index) => <details className="controller-write" key={`${write.wake}-${write.tick}-${write.name}-${index}`}>
        <summary>
          <span className={`controller-status ${write.installed ? "installed" : "rejected"}`} />
          <b>{write.label ?? `${write.name} draft`}</b>
          <span>{write.effective_from_frame_step == null
            ? `authored near replay tick ${write.frame_step}`
            : `takes effect at replay tick ${write.effective_from_frame_step}`}</span>
        </summary>
        <div className="controller-body">
          <div className="controller-toolbar">
            <span>{!write.installed
              ? "Rejected by safety gate"
              : write.effective_from_frame_step == null
                ? `Installed at controller tick ${write.tick}, but never drove a live tick`
                : `Drove replay ticks ${write.effective_from_frame_step}–${write.effective_until_frame_step ?? "end"} · wake ${write.wake}`}</span>
            <button onClick={() => onJump(write.effective_from_frame_step ?? write.frame_step)}>
              {write.effective_from_frame_step == null ? "View authored state" : "View first controlled tick"}
            </button>
          </div>
          {write.reads.length > 0 && <p className="controller-contract">Reads: {write.reads.join(" · ")}</p>}
          {Object.keys(write.params).length > 0 && <p className="controller-contract">Parameters: {JSON.stringify(write.params)}</p>}
          {write.errors.length > 0 && <ul className="controller-errors">{write.errors.map((item, errorIndex) => <li key={errorIndex}>{item}</li>)}</ul>}
          <pre><code>{write.source || "# No source was returned"}</code></pre>
        </div>
      </details>)}
    </div>
  </details>;
}

interface RunTreeNode {
  run: Run;
  children: RunTreeNode[];
}

interface RunExperimentGroup {
  key: string;
  code: string;
  name: string;
  roots: RunTreeNode[];
  perturbationCount: number;
  latestStartedAt: number;
}

const LEGACY_POLICY_NAMES: Record<string, string> = {
  "racing-line": "oracle-racing-line",
  "racing-agent": "telemetry-direct",
  "racing-agent-strategy": "telemetry-strategy",
  "racing-agent-hierarchical": "telemetry-hierarchical",
  "racing-agent-reflex": "telemetry-reflex",
  "racing-agent-reflex-vision": "vision-reflex-sim-rehearsal",
  "racing-agent-cone-visual": "vision-2d-direct",
  "racing-agent-2d-predictive-skills": "vision-2d-predictive-skills",
  "racing-agent-3d-visual-tick": "vision-3d-direct-every-tick",
  "racing-agent-3d-visual-short": "vision-3d-direct-short",
  "racing-agent-3d-visual-short-speed-road": "vision-3d-direct-short-features",
  "racing-agent-3d-predictive-skills": "vision-3d-predictive-skills",
  "constant-intent": "baseline-constant-intent",
  "wanderer": "baseline-random",
  "external-player": "external-telemetry-player",
};

const POLICY_LABELS: Record<string, string> = {
  "vision-2d-predictive-skills": "Vision Controller Agent · predictive skills · 2D",
  "vision-2d-direct": "Vision Action Agent · 2D",
  "vision-reflex-sim-rehearsal": "Vision Controller Agent · 2D",
  "vision-3d-direct-every-tick": "Vision Action Agent · every tick · 3D",
  "vision-3d-direct-short": "Vision Action Agent · short horizon · 3D",
  "vision-3d-direct-short-features": "Vision Action Agent · road features · 3D",
  "vision-3d-predictive-skills": "Vision Controller Agent · predictive skills · 3D",
  "telemetry-direct": "Diagnostic · telemetry action agent",
  "telemetry-strategy": "Diagnostic · telemetry strategy",
  "telemetry-hierarchical": "Diagnostic · telemetry hierarchical",
  "telemetry-reflex": "Diagnostic · telemetry controller agent",
  "oracle-racing-line": "Oracle racing line",
};

function canonicalPolicyName(value: string) {
  return LEGACY_POLICY_NAMES[value] || value;
}

function humanizeRunLabel(value: string) {
  const canonical = canonicalPolicyName(value);
  return POLICY_LABELS[canonical] ?? canonical.replaceAll("_", " ").replaceAll("-", " ");
}

function runStartedAt(run: Run) {
  return new Date(run.started_at).getTime() || 0;
}

function runUsageFacts(run: Run) {
  const aggression = run.player_aggression == null
    ? ""
    : ` · ${Math.round(run.player_aggression * 100)}% aggression`;
  return `${run.player_turns ?? 0} model calls · ${run.token_usage.toLocaleString()} tokens${aggression}`;
}

function RunMetricStrip({ run }: { run: Run }) {
  const succeeded = run.status === "succeeded";
  const reason = run.result_reason ?? run.status;
  return <section className="run-metric-strip" aria-label="Selected run metrics">
    <article>
      <span>Outcome</span>
      <strong className={succeeded ? "success" : ""}>{run.status}</strong>
      <small title={reason}>{reason}</small>
    </article>
    <article>
      <span>Recorded ticks</span>
      <strong>{run.frames.length.toLocaleString()}</strong>
      <small>{run.total_reward.toFixed(2)} reward</small>
    </article>
    <article>
      <span>Model calls</span>
      <strong>{(run.player_turns ?? 0).toLocaleString()}</strong>
      <small>{run.player_aggression == null ? "default aggression" : `${Math.round(run.player_aggression * 100)}% aggression`}</small>
    </article>
    <article>
      <span>Token spend</span>
      <strong>{run.token_usage.toLocaleString()}</strong>
      <small>{(run.input_tokens ?? 0).toLocaleString()} in · {(run.output_tokens ?? 0).toLocaleString()} out</small>
    </article>
  </section>;
}

function sameLogicalRun(left: Run, right: Run) {
  if (canonicalPolicyName(left.policy_name) !== canonicalPolicyName(right.policy_name) || left.seed !== right.seed) return false;
  if (!left.address || !right.address) return left.environment_id === right.environment_id;
  return left.address.environment === right.address.environment && left.address.variant === right.address.variant;
}

/** Turn both replay forks and experiment-matrix perturbations into one honest tree.
 *  Forks already name their parent. Older matrix runs do not, so a perturbed row is
 *  paired with the closest baseline that used the same driver, seed, and environment. */
function buildRunExperimentGroups(runs: Run[]): RunExperimentGroup[] {
  const grouped = new Map<string, Run[]>();
  for (const run of runs) {
    const key = run.address?.experiment
      ? `experiment-${run.address.experiment}`
      : `legacy-${run.study_name ?? "ad-hoc"}`;
    grouped.set(key, [...(grouped.get(key) ?? []), run]);
  }

  return [...grouped.entries()].map(([key, experimentRuns]) => {
    const ordered = [...experimentRuns].sort((left, right) => runStartedAt(left) - runStartedAt(right));
    const runIds = new Set(ordered.map((item) => item.id));
    const effectiveParents = new Map<string, string>();

    for (const item of ordered) {
      if (item.parent_run_id && runIds.has(item.parent_run_id)) {
        effectiveParents.set(item.id, item.parent_run_id);
        continue;
      }
      if (!item.perturbation) continue;
      const baseline = ordered
        .filter((candidate) => !candidate.parent_run_id && !candidate.perturbation && sameLogicalRun(item, candidate))
        .sort((left, right) => Math.abs(runStartedAt(item) - runStartedAt(left)) - Math.abs(runStartedAt(item) - runStartedAt(right)))[0];
      if (baseline) effectiveParents.set(item.id, baseline.id);
    }

    const nodes = new Map(ordered.map((item) => [item.id, { run: item, children: [] as RunTreeNode[] }]));
    for (const [childId, parentId] of effectiveParents) {
      const child = nodes.get(childId);
      const parent = nodes.get(parentId);
      if (child && parent) parent.children.push(child);
    }
    for (const node of nodes.values()) node.children.sort((left, right) => runStartedAt(left.run) - runStartedAt(right.run));
    const roots = ordered.filter((item) => !effectiveParents.has(item.id)).map((item) => nodes.get(item.id)!);
    const experimentNumber = ordered.find((item) => item.address)?.address?.experiment;
    return {
      key,
      code: experimentNumber ? `EXP-${String(experimentNumber).padStart(3, "0")}` : "EXP-LEGACY",
      name: ordered.find((item) => item.study_name)?.study_name ?? "Ad hoc runs",
      roots,
      perturbationCount: ordered.length - roots.length,
      latestStartedAt: Math.max(...ordered.map(runStartedAt)),
    };
  }).sort((left, right) => right.latestStartedAt - left.latestStartedAt);
}

function treeContainsRun(node: RunTreeNode, runId?: string): boolean {
  return node.run.id === runId || node.children.some((child) => treeContainsRun(child, runId));
}

function treeRunCount(node: RunTreeNode): number {
  return 1 + node.children.reduce((total, child) => total + treeRunCount(child), 0);
}

function RunTreeEntry({ node, code, selectedId, select, root = false }: {
  node: RunTreeNode;
  code: string;
  selectedId?: string;
  select(id: string): void;
  root?: boolean;
}) {
  const item = node.run;
  const active = item.id === selectedId;
  const containsSelected = treeContainsRun(node, selectedId);
  const [expanded, setExpanded] = useState(containsSelected);
  useEffect(() => { if (containsSelected) setExpanded(true); }, [containsSelected]);
  const title = root
    ? humanizeRunLabel(item.policy_name)
    : item.perturbation?.condition
      ? item.perturbation.condition
      : item.perturbation?.kind ? humanizeRunLabel(item.perturbation.kind) : "Replay fork";
  const facts = root
    ? `${item.status} · ${item.frames.length} ticks · ${runUsageFacts(item)} · seed ${item.seed}`
    : `${item.status} · ${item.frames.length} ticks · ${runUsageFacts(item)}${item.fork_step == null ? "" : ` · from tick ${item.fork_step}`}`;
  const contents = <>
    <span className="run-tree-code">{code}</span>
    <span className="run-tree-copy"><b>{title}</b><small>{facts}</small></span>
    <span className={`run-result-dot ${item.status}`} title={item.status} />
  </>;

  if (root && node.children.length > 0) return <details
    className="run-folder"
    open={expanded}
    onToggle={(event) => setExpanded(event.currentTarget.open)}
  >
    <summary className={active ? "run-tree-entry active" : "run-tree-entry"} onClick={() => select(item.id)}>
      {contents}<span className="run-branch-count">{node.children.length}</span>
    </summary>
    <div className="perturbation-list">
      {node.children.map((child, index) => <RunTreeEntry
        key={child.run.id} node={child} code={`P-${String(index + 1).padStart(2, "0")}`}
        selectedId={selectedId} select={select}
      />)}
    </div>
  </details>;

  return <div className={root ? "run-tree-leaf" : "perturbation-node"}>
    <button className={active ? "run-tree-entry active" : "run-tree-entry"} onClick={() => select(item.id)}>{contents}</button>
    {node.children.length > 0 && <div className="perturbation-list nested">
      {node.children.map((child, index) => <RunTreeEntry
        key={child.run.id} node={child} code={`${code}.${index + 1}`}
        selectedId={selectedId} select={select}
      />)}
    </div>}
  </div>;
}

function ExperimentRunFolder({ group, initiallyOpen, selectedId, select, remove, deleting }: {
  group: RunExperimentGroup;
  initiallyOpen: boolean;
  selectedId?: string;
  select(id: string): void;
  remove(group: RunExperimentGroup): void;
  deleting: boolean;
}) {
  const selected = group.roots.some((root) => treeContainsRun(root, selectedId));
  const [expanded, setExpanded] = useState(initiallyOpen || selected);
  useEffect(() => { if (selected) setExpanded(true); }, [selected]);
  return <details className="experiment-folder" open={expanded} onToggle={(event) => setExpanded(event.currentTarget.open)}>
    <summary className={selected ? "experiment-summary active" : "experiment-summary"}>
      <span className="experiment-code">{group.code}</span>
      <span className="experiment-copy">
        <b>{group.name}</b>
        <small>{group.roots.length} run{group.roots.length === 1 ? "" : "s"} · {group.perturbationCount} perturbation{group.perturbationCount === 1 ? "" : "s"}</small>
      </span>
      <button className="tree-delete" disabled={deleting} onClick={(event) => {
        event.preventDefault();
        event.stopPropagation();
        remove(group);
      }}>{deleting ? "Deleting…" : "Delete"}</button>
    </summary>
    <div className="run-folder-list">
      {group.roots.map((root, index) => <RunTreeEntry
        key={root.run.id} node={root} code={`RUN-${String(index + 1).padStart(3, "0")}`}
        selectedId={selectedId} select={select} root
      />)}
    </div>
  </details>;
}

function RunsTree({ runs, selectedId, select, remove, deletingKey }: {
  runs: Run[]; selectedId?: string; select(id: string): void;
  remove(group: RunExperimentGroup): void; deletingKey?: string;
}) {
  const groups = buildRunExperimentGroups(runs);
  if (!groups.length) return <p className="muted">Launch a run from Experiments.</p>;
  return <div className="runs-tree">
    {groups.map((group, groupIndex) => <ExperimentRunFolder
      key={group.key} group={group} initiallyOpen={groupIndex === 0}
      selectedId={selectedId} select={select} remove={remove} deleting={deletingKey === group.key}
    />)}
  </div>;
}

function RunsView({ environment, environments, runs, run, select, selectEnvironment, refresh, deleteExperiment, activity, openArtifact }: {
  environment?: Environment; environments: Environment[]; runs: Run[]; run?: Run;
  select(id: string): void; selectEnvironment(id: string): void; refresh(environmentId?: string): Promise<void>;
  deleteExperiment(experimentKey: string, environmentId: string): Promise<void>;
  activity: AgentMessage[]; openArtifact(artifact: ArtifactLink): void;
}) {
  const [step, setStep] = useState(0);
  const [camera, setCamera] = useState<Camera>("third-person");
  const [viewMode, setViewMode] = useState<"replay" | "trajectory">("replay");
  const [opening, setOpening] = useState(false);
  const [forkStep, setForkStep] = useState<number>();
  const [forkCondition, setForkCondition] = useState("");
  const [forking, setForking] = useState(false);
  const [forkError, setForkError] = useState<string>();
  const [deletingRun, setDeletingRun] = useState(false);
  const [deletingExperimentKey, setDeletingExperimentKey] = useState<string>();
  const [deleteError, setDeleteError] = useState<string>();
  useEffect(() => { setStep(0); setViewMode("replay"); setForkStep(undefined); setForkCondition(""); setForkError(undefined); }, [run?.id]);
  const frame = run?.frames[Math.min(step, Math.max(0, (run?.frames.length ?? 1) - 1))];

  function removeExperiment(group: RunExperimentGroup) {
    if (!environment) return;
    const count = group.roots.reduce((total, root) => total + treeRunCount(root), 0);
    if (!window.confirm(`Delete ${group.code} and all ${count} run${count === 1 ? "" : "s"} and perturbations it contains? The circuit will be kept. This cannot be undone.`)) return;
    setDeleteError(undefined);
    setDeletingExperimentKey(group.key);
    void deleteExperiment(group.key, environment.id)
      .catch((reason: Error) => setDeleteError(reason.message))
      .finally(() => setDeletingExperimentKey(undefined));
  }

  function removeRun() {
    if (!run || !environment) return;
    const descendants = runs.filter((candidate) => {
      let parentId = candidate.parent_run_id;
      const visited = new Set<string>();
      while (parentId && !visited.has(parentId)) {
        if (parentId === run.id) return true;
        visited.add(parentId);
        parentId = runs.find((item) => item.id === parentId)?.parent_run_id;
      }
      return false;
    }).length;
    const suffix = descendants ? ` This also deletes ${descendants} forked perturbation${descendants === 1 ? "" : "s"}.` : "";
    if (!window.confirm(`Delete this run?${suffix} This cannot be undone.`)) return;
    setDeleteError(undefined);
    setDeletingRun(true);
    void api.deleteRun(run.id)
      .then(() => refresh(environment.id))
      .catch((reason: Error) => setDeleteError(reason.message))
      .finally(() => setDeletingRun(false));
  }

  return <section className="workspace">
    <aside className="sidebar runs-sidebar">
      <div className="eyebrow">Circuit</div>
      <select value={environment?.id ?? ""} onChange={(event) => selectEnvironment(event.target.value)}>
        {environments.map((item) => <option key={item.id} value={item.id}>{item.scene.name}</option>)}
      </select>
      <div className="eyebrow spaced">Experiment index</div>
      <RunsTree
        runs={runs} selectedId={run?.id} select={select}
        remove={removeExperiment} deletingKey={deletingExperimentKey}
      />
      {deleteError && <p className="delete-error">{deleteError}</p>}
    </aside>
    <section className="stage">
      {!run || !environment ? <div className="empty-preview">Launch a run from Experiments.</div> : <>
        <div className="run-stage-header">
          <h1>{humanizeRunLabel(run.policy_name)}</h1>
          <button className="danger-button" disabled={deletingRun || run.status === "pending" || run.status === "running"} onClick={removeRun}>
            {deletingRun ? "Deleting…" : "Delete run"}
          </button>
        </div>
        <RunMetricStrip run={run} />
        <ControllerScriptViewer writes={run.controller_writes ?? []} onJump={(frameStep) => {
          const index = run.frames.findIndex((candidate) => candidate.step >= frameStep);
          setStep(index >= 0 ? index : Math.max(0, run.frames.length - 1));
          setViewMode("replay");
        }} />
        <div className="run-view-picker" aria-label="Run view">
          <button className={viewMode === "replay" ? "active" : ""} onClick={() => setViewMode("replay")}>Replay</button>
          <button className={viewMode === "trajectory" ? "active" : ""} onClick={() => setViewMode("trajectory")}>Top-down path</button>
        </div>
        {viewMode === "trajectory"
          ? <>
            <WorldView scene={environment.scene} frame={run.frames[run.frames.length - 1]} trajectory={run.frames} selectedPoint={forkStep === undefined ? undefined : run.frames[forkStep - 1]?.privileged_state.player} onMapClick={(point) => {
              const nearest = run.frames.reduce((best, candidate, index) => {
                const dx = candidate.privileged_state.player.x - point.x;
                const dy = candidate.privileged_state.player.y - point.y;
                return dx * dx + dy * dy < best.distance ? { index, distance: dx * dx + dy * dy } : best;
              }, { index: 0, distance: Number.POSITIVE_INFINITY });
              setForkStep(nearest.index + 1);
            }} />
            <div className="fork-panel">
              <b>{forkStep === undefined ? "Click the path to select a replay point" : `Fork from tick ${run.frames[forkStep - 1]?.step ?? 0}`}</b>
              <input
                value={forkCondition}
                onChange={(event) => setForkCondition(event.target.value)}
                placeholder={run.guidance_supported
                  ? "e.g. Lower grip and tell the driver to brake earlier"
                  : "e.g. Lower grip, increase drag, or replay exactly"}
                maxLength={600}
                aria-label="Natural-language fork condition"
              />
              <button disabled={forkStep === undefined || forking || !run.fork_supported} onClick={() => {
                if (forkStep === undefined) return;
                setForking(true);
                setForkError(undefined);
                void api.forkRun(run.id, { fork_step: forkStep, condition: forkCondition.trim() || undefined })
                  .then(async (fork) => { await refresh(environment.id); select(fork.id); })
                  .catch((reason: Error) => setForkError(reason.message))
                  .finally(() => setForking(false));
              }}>{forking ? "Forking…" : forkCondition.trim() ? "Fork with intervention" : "Fork continuation"}</button>
              <span className={`correction-contract ${run.guidance_supported ? "supported" : "unsupported"}`}>
                {run.guidance_supported
                  ? "Describe one engine change and/or a correction for every subsequent model decision."
                  : run.fork_supported
                    ? "Describe one engine change. This policy cannot consume driving corrections."
                    : "This controller-based policy cannot restore a fork yet."}
              </span>
              {forkError && <span className="fork-error">{forkError}</span>}
            </div>
          </>
          : isElevated(environment.scene)
          ? <View3D
              src={api.runView3dUrl(run.id, frame?.step ?? 0, camera === "plan" ? "third-person" : camera)}
              camera={camera} setCamera={setCamera} elevation={environment.scene.elevation!}
            ><WorldView scene={environment.scene} frame={frame} /></View3D>
          : <WorldView scene={environment.scene} frame={frame} />}
        {viewMode === "replay" && <div className="timeline">
          <div className="timeline-label">
            <span><b>tick {frame?.step ?? 0}</b> {frame?.action ?? "idle"}</span>
            <span>{frame?.events.join(" · ") || "on circuit"}</span>
          </div>
          <input
            type="range" min="0" max={Math.max(0, run.frames.length - 1)} value={step}
            onChange={(event) => setStep(Number(event.target.value))}
          />
        </div>}
        <div className="action-row">
          <button disabled={opening} onClick={() => {
            setOpening(true);
            void api.openNativeViewer(run.id).finally(() => setOpening(false));
          }}>{opening ? "Opening…" : "Open desktop replay ↗"}</button>
          {frame?.decision && <span className="muted">{frame.decision.subgoal}</span>}
        </div>
      </>}
    </section>
    <ActivityPanel messages={activity} openArtifact={openArtifact} />
  </section>;
}
