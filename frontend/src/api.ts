import type { AgentMessage, Dimensions, ElevationSpec, Environment, Experiment, ExperimentRequest, ForkRequest, Run, StreamEvent, TrackDrawing } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
    ...init,
  });
  if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail ?? "Request failed");
  return response.json() as Promise<T>;
}

/** Read a newline-delimited JSON body as it arrives.
 *
 *  A chunk can split a line anywhere, so the tail is held back until its newline shows up;
 *  parsing per chunk instead would drop roughly every long event. */
async function* streamNdjson(path: string, body: unknown): AsyncGenerator<StreamEvent> {
  const response = await fetch(`/api${path}`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  });
  if (!response.ok) throw new Error((await response.json().catch(() => null))?.detail ?? "Request failed");
  if (!response.body) throw new Error("This browser cannot read a streaming response");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    buffer = lines.pop() ?? "";
    for (const line of lines) if (line.trim()) yield JSON.parse(line) as StreamEvent;
  }
  if (buffer.trim()) yield JSON.parse(buffer) as StreamEvent;
}

export const api = {
  drawings: () => request<TrackDrawing[]>("/drawings"),
  createDrawing: (name: string, points: {x: number; y: number}[]) => request<TrackDrawing>("/drawings", {
    method: "POST", body: JSON.stringify({ name, points }),
  }),
  deleteDrawing: (drawingId: string) => request<{deleted_drawing_id: string}>(`/drawings/${drawingId}`, { method: "DELETE" }),
  environments: () => request<Environment[]>("/environments"),
  deleteEnvironment: (environmentId: string) => request<{
    deleted_environment_ids: string[]; deleted_run_ids: string[]; deleted_experiment_ids: string[];
  }>(`/environments/${environmentId}`, { method: "DELETE" }),
  agentMessages: (role: "main" | "environment", environmentId?: string) => request<AgentMessage[]>(`/agents/${role}/messages${environmentId ? `?environment_id=${environmentId}` : ""}`),
  agentActivity: () => request<AgentMessage[]>("/agent-activity?limit=40"),
  streamCoordinator: (message: string, dimensions: Dimensions = "2d", elevation?: ElevationSpec) => streamNdjson("/coordinator/stream", { message, dimensions, elevation }),
  streamAgent: (role: "main" | "environment", message: string, environmentId?: string) => streamNdjson(`/agents/${role}/stream${environmentId ? `?environment_id=${environmentId}` : ""}`, { message }),
  runs: (environmentId?: string) => request<Run[]>(`/runs${environmentId ? `?environment_id=${environmentId}` : ""}`),
  deleteRun: (runId: string) => request<{ deleted_run_ids: string[]; deleted_experiment_ids: string[] }>(`/runs/${runId}`, { method: "DELETE" }),
  deleteExperiment: (experimentKey: string, environmentId: string) => request<{ deleted_run_ids: string[]; deleted_experiment_ids: string[]; experiment_key: string }>(
    `/experiments/${encodeURIComponent(experimentKey)}?environment_id=${encodeURIComponent(environmentId)}`,
    { method: "DELETE" },
  ),
  environmentView3dUrl: (environmentId: string, camera: string, orbit?: { yaw: number; pitch: number; distance: number; focus: string }, width = 900, height = 520) =>
    `/api/environments/${environmentId}/view3d?camera=${camera}&width=${width}&height=${height}`
    + (orbit ? `&yaw=${orbit.yaw}&pitch=${orbit.pitch}&distance=${orbit.distance}&focus=${orbit.focus}` : ""),
  streamExperiment: (environmentId: string, message: string) => streamNdjson(`/environments/${environmentId}/experiment/stream`, { message }),
  createExperiment: (requestBody: ExperimentRequest) => request<Experiment>("/experiments", { method: "POST", body: JSON.stringify(requestBody) }),
  runView3dUrl: (runId: string, step: number, camera: string, width = 900, height = 520) => `/api/runs/${runId}/view3d?step=${step}&camera=${camera}&width=${width}&height=${height}`,
  openNativeViewer: (runId: string) => request<{ status: string; bundle_path: string; renderer: string }>(`/runs/${runId}/open-native-viewer`, { method: "POST" }),
  forkRun: (runId: string, requestBody: ForkRequest) => request<Run>(`/runs/${runId}/fork`, { method: "POST", body: JSON.stringify(requestBody) }),
};
