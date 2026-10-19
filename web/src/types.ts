/** Wire types for the live runner API (mirrors server/app/models.py). */

export type StageId = "p0" | "p1" | "p2" | "p3" | "p4" | "p5" | "p6";
export type StageStatus = "idle" | "running" | "done" | "failed";
export type RunStatus = "running" | "completed" | "failed";

export interface DatabaseStatus {
  state: "ok" | "offline";
  detail?: string | null;
  graphs?: number | null;
}

export interface ProviderStatus {
  name: string;
  model?: string | null;
  available: boolean;
  detail?: string | null;
}

export interface Health {
  database: DatabaseStatus;
  provider: ProviderStatus;
  active_run?: string | null;
}

export interface CaseInfo {
  case: string;
  service: string;
  fault: string;
  instance: string;
  source_url?: string | null;
}

export interface ActivityEntry {
  seq: number;
  stage?: string | null;
  task?: string | null;
  ms?: number | null;
  tokens_in?: number | null;
  tokens_out?: number | null;
  model?: string | null;
  usage?: string | null;
  finish?: string | null;
  status: string;
}

export interface StageDetailEntry {
  k: string;
  v: string;
}

export interface StageState {
  id: StageId;
  name: string;
  status: StageStatus;
  line?: string | null;
  detail: StageDetailEntry[];
}

export interface CostState {
  calls: number;
  tokens_in: number;
  tokens_out: number;
  tokens_total: number;
  tokens_charged: number;
  wall_s: number;
  unknown_usage_attempts: number;
  compliant: boolean;
  budget: Record<string, number>;
}

export interface RunResult {
  tier?: string | null;
  verdict?: string | null;
  confidence?: number | null;
  confidence_rule?: string | null;
  winner?: string | null;
  reasons: string[];
  next_action?: string | null;
  summary: string[];
  abstained: boolean;
  markdown?: string | null;
  session_graph?: string | null;
  mode?: "full" | "preview";
  candidates?: Array<{ id: string; rank?: number | null; score: number }>;
  truncated?: boolean;
}

export interface RunSnapshot {
  run_id: string;
  case_id: string;
  kind?: "benchmark" | "custom";
  mode?: "full" | "preview";
  status: RunStatus;
  started_at: number;
  finished_at?: number | null;
  server_time?: number | null;
  duration_s: number;
  stages: StageState[];
  activity?: ActivityEntry[];
  cost?: CostState | null;
  result?: RunResult | null;
  error?: string | null;
  cleanup?: string | null;
}

export interface StoredRun {
  run_id: string;
  case_id: string;
  status: string;
  started_at: number;
  duration_s?: number | null;
  tier?: string | null;
  winner?: string | null;
  calls?: number | null;
  tokens_total?: number | null;
}

export interface ResultsPage {
  runs: StoredRun[];
}

export interface ProbeRequest {
  host: string;
  port: number;
  graph: string;
  anchor_service?: string | null;
  username?: string | null;
  password?: string | null;
}

export interface SchemaCheck {
  name: string;
  status: "pass" | "warn" | "fail";
  detail: string;
}

export interface ProbeResult {
  state: "ok" | "error";
  verdict: "ready" | "incomplete" | "unreachable" | "blocked";
  checks: SchemaCheck[];
  graphs: string[];
  counts?: Record<string, number> | null;
  anchor?: string | null;
  host?: string | null;
  port?: number | null;
  graph?: string | null;
  time_range?: { t0: number; t1: number } | null;
  topology?: Topology | null;
  detail?: string | null;
}

export interface CustomRunRequest extends ProbeRequest {
  anchor_service: string;
  window_t0: number;
  window_t1: number;
  onset_ts?: number | null;
  symptom?: string | null;
  incident_id?: string | null;
  mode: "preview" | "full";
  label?: string | null;
}

export interface TopologyNode {
  id: string;
  name: string;
  level: number;
  criticality: number;
}

export interface TopologyEdge {
  source: string;
  target: string;
}

export interface Topology {
  nodes: TopologyNode[];
  edges: TopologyEdge[];
  excluded: string[];
  provenance?: string | null;
  note?: string | null;
  caveat?: string | null;
  derived_levels?: boolean;
}

export interface MetricExcursion {
  service: string;
  column: string;
  metric?: string | null;
  z: number;
  direction?: string | null;
  value?: number | null;
  ts?: number | null;
  baseline_mean?: number | null;
  baseline_std?: number | null;
}

export interface LogSample {
  ts: number;
  service: string;
  message: string;
}

export interface SpanService {
  service: string;
  spans: number;
  errors: number;
  p95_ms: number;
}

export interface SlowSpan {
  service: string;
  op: string;
  ms: number;
  error: boolean;
  ts: number;
}

export interface CaseEvidence {
  case: string;
  onset_ms?: number | null;
  window: { t0?: number; t1?: number };
  metrics: MetricExcursion[];
  logs: {
    total_errors: number;
    by_service: Array<{ service: string; count: number }>;
    samples: LogSample[];
    band_s?: number | null;
  };
  spans: {
    total: number;
    errors: number;
    admitted?: number | null;
    services: SpanService[];
    slowest: SlowSpan[];
  };
  counts: Record<string, number>;
}
