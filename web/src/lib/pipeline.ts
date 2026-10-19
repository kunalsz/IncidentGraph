import type { StageState } from "../types";

/** Stage names before any run has started, so the pipeline shape is visible. */
export const IDLE_STAGES: StageState[] = [
  { id: "p0", name: "resolve", status: "idle", line: null, detail: [] },
  { id: "p1", name: "scope", status: "idle", line: null, detail: [] },
  { id: "p2", name: "rank", status: "idle", line: null, detail: [] },
  { id: "p3", name: "investigate", status: "idle", line: null, detail: [] },
  { id: "p4", name: "validate", status: "idle", line: null, detail: [] },
  { id: "p5", name: "select", status: "idle", line: null, detail: [] },
  { id: "p6", name: "report", status: "idle", line: null, detail: [] },
];
