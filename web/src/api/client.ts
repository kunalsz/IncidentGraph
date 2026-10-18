/** Typed access to the live runner API. */

import type {
  CaseEvidence,
  CaseInfo,
  CustomRunRequest,
  Health,
  ProbeRequest,
  ProbeResult,
  ResultsPage,
  RunSnapshot,
  Topology,
} from "../types";

const BASE = "/api";

class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, init);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = (await response.json()) as { detail?: string };
      detail = body.detail ?? detail;
    } catch {
      /* keep statusText */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as T;
}

export const api = {
  health: (signal?: AbortSignal) => request<Health>("/health", { signal }),
  cases: (signal?: AbortSignal) => request<CaseInfo[]>("/cases", { signal }),
  results: (signal?: AbortSignal) => request<ResultsPage>("/results", { signal }),
  startRun: (caseId: string) =>
    request<RunSnapshot>("/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ case_id: caseId }),
    }),
  run: (runId: string, signal?: AbortSignal) =>
    request<RunSnapshot>(`/runs/${encodeURIComponent(runId)}`, { signal }),
  probe: (body: ProbeRequest) =>
    request<ProbeResult>("/dashboard/probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  startCustomRun: (body: CustomRunRequest) =>
    request<RunSnapshot>("/dashboard/runs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }),
  topology: (signal?: AbortSignal) => request<Topology>("/topology", { signal }),
  caseEvidence: (caseId: string, signal?: AbortSignal) =>
    request<CaseEvidence>(`/cases/${encodeURIComponent(caseId)}/evidence`, { signal }),
};

export const streamUrl = (runId: string): string =>
  `${BASE}/runs/${encodeURIComponent(runId)}/stream`;

export { ApiError };
