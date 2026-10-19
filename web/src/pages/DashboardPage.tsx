import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import { CostLine } from "../components/CostLine";
import { DependencyGraph } from "../components/DependencyGraph";
import { StageLedger } from "../components/StageLedger";
import { Verdict } from "../components/Verdict";
import type { RunStoreApi } from "../hooks/useRunStore";
import { useElapsed } from "../hooks/useElapsed";
import { IDLE_STAGES } from "../lib/pipeline";
import type { Health, ProbeResult } from "../types";

interface DashboardPageProps {
  health: Health | null;
  store: RunStoreApi;
  onFinished: () => void;
}

const pad = (value: number): string => String(value).padStart(2, "0");

/** epoch milliseconds → the value a datetime-local input expects (local time) */
const toInput = (ms?: number | null): string => {
  if (!ms) return "";
  const date = new Date(ms);
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(date.getHours())}:${pad(date.getMinutes())}`;
};

const fromInput = (value: string): number | null => {
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const INPUT_KEY = "incidentgraph.dashboard-target.v1";

const EMPTY_FORM = {
  host: "127.0.0.1",
  port: "6379",
  graph: "incidentgraph:demo",
  anchor: "service:checkoutservice",
  t0: "",
  t1: "",
  onset: "",
  symptom: "",
  username: "",
  password: "",
};

type Form = typeof EMPTY_FORM;

function readForm(): Form {
  try {
    const raw = window.localStorage.getItem(INPUT_KEY);
    return raw ? { ...EMPTY_FORM, ...(JSON.parse(raw) as Partial<Form>) } : EMPTY_FORM;
  } catch {
    return EMPTY_FORM;
  }
}

/** The graph contract the engine reads; shown so a caller can prepare theirs. */
const REQUIREMENTS: Array<[string, string]> = [
  ["Service nodes", "`(:Service {id})` — the anchor and every candidate; optional `criticality`."],
  ["Dependency edges", "`(:Service)-[:DEPENDS_ON]->(:Service)` — scope expansion and path checks."],
  ["Telemetry", "`(:MetricAnomaly)`, `(:LogEvent)`, `(:Span)` with a millisecond `ts`."],
  ["Ownership", "each record attached with `(:record)-[:OBSERVED_ON]->(:Service|:Instance)`."],
  ["Authoritative onset", "the time the incident was reported or injected. The engine refuses to invent a clock."],
  ["Timestamps", "epoch milliseconds, consistent with the incident window you enter."],
];

export function DashboardPage({ health, store, onFinished }: DashboardPageProps) {
  const [form, setForm] = useState<Form>(readForm);
  const [probe, setProbe] = useState<ProbeResult | null>(null);
  const [checking, setChecking] = useState(false);
  const [starting, setStarting] = useState<string | null>(null);

  useEffect(() => {
    window.localStorage.setItem(INPUT_KEY, JSON.stringify(form));
  }, [form]);

  const set = (key: keyof Form) => (value: string) =>
    setForm((previous) => ({ ...previous, [key]: value }));

  const key = `${form.host}:${form.port}/${form.graph}`;
  const snapshot = store.runs[key] ?? null;
  const elapsed = useElapsed(snapshot);
  const running = store.running;
  const runningElsewhere = Boolean(running && running.case_id !== key);
  const target = useMemo(
    () => ({ host: form.host.trim(), port: Number(form.port) || 6379, graph: form.graph.trim() }),
    [form.host, form.port, form.graph],
  );

  const check = useCallback(async () => {
    setChecking(true);
    store.setError(null);
    try {
      const result = await api.probe({ ...target, anchor_service: form.anchor.trim() || null });
      setProbe(result);
      if (result.time_range) {
        setForm((previous) => ({
          ...previous,
          t0: previous.t0 || toInput(result.time_range?.t0),
          t1: previous.t1 || toInput(result.time_range?.t1),
        }));
      }
    } catch (error) {
      store.setError(error instanceof Error ? error.message : "the database could not be checked");
    } finally {
      setChecking(false);
    }
  }, [form.anchor, store, target]);

  const run = useCallback(
    async (mode: "preview" | "full") => {
      const t0 = fromInput(form.t0);
      const t1 = fromInput(form.t1);
      const onset = fromInput(form.onset);
      if (t0 === null || t1 === null || t1 <= t0) {
        store.setError("a start and end time are required, with the end after the start");
        return;
      }
      if (onset === null) {
        store.setError(
          "an authoritative onset time is required: the engine will not invent a clock",
        );
        return;
      }
      if (onset < t0 || onset > t1) {
        store.setError("the onset must fall inside the window");
        return;
      }
      setStarting(mode);
      store.setError(null);
      try {
        await store.start(() =>
          api.startCustomRun({
            ...target,
            anchor_service: form.anchor.trim(),
            window_t0: t0,
            window_t1: t1,
            onset_ts: onset,
            symptom: form.symptom.trim() || null,
            mode,
            label: key,
            username: form.username.trim() || null,
            password: form.password || null,
          }),
        );
      } catch (error) {
        store.setError(error instanceof Error ? error.message : "the run could not be started");
      } finally {
        setStarting(null);
      }
    },
    [form, key, store, target],
  );

  useEffect(() => {
    if (snapshot && snapshot.status !== "running") onFinished();
  }, [snapshot?.run_id, snapshot?.status, onFinished, snapshot]);

  const ready = probe?.verdict === "ready";
  const canRun = ready && !running && !starting;
  const databaseReady = health?.database.state === "ok";
  const providerReady = health?.provider.available ?? false;

  return (
    <div className="page">
      <section className="page__intro">
        <h1>Investigate your own graph</h1>
        <p>
          Connect a reachable FalkorDB instance, state the incident window, and run the pipeline
          against a disposable copy. Your graph is only read: the run executes on a session copy
          that is deleted afterwards.
        </p>
      </section>

      <section className="panel">
        <h2>Target database</h2>
        <div className="fields">
          <label>
            <span>host</span>
            <input value={form.host} onChange={(event) => set("host")(event.target.value)} />
          </label>
          <label>
            <span>port</span>
            <input value={form.port} onChange={(event) => set("port")(event.target.value)} />
          </label>
          <label>
            <span>graph</span>
            <input value={form.graph} onChange={(event) => set("graph")(event.target.value)} />
          </label>
          <label>
            <span>anchor service</span>
            <input value={form.anchor} onChange={(event) => set("anchor")(event.target.value)} />
          </label>
        </div>
        <details className="fields__more">
          <summary>authentication, if your instance requires it</summary>
          <div className="fields">
            <label>
              <span>username</span>
              <input value={form.username} onChange={(event) => set("username")(event.target.value)} />
            </label>
            <label>
              <span>password</span>
              <input
                type="password"
                value={form.password}
                onChange={(event) => set("password")(event.target.value)}
              />
            </label>
          </div>
        </details>
        <div className="actions">
          <button type="button" className="run" disabled={checking} onClick={() => void check()}>
            {checking ? "checking" : "Check graph"}
          </button>
          <span className="actions__hint">
            read-only verification of the required schema
          </span>
        </div>

        {probe && (
          <div className="checks" data-verdict={probe.verdict}>
            <p className="checks__verdict">
              {probe.verdict === "ready"
                ? "This graph is ready to investigate."
                : (probe.detail ?? "This graph is not ready yet.")}
            </p>
            <ul>
              {probe.checks.map((item) => (
                <li key={item.name} data-status={item.status}>
                  <span className="checks__mark" aria-hidden="true">
                    {item.status === "pass" ? "●" : item.status === "warn" ? "◐" : "✕"}
                  </span>
                  <span className="checks__name">{item.name}</span>
                  <span className="checks__detail">{item.detail}</span>
                </li>
              ))}
            </ul>
            {probe.graphs.length > 0 && (
              <p className="checks__graphs">
                graphs on this instance: {probe.graphs.slice(0, 8).join(", ")}
                {probe.graphs.length > 8 ? " …" : ""}
              </p>
            )}
          </div>
        )}

        {probe?.topology && probe.topology.nodes.length > 0 && (
          <div className="probe-graph">
            <div className="probe-graph__head">
              <span>your graph, as read from the database</span>
              <span className="muted">
                {probe.topology.nodes.length} services, {probe.topology.edges.length} dependencies
                {probe.topology.derived_levels ? ", depth derived from the edges" : ""}
              </span>
            </div>
            <DependencyGraph
              topology={probe.topology}
              mark={{ id: form.anchor.trim() || null, label: "your anchor" }}
            />
          </div>
        )}
      </section>

      <section className="panel">
        <h2>Incident scope</h2>
        <div className="fields">
          <label>
            <span>window start (t0)</span>
            <input
              type="datetime-local"
              value={form.t0}
              onChange={(event) => set("t0")(event.target.value)}
            />
          </label>
          <label>
            <span>window end (t1)</span>
            <input
              type="datetime-local"
              value={form.t1}
              onChange={(event) => set("t1")(event.target.value)}
            />
          </label>
          <label>
            <span>onset (required)</span>
            <input
              type="datetime-local"
              value={form.onset}
              onChange={(event) => set("onset")(event.target.value)}
            />
          </label>
          <label className="fields__wide">
            <span>symptom, optional</span>
            <input
              placeholder="what an operator would report"
              value={form.symptom}
              onChange={(event) => set("symptom")(event.target.value)}
            />
          </label>
        </div>
        <div className="actions">
          <button
            type="button"
            className="run"
            disabled={!canRun}
            onClick={() => void run("preview")}
          >
            {starting === "preview" ? "starting" : "Preview ranking"}
          </button>
          <button
            type="button"
            className="run run--paid"
            disabled={!canRun || !providerReady}
            onClick={() => void run("full")}
          >
            {starting === "full" ? "starting" : "Run full investigation"}
          </button>
          <span className="actions__hint">
            preview runs deterministic ranking only, with no model calls
          </span>
        </div>
        {runningElsewhere && (
          <p className="notice notice--quiet">
            {running?.case_id} is still running. One run at a time.
          </p>
        )}
        {!databaseReady && (
          <p className="notice">
            the local runner database is offline
            {health?.database.detail ? ` (${health.database.detail})` : ""}
          </p>
        )}
        {store.error && <p className="notice">{store.error}</p>}
      </section>

      <section className="panel panel--plain">
        <StageLedger
          stages={snapshot?.stages ?? IDLE_STAGES}
          activity={snapshot?.activity ?? []}
          neverRun={!snapshot}
        />
        {snapshot && (
          <CostLine cost={snapshot.cost} status={snapshot.status} elapsed={elapsed} />
        )}
        {snapshot?.result && <Verdict snapshot={snapshot} />}
      </section>

      <section className="panel">
        <h2>Required graph schema</h2>
        <dl className="requirements">
          {REQUIREMENTS.map(([term, detail]) => (
            <div key={term}>
              <dt>{term}</dt>
              <dd>{detail}</dd>
            </div>
          ))}
        </dl>
        <p className="checks__graphs">
          The check above verifies each of these and names what is missing. Telemetry timestamps
          must be epoch milliseconds; the window you enter is compared against them directly.
        </p>
      </section>
    </div>
  );
}
