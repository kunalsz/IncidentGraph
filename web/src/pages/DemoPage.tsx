import { useCallback, useEffect, useMemo, useState } from "react";

import { api } from "../api/client";
import { CaseEvidencePanel } from "../components/CaseEvidencePanel";
import { CasePicker, type CaseMark } from "../components/CasePicker";
import { CostLine } from "../components/CostLine";
import { DependencyGraph } from "../components/DependencyGraph";
import { RunControls } from "../components/RunControls";
import { StageLedger } from "../components/StageLedger";
import { Verdict } from "../components/Verdict";
import type { RunStoreApi } from "../hooks/useRunStore";
import { useElapsed } from "../hooks/useElapsed";
import { IDLE_STAGES } from "../lib/pipeline";
import type { CaseInfo, Health, Topology } from "../types";

const SELECTED_KEY = "incidentgraph.selected-case.v1";

interface DemoPageProps {
  health: Health | null;
  store: RunStoreApi;
  onFinished: () => void;
}

export function DemoPage({ health, store, onFinished }: DemoPageProps) {
  const [cases, setCases] = useState<CaseInfo[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(
    () => window.localStorage.getItem(SELECTED_KEY),
  );
  const [starting, setStarting] = useState(false);
  const [topology, setTopology] = useState<Topology | null>(null);
  const setStoreError = store.setError;

  useEffect(() => {
    const controller = new AbortController();
    api
      .topology(controller.signal)
      .then(setTopology)
      .catch(() => undefined);
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    api
      .cases(controller.signal)
      .then((list) => {
        setCases(list);
        setSelected((current) =>
          current && list.some((item) => item.case === current) ? current : list[0]?.case ?? null,
        );
        setListError(null);
      })
      .catch(() => {
        // An abort is this effect cleaning up, not a failure to report.
        if (!controller.signal.aborted) setListError("the case list could not be loaded");
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (selected) window.localStorage.setItem(SELECTED_KEY, selected);
  }, [selected]);

  const snapshot = selected ? store.runs[selected] ?? null : null;
  const running = store.running;
  const runningElsewhere = Boolean(running && running.case_id !== selected);
  const elapsed = useElapsed(snapshot);

  const marks = useMemo(() => {
    const result: Record<string, CaseMark> = {};
    for (const [key, item] of Object.entries(store.runs)) {
      result[key] =
        item.status === "running" ? "running" : item.status === "completed" ? "done" : "failed";
    }
    return result;
  }, [store.runs]);

  const start = useCallback(async () => {
    if (!selected) return;
    setStarting(true);
    try {
      await store.start(() => api.startRun(selected));
    } catch (error) {
      setStoreError(error instanceof Error ? error.message : "the run could not be started");
    } finally {
      setStarting(false);
    }
  }, [selected, store, setStoreError]);

  useEffect(() => {
    if (snapshot && snapshot.status !== "running") onFinished();
  }, [snapshot?.run_id, snapshot?.status, onFinished, snapshot]);

  const caseInfo = cases.find((item) => item.case === selected) ?? null;
  const marked = Boolean(snapshot);

  return (
    <>
      <section className="graph-panel">
        <details className="graph-block" open>
          <summary>
            <span>service dependency graph</span>
            <span className="muted">
              {topology ? topology.note : "resolving the dependency map…"}
            </span>
          </summary>
          {topology ? (
            <DependencyGraph
              topology={topology}
              mark={{
                id: caseInfo ? `service:${caseInfo.service}` : null,
                label: "injected fault target",
              }}
            />
          ) : (
            <p className="muted">resolving the dependency map…</p>
          )}
        </details>
      </section>

      <div className="frame">
        <CasePicker
          cases={cases}
          selected={selected}
          marks={marks}
          disabled={Boolean(running)}
          onSelect={setSelected}
        />
        <main>
          <RunControls
            caseInfo={caseInfo}
            health={health}
            running={Boolean(running)}
            runningCase={running?.case_id ?? null}
            starting={starting}
            runningElsewhere={runningElsewhere}
            sourceUrl={caseInfo?.source_url ?? null}
            onRun={() => void start()}
          />
          {selected && <CaseEvidencePanel key={selected} caseId={selected} />}
          <StageLedger
            stages={snapshot?.stages ?? IDLE_STAGES}
            activity={snapshot?.activity ?? []}
            neverRun={!marked}
          />
          {snapshot && (
            <CostLine cost={snapshot.cost} status={snapshot.status} elapsed={elapsed} />
          )}
          {listError && <p className="notice">{listError}</p>}
          {store.error && <p className="notice">{store.error}</p>}
          {snapshot?.result && <Verdict snapshot={snapshot} />}
        </main>
      </div>
    </>
  );
}
