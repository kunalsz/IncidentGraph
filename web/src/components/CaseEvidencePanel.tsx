import { useCallback, useRef, useState } from "react";

import { api } from "../api/client";
import { dateTime, timeOfDay } from "../lib/format";
import type { CaseEvidence } from "../types";

/**
 * The case's raw telemetry, summarised: what actually went wrong, from the same
 * files the pipeline ingests. Loaded on first open, per case.
 */
export function CaseEvidencePanel({ caseId }: { caseId: string }) {
  const [evidence, setEvidence] = useState<CaseEvidence | null>(null);
  const [state, setState] = useState<"idle" | "loading" | "ready" | "error">("idle");
  const loading = useRef(false);

  const load = useCallback(() => {
    if (loading.current || evidence) return;
    loading.current = true;
    setState("loading");
    api
      .caseEvidence(caseId)
      .then((data) => {
        setEvidence(data);
        setState("ready");
      })
      .catch(() => {
        loading.current = false;
        setState("error");
      });
  }, [caseId, evidence]);

  const summary = evidence
    ? `${evidence.counts.metric_series} metric excursions, ` +
      `${evidence.counts.error_logs} error logs, ` +
      `${evidence.counts.spans.toLocaleString()} spans traced`
    : "inspect this case's metrics, logs and spans";

  return (
    <details
      className="evidence"
      onToggle={(event) => {
        if (event.currentTarget.open) load();
      }}
    >
      <summary>
        <span>telemetry evidence for this case</span>
        <span className="muted">{state === "loading" ? "reading the case files…" : summary}</span>
      </summary>

      {state === "error" && <p className="notice">the case telemetry could not be read</p>}

      {evidence && (
        <div className="evidence__body">
          <p className="evidence__lead">
            The fault was injected at {dateTime(evidence.onset_ms)}; the baseline is the preceding
            twenty minutes. A z-score measures how far a metric moved from that baseline.
          </p>

          <h3>Metric anomalies, ranked by anomaly score (|z|)</h3>
          <table>
            <thead>
              <tr>
                <th>service</th>
                <th>metric</th>
                <th>z</th>
                <th>direction</th>
                <th>baseline to peak</th>
                <th>when</th>
              </tr>
            </thead>
            <tbody>
              {evidence.metrics.map((row) => (
                <tr key={row.column}>
                  <td>{row.service}</td>
                  <td className="mono-cell">{row.metric ?? row.column}</td>
                  <td className="mono-cell">{row.z.toFixed(1)}</td>
                  <td>{row.direction === "up" ? "up" : row.direction === "down" ? "down" : "—"}</td>
                  <td className="mono-cell">
                    {row.baseline_mean !== null && row.baseline_mean !== undefined
                      ? `${row.baseline_mean.toPrecision(3)} → ${(row.value ?? 0).toPrecision(3)}`
                      : "—"}
                  </td>
                  <td className="mono-cell">{timeOfDay(row.ts)}</td>
                </tr>
              ))}
            </tbody>
          </table>

          <h3>Error logs near the onset</h3>
          {evidence.logs.total_errors === 0 ? (
            <p className="evidence__empty">
              No error-level log lines in this window; the evidence for this case is metric and
              trace based.
            </p>
          ) : (
            <>
              <p className="evidence__lead">
                {evidence.logs.total_errors} error lines in the window
                {evidence.logs.by_service.length > 0
                  ? `, mostly from ${evidence.logs.by_service
                      .slice(0, 3)
                      .map((item) => `${item.service} (${item.count})`)
                      .join(", ")}`
                  : ""}
                .
              </p>
              <ul className="evidence__list">
                {evidence.logs.samples.map((sample, index) => (
                  <li key={`${sample.ts}-${index}`}>
                    <span className="mono-cell">{timeOfDay(sample.ts)}</span>
                    <span>{sample.service}</span>
                    <span className="mono-cell">{sample.message}</span>
                  </li>
                ))}
              </ul>
            </>
          )}

          <h3>Traces</h3>
          <p className="evidence__lead">
            {evidence.spans.total.toLocaleString()} spans in the window
            {evidence.spans.admitted
              ? `; the pipeline admits a bounded sample of ${evidence.spans.admitted.toLocaleString()}`
              : ""}
            . {evidence.spans.errors > 0
              ? `${evidence.spans.errors.toLocaleString()} carry an error status.`
              : "No span carries an error status, so the observed symptom is latency degradation rather than failure."}
          </p>
          <table>
            <thead>
              <tr>
                <th>service</th>
                <th>spans</th>
                <th>errors</th>
                <th>p95</th>
              </tr>
            </thead>
            <tbody>
              {evidence.spans.services.map((row) => (
                <tr key={row.service}>
                  <td>{row.service}</td>
                  <td className="mono-cell">{row.spans.toLocaleString()}</td>
                  <td className="mono-cell">{row.errors}</td>
                  <td className="mono-cell">{row.p95_ms.toFixed(1)} ms</td>
                </tr>
              ))}
            </tbody>
          </table>
          {evidence.spans.slowest.length > 0 && (
            <ul className="evidence__list">
              {evidence.spans.slowest.slice(0, 4).map((span, index) => (
                <li key={`${span.ts}-${index}`}>
                  <span className="mono-cell">{span.ms.toFixed(0)} ms</span>
                  <span>{span.service}</span>
                  <span className="mono-cell">{span.op}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </details>
  );
}
