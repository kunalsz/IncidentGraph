import { Markdown } from "./Markdown";
import type { RunResult, RunSnapshot } from "../types";

const sentence = (text: string): string =>
  /[.!?]$/.test(text) ? text : `${text}.`;

interface VerdictProps {
  snapshot: RunSnapshot;
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="verdict__row">
      <span className="verdict__label">{label}</span>
      <span className="verdict__value">{value}</span>
    </div>
  );
}

/** Deterministic ranking preview: candidates only, no verdict exists yet. */
function PreviewVerdict({ result }: { result: RunResult }) {
  return (
    <section className="verdict">
      <h2 className="verdict__tier verdict__tier--quiet">ranking only</h2>
      <p className="verdict__lede">
        Deterministic ranking over the graph. No model was invoked and no hypothesis was validated.
      </p>
      <ol className="ranking">
        {(result.candidates ?? []).map((candidate) => (
          <li key={candidate.id}>
            <span className="ranking__id">{candidate.id}</span>
            <span className="ranking__score">{candidate.score.toFixed(4)}</span>
          </li>
        ))}
      </ol>
      {result.truncated && (
        <p className="verdict__foot">The ranking hit a bound; lower-ranked candidates may be missing.</p>
      )}
      <p className="verdict__foot">{result.next_action}</p>
    </section>
  );
}

/**
 * The closing entry: what the pipeline concluded, why, and what it cost.
 * Labels are aligned so the conclusion can be read in one pass.
 */
export function Verdict({ snapshot }: VerdictProps) {
  const result = snapshot.result;
  if (!result) return null;
  if (result.mode === "preview") return <PreviewVerdict result={result} />;

  const decision = result.abstained
    ? "no candidate cleared the validation gate"
    : `${result.winner} selected`;
  const confidence =
    result.confidence === null || result.confidence === undefined
      ? "not scored"
      : `${result.confidence.toFixed(2)} (${result.confidence_rule ?? "unknown rule"})`;

  return (
    <section className="verdict" data-tier={result.tier ?? ""}>
      <h2 className="verdict__tier">{result.tier ?? "no verdict"}</h2>
      <p className="verdict__lede">{decision}</p>
      <dl className="verdict__facts">
        {result.reasons.length > 0 && (
          <Row label="reason" value={sentence(result.reasons[0])} />
        )}
        <Row label="confidence" value={confidence} />
        {result.next_action && <Row label="next step" value={result.next_action} />}
        {result.summary.slice(0, 1).map((line) => (
          <Row key={line} label="finding" value={sentence(line)} />
        ))}
        {snapshot.cleanup && (
          <Row
            label="cleanup"
            value={
              snapshot.cleanup === "deleted"
                ? "the copied session graph was deleted"
                : `session graph cleanup: ${snapshot.cleanup}`
            }
          />
        )}
        {snapshot.error && <Row label="stopped" value={snapshot.error} />}
      </dl>
      {result.markdown && (
        <details className="verdict__report">
          <summary>full investigation report</summary>
          <Markdown source={result.markdown} />
        </details>
      )}
    </section>
  );
}
