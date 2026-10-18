import { compact, millis } from "../lib/format";
import type { ActivityEntry, StageState, StageStatus } from "../types";

const GLYPH: Record<StageStatus, string> = {
  idle: "○",
  running: "◐",
  done: "●",
  failed: "✕",
};

const WORD: Record<StageStatus, string> = {
  idle: "not run",
  running: "running",
  done: "done",
  failed: "failed",
};

/** The model calls made inside the running stage, newest last. */
function Activity({ entries }: { entries: ActivityEntry[] }) {
  return (
    <div className="activity">
      {entries.map((entry) => (
        <div key={entry.seq} className="activity__row" data-status={entry.status}>
          <span className="activity__task">{entry.task ?? "call"}</span>
          <span className="activity__meta">
            {millis(entry.ms)}
            {typeof entry.tokens_in === "number"
              ? ` · ${compact(entry.tokens_in)} in / ${compact(entry.tokens_out)} out`
              : ""}
          </span>
        </div>
      ))}
    </div>
  );
}

interface StageRowProps {
  stage: StageState;
  index: number;
  activity: ActivityEntry[];
}

function StageRow({ stage, index, activity }: StageRowProps) {
  const running = stage.status === "running";
  const recorded = activity.slice(-8);
  const line =
    stage.line ??
    (running && activity.length > 0
      ? `${activity.length} model call${activity.length === 1 ? "" : "s"} so far`
      : WORD[stage.status]);
  // Live calls are shown while the stage runs; after it finishes they stay
  // inspectable behind the same disclosure as the stage's own detail.
  const expandable = stage.detail.length > 0 || recorded.length > 0;

  return (
    <details className="stage" data-status={stage.status} data-expandable={expandable}>
      <summary>
        <span className="stage__index">{String(index).padStart(2, "0")}</span>
        <span className="stage__name">{stage.name}</span>
        <span className="stage__glyph" aria-hidden="true">
          {GLYPH[stage.status]}
        </span>
        <span className="stage__line">{line}</span>
      </summary>
      {running && recorded.length > 0 && <Activity entries={recorded.slice(-5)} />}
      {!running && expandable && (
        <div className="stage__disclosure">
          {recorded.length > 0 && <Activity entries={recorded} />}
          {stage.detail.length > 0 && (
            <dl className="stage__detail">
              {stage.detail.map((entry, position) => (
                <div key={`${entry.k}-${position}`}>
                  <dt>{entry.k}</dt>
                  <dd>{entry.v}</dd>
                </div>
              ))}
            </dl>
          )}
        </div>
      )}
    </details>
  );
}

interface StageLedgerProps {
  stages: StageState[];
  activity?: ActivityEntry[];
  neverRun?: boolean;
}

/** The seven pipeline stages as numbered ledger rows, each expandable. */
export function StageLedger({ stages, activity = [], neverRun }: StageLedgerProps) {
  return (
    <div className="ledger">
      {neverRun && <p className="ledger__note">This target has not been run yet.</p>}
      {stages.map((stage, index) => (
        <StageRow
          key={stage.id}
          stage={stage}
          index={index + 1}
          activity={activity.filter((entry) => entry.stage === stage.id)}
        />
      ))}
    </div>
  );
}
