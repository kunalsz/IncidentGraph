import type { CaseInfo, Health } from "../types";

interface RunControlsProps {
  caseInfo: CaseInfo | null;
  health: Health | null;
  running: boolean;
  runningCase: string | null;
  starting: boolean;
  runningElsewhere: boolean;
  sourceUrl?: string | null;
  onRun: () => void;
}

const meta = (item: CaseInfo): string =>
  `${item.fault} fault, instance ${item.instance}, 20 minute window`;

function blockedReason(health: Health | null): string | null {
  if (!health) return "the runner API is not reachable";
  if (health.database.state !== "ok") {
    return `the database is offline (${health.database.detail ?? "unknown"}); start FalkorDB and retry`;
  }
  if (!health.provider.available) {
    return `no live provider is configured (resolved to ${health.provider.name})`;
  }
  return null;
}

/** Case identity plus the single action the page offers. */
export function RunControls({
  caseInfo,
  health,
  running,
  runningCase,
  starting,
  runningElsewhere,
  sourceUrl,
  onRun,
}: RunControlsProps) {
  const blocked = blockedReason(health);
  const disabled = !caseInfo || running || starting || blocked !== null;

  return (
    <>
      <div className="case-head">
        <div>
          <h1 className="case-head__id">
            {caseInfo ? (
              sourceUrl ? (
                <a
                  className="case-head__link"
                  href={sourceUrl}
                  target="_blank"
                  rel="noreferrer"
                  title="the recorded case files in the source dataset"
                >
                  {caseInfo.case}
                </a>
              ) : (
                caseInfo.case
              )
            ) : (
              "no case selected"
            )}
          </h1>
          {caseInfo && <div className="case-head__meta">{meta(caseInfo)}</div>}
        </div>
        <button
          type="button"
          className="run"
          data-state={runningElsewhere ? "locked" : running ? "running" : "idle"}
          disabled={disabled}
          onClick={onRun}
        >
          {running ? "running" : starting ? "starting" : "Run pipeline"}
        </button>
      </div>
      {runningElsewhere && (
        <p className="notice notice--quiet">
          {runningCase} is still running. One run at a time; this case can start when it finishes.
        </p>
      )}
      {blocked && !running && <p className="notice">{blocked}</p>}
    </>
  );
}
