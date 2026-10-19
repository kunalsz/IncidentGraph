import { clock, count } from "../lib/format";
import type { CostState, RunStatus } from "../types";

interface CostLineProps {
  cost?: CostState | null;
  status: RunStatus;
  /** Live elapsed seconds; falls back to the server's own wall time. */
  elapsed?: number | null;
}

/** What this run spent: provider calls, tokens and wall clock. */
export function CostLine({ cost, status, elapsed }: CostLineProps) {
  if (!cost) return null;
  const seconds = elapsed ?? cost.wall_s;
  return (
    <div className="cost">
      <span>
        <b>{count(cost.calls)}</b> provider calls
      </span>
      <span>
        <b>{count(cost.tokens_in)}</b> tokens in, <b>{count(cost.tokens_out)}</b> out
      </span>
      <span>
        <b>{clock(seconds)}</b> elapsed
      </span>
      <span>
        {status === "running" ? "running" : cost.compliant ? "within caps" : "caps exceeded"}
      </span>
    </div>
  );
}
