/** Number and time formatting shared by the ledger, cost line and activity. */

export const compact = (value?: number | null): string => {
  if (value === null || value === undefined) return "—";
  if (value < 1000) return String(value);
  if (value < 1_000_000) return `${(value / 1000).toFixed(value < 10_000 ? 1 : 0)}k`;
  return `${(value / 1_000_000).toFixed(1)}M`;
};

export const count = (value?: number | null): string =>
  value === null || value === undefined ? "0" : value.toLocaleString();

/** mm:ss from seconds. */
export const clock = (seconds?: number | null): string => {
  const total = Math.max(0, Math.round(seconds ?? 0));
  const minutes = Math.floor(total / 60);
  return `${String(minutes).padStart(2, "0")}:${String(total % 60).padStart(2, "0")}`;
};

export const millis = (value?: number | null): string =>
  value === null || value === undefined
    ? "—"
    : value < 1000
      ? `${value}ms`
      : `${(value / 1000).toFixed(1)}s`;

/** Epoch milliseconds as a local wall-clock time. */
export const timeOfDay = (value?: number | null): string => {
  if (!value) return "—";
  const date = new Date(value);
  const pad = (part: number) => String(part).padStart(2, "0");
  return `${pad(date.getHours())}:${pad(date.getMinutes())}:${pad(date.getSeconds())}`;
};

/** Epoch milliseconds as a local date and time, for provenance lines. */
export const dateTime = (value?: number | null): string => {
  if (!value) return "—";
  const date = new Date(value);
  const pad = (part: number) => String(part).padStart(2, "0");
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())} ` +
    `${pad(date.getHours())}:${pad(date.getMinutes())}`;
};
