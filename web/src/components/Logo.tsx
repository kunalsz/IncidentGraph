/**
 * Project mark: three connected services with one highlighted cause.
 * A topology glyph, not decoration — it is what the pipeline reasons over.
 */
export function Logo({ size = 22 }: { size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      role="img"
      aria-label="IncidentGraph"
      focusable="false"
    >
      <g stroke="currentColor" strokeWidth="1.3" fill="none" opacity="0.75">
        <path d="M5.5 6.5 L12 12" />
        <path d="M18.5 6.5 L12 12" />
        <path d="M12 12 L12 18.5" />
      </g>
      <circle cx="5.5" cy="6.5" r="2.1" fill="none" stroke="currentColor" strokeWidth="1.3" />
      <circle cx="18.5" cy="6.5" r="2.1" fill="none" stroke="currentColor" strokeWidth="1.3" />
      <circle cx="12" cy="18.5" r="2.1" fill="currentColor" className="logo__cause" />
    </svg>
  );
}
