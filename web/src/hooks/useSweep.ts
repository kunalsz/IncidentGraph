import { useEffect, useState } from "react";

/** True when the visitor asked the system to reduce motion. */
export function usePrefersReducedMotion(): boolean {
  const [reduced, setReduced] = useState(
    () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
  );

  useEffect(() => {
    const query = window.matchMedia("(prefers-reduced-motion: reduce)");
    const onChange = () => setReduced(query.matches);
    query.addEventListener("change", onChange);
    return () => query.removeEventListener("change", onChange);
  }, []);

  return reduced;
}

/** Index that advances on a timer, for animations driven by React state. */
export function useSweep(count: number, stepMs = 900, pauseMs = 2200): number {
  const reduced = usePrefersReducedMotion();
  const [index, setIndex] = useState(0);

  useEffect(() => {
    if (reduced) return;
    const timer = window.setTimeout(
      () => setIndex((current) => (current + 1) % (count + 1)),
      index === count ? pauseMs : stepMs,
    );
    return () => window.clearTimeout(timer);
  }, [index, count, stepMs, pauseMs, reduced]);

  return reduced ? count : index;
}
