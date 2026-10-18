import type { CaseInfo, StageStatus } from "../types";

export type CaseMark = "running" | "done" | "failed";

const GLYPH: Record<CaseMark, string> = {
  running: "◐",
  done: "●",
  failed: "✕",
};

interface CasePickerProps {
  cases: CaseInfo[];
  selected: string | null;
  marks: Record<string, CaseMark>;
  disabled: boolean;
  onSelect: (caseId: string) => void;
}

/** Each case carries its own state: a mark shows what has already been run. */
export function CasePicker({ cases, selected, marks, disabled, onSelect }: CasePickerProps) {
  return (
    <nav className="picker">
      <div className="picker__title">RCAEval cases</div>
      <div className="picker__list">
        {cases.map((item) => {
          const mark = marks[item.case];
          return (
            <button
              key={item.case}
              type="button"
              className="picker__case"
              aria-current={item.case === selected}
              disabled={disabled}
              onClick={() => onSelect(item.case)}
            >
              <span>
                {item.service} {item.fault} {item.instance}
              </span>
              {mark && (
                <span className="picker__mark" data-mark={mark} aria-label={mark}>
                  {GLYPH[mark]}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </nav>
  );
}

export type { StageStatus };
