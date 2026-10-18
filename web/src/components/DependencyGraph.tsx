import { useMemo, useState } from "react";

import { timeOfDay } from "../lib/format";
import type { Topology, TopologyNode } from "../types";

/* Layered layout: a column per dependency depth, callers on the left. */
const COLUMN_X = 176;
const ROW_GAP = 46;
const PAD_X = 92;
const PAD_Y = 40;
/** Room for the label text that sits to the right of the last column. */
const LABEL_SPACE = 132;
const CAPTION: Record<number, string> = {
  0: "ingress",
  1: "frontend",
  2: "services",
  3: "downstream",
};

interface Position {
  x: number;
  y: number;
}

function layout(topology: Topology) {
  const byLevel = new Map<number, TopologyNode[]>();
  for (const node of topology.nodes) {
    const list = byLevel.get(node.level) ?? [];
    list.push(node);
    byLevel.set(node.level, list);
  }
  const levels = [...byLevel.keys()].sort((a, b) => a - b);
  const tallest = Math.max(...[...byLevel.values()].map((list) => list.length), 1);
  const height = PAD_Y * 2 + (tallest - 1) * ROW_GAP;
  const positions = new Map<string, Position>();
  for (const level of levels) {
    // Most critical first, so the eye lands on what matters.
    const ordered = [...(byLevel.get(level) ?? [])].sort(
      (a, b) => b.criticality - a.criticality || a.name.localeCompare(b.name),
    );
    ordered.forEach((node, index) => {
      positions.set(node.id, {
        x: PAD_X + level * COLUMN_X,
        y: height / 2 + (index - (ordered.length - 1) / 2) * ROW_GAP,
      });
    });
  }
  return {
    positions,
    levels,
    width: PAD_X + Math.max(...levels, 0) * COLUMN_X + LABEL_SPACE,
    height,
  };
}

const curve = (from: Position, to: Position): string =>
  `M${from.x},${from.y} C${from.x + 46},${from.y} ${to.x - 46},${to.y} ${to.x},${to.y}`;

interface DependencyGraphProps {
  topology: Topology;
  /** Optional highlighted node: which id, and what the highlight means. */
  mark?: { id: string | null; label: string } | null;
}

/* Ids may or may not carry the `service:` prefix depending on the graph. */
const bare = (value?: string | null): string => (value ?? "").replace(/^service:/, "");

/** The dependency map a case shares, or the one read from your own database.
 *  Drawn once and inspectable per node. */
export function DependencyGraph({ topology, mark }: DependencyGraphProps) {
  const [selected, setSelected] = useState<string | null>(null);
  const { positions, levels, width, height } = useMemo(() => layout(topology), [topology]);

  const byId = new Map(topology.nodes.map((node) => [node.id, node]));
  const outgoing = (id: string) => topology.edges.filter((edge) => edge.source === id);
  const incoming = (id: string) => topology.edges.filter((edge) => edge.target === id);
  const node = selected ? byId.get(selected) ?? null : null;
  const markedId = mark?.id ? bare(mark.id) : null;

  return (
    <div className="graph">
      <div className="graph__scroll">
        <svg viewBox={`0 0 ${width} ${height}`} width={width} height={height}
             role="img" aria-label="Service dependency graph">
          {levels.map((level) => (
            <text key={level} className="graph__caption"
                  x={PAD_X + level * COLUMN_X} y={16}>
              {CAPTION[level] ?? `level ${level}`}
            </text>
          ))}

          {topology.edges.map((edge) => {
            const from = positions.get(edge.source);
            const to = positions.get(edge.target);
            if (!from || !to) return null;
            const active = selected !== null
              && (edge.source === selected || edge.target === selected);
            return (
              <path key={`${edge.source}->${edge.target}`}
                    className={active ? "graph__edge graph__edge--active" : "graph__edge"}
                    d={curve(from, to)} />
            );
          })}

          {topology.nodes.map((item) => {
            const position = positions.get(item.id);
            if (!position) return null;
            const isMarked = markedId !== null && bare(item.id) === markedId;
            const isSelected = selected === item.id;
            return (
              <g key={item.id} className="graph__node" data-marked={isMarked}
                 data-selected={isSelected}
                 onClick={() => setSelected(isSelected ? null : item.id)}
                 onMouseDown={(event) => event.preventDefault()}
                 onKeyDown={(event) => {
                   if (event.key === "Enter" || event.key === " ") {
                     event.preventDefault();
                     setSelected(isSelected ? null : item.id);
                   }
                 }}
                 role="button" tabIndex={0}
                 aria-label={`${item.name}, level ${item.level}`}>
                <circle cx={position.x} cy={position.y} r={isMarked ? 7 : 5.5} />
                <text x={position.x + 14} y={position.y + 4}>{item.name}</text>
              </g>
            );
          })}
        </svg>
      </div>

      <div className="graph__legend">
        {markedId !== null && (
          <span>
            <i data-kind="marked" /> {mark?.label ?? "highlighted"}
          </span>
        )}
        <span><i data-kind="service" /> service</span>
        <span className="muted">select a service to inspect its dependencies</span>
      </div>

      {node && (
        <dl className="graph__inspector">
          <div>
            <dt>service</dt>
            <dd>{node.name}</dd>
          </div>
          <div>
            <dt>depth</dt>
            <dd>{node.level} ({CAPTION[node.level] ?? "level " + node.level})</dd>
          </div>
          <div>
            <dt>criticality</dt>
            <dd>{node.criticality.toFixed(2)}</dd>
          </div>
          <div>
            <dt>depends on</dt>
            <dd>
              {outgoing(node.id).length === 0
                ? "no downstream dependencies"
                : outgoing(node.id).map((edge) => byId.get(edge.target)?.name ?? edge.target).join(", ")}
            </dd>
          </div>
          <div>
            <dt>called by</dt>
            <dd>
              {incoming(node.id).length === 0
                ? "no upstream callers"
                : incoming(node.id).map((edge) => byId.get(edge.source)?.name ?? edge.source).join(", ")}
            </dd>
          </div>
        </dl>
      )}
    </div>
  );
}

export type { Topology };
export { timeOfDay };
