import { Fragment, type ReactNode } from "react";

/**
 * Small renderer for the engine's report markdown.
 *
 * It emits React elements rather than HTML, so nothing from a graph — service
 * names, log messages, evidence text — can inject markup. The engine's renderer
 * uses a deliberately small subset: `#`/`##` headings, bullet lists, `**bold**`,
 * fenced blocks, and Pandoc-style backslash escapes plus HTML entities.
 */

const ENTITY: Record<string, string> = {
  "&gt;": ">",
  "&lt;": "<",
  "&amp;": "&",
  "&quot;": '"',
  "&#39;": "'",
};

export const unescapeText = (text: string): string =>
  text
    .replace(/\\([\\`*_{}[\]()#+\-.!:"'<>|~])/g, "$1")
    .replace(/&(gt|lt|amp|quot|#39);/g, (match) => ENTITY[match] ?? match);

function inline(text: string, keyPrefix: string): ReactNode[] {
  return text
    .split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
    .filter((part) => part !== "")
    .map((part, index) => {
      const key = `${keyPrefix}-${index}`;
      if (part.startsWith("**") && part.endsWith("**") && part.length > 4) {
        return <strong key={key}>{unescapeText(part.slice(2, -2))}</strong>;
      }
      if (part.startsWith("`") && part.endsWith("`") && part.length > 2) {
        return <code key={key}>{unescapeText(part.slice(1, -1))}</code>;
      }
      return <Fragment key={key}>{unescapeText(part)}</Fragment>;
    });
}

export function Markdown({ source }: { source: string }) {
  const lines = source.replace(/\r\n/g, "\n").split("\n");
  const blocks: ReactNode[] = [];
  let list: string[] = [];
  let ordered = false;
  let paragraph: string[] = [];
  let code: string[] | null = null;

  const flushList = () => {
    if (!list.length) return;
    const key = `list-${blocks.length}`;
    const items = list.map((item, index) => (
      <li key={`${key}-${index}`}>{inline(item, `${key}-${index}`)}</li>
    ));
    blocks.push(ordered ? <ol key={key}>{items}</ol> : <ul key={key}>{items}</ul>);
    list = [];
    ordered = false;
  };

  const flushParagraph = () => {
    if (!paragraph.length) return;
    const key = `p-${blocks.length}`;
    blocks.push(<p key={key}>{inline(paragraph.join(" "), key)}</p>);
    paragraph = [];
  };

  for (const raw of lines) {
    const line = raw.trimEnd();

    if (code !== null) {
      if (line.startsWith("```")) {
        blocks.push(
          <pre key={`code-${blocks.length}`}>
            <code>{code.join("\n")}</code>
          </pre>,
        );
        code = null;
      } else {
        code.push(raw);
      }
      continue;
    }

    if (line.startsWith("```")) {
      flushList();
      flushParagraph();
      code = [];
      continue;
    }

    if (!line.trim()) {
      flushList();
      flushParagraph();
      continue;
    }

    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      flushList();
      flushParagraph();
      const level = heading[1].length;
      const content = inline(heading[2], `h-${blocks.length}`);
      const key = `h-${blocks.length}`;
      if (level === 1) blocks.push(<h1 key={key}>{content}</h1>);
      else if (level === 2) blocks.push(<h2 key={key}>{content}</h2>);
      else blocks.push(<h3 key={key}>{content}</h3>);
      continue;
    }

    const bullet = /^\s*[-*]\s+(.*)$/.exec(line);
    if (bullet) {
      flushParagraph();
      if (ordered) flushList();
      list.push(bullet[1]);
      continue;
    }

    const numbered = /^\s*\d+[.)]\s+(.*)$/.exec(line);
    if (numbered) {
      flushParagraph();
      if (!ordered) flushList();
      ordered = true;
      list.push(numbered[1]);
      continue;
    }

    flushList();
    paragraph.push(line.trim());
  }

  if (code !== null && code.length) {
    blocks.push(
      <pre key={`code-${blocks.length}`}>
        <code>{code.join("\n")}</code>
      </pre>,
    );
  }
  flushList();
  flushParagraph();

  return <div className="markdown">{blocks}</div>;
}
