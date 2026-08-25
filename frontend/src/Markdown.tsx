import type { ReactNode } from "react";

/** A small markdown renderer for chat replies.
 *
 *  It emits React elements rather than HTML, so model output is never interpolated into the
 *  DOM as markup and no sanitiser is needed. It covers what a reply actually uses —
 *  headings, lists, fenced and inline code, bold, italic, links — and deliberately not the
 *  rest: tables and images in a chat transcript are a sign the answer belongs in an
 *  artifact, not in the message.
 *
 *  It also has to tolerate half-written input, because replies stream. An unclosed `**` or a
 *  fence with no terminator renders as plain text and resolves itself on the next token,
 *  instead of flickering between broken layouts.
 */
export function Markdown({ text }: { text: string }) {
  return <>{blocks(text)}</>;
}

function blocks(text: string): ReactNode[] {
  const lines = text.split("\n");
  const out: ReactNode[] = [];
  let index = 0;
  let key = 0;

  while (index < lines.length) {
    const line = lines[index];

    if (line.trimStart().startsWith("```")) {
      const body: string[] = [];
      index += 1;
      while (index < lines.length && !lines[index].trimStart().startsWith("```")) {
        body.push(lines[index]);
        index += 1;
      }
      index += 1; // Closing fence, or the end of a stream that has not written it yet.
      out.push(<pre key={key++}><code>{body.join("\n")}</code></pre>);
      continue;
    }

    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      const depth = heading[1].length;
      const content = inline(heading[2]);
      out.push(depth <= 2
        ? <h3 key={key++} className="md-h2">{content}</h3>
        : <h4 key={key++} className="md-h3">{content}</h4>);
      index += 1;
      continue;
    }

    if (/^\s*([-*•]|\d+\.)\s+/.test(line)) {
      const ordered = /^\s*\d+\./.test(line);
      const items: ReactNode[] = [];
      while (index < lines.length && /^\s*([-*•]|\d+\.)\s+/.test(lines[index])) {
        items.push(<li key={items.length}>{inline(lines[index].replace(/^\s*([-*•]|\d+\.)\s+/, ""))}</li>);
        index += 1;
      }
      out.push(ordered ? <ol key={key++} className="md-list">{items}</ol> : <ul key={key++} className="md-list">{items}</ul>);
      continue;
    }

    if (!line.trim()) {
      index += 1;
      continue;
    }

    // Consecutive non-blank lines are one paragraph, so a wrapped sentence does not become
    // several stacked blocks with gaps between them.
    const paragraph: string[] = [];
    while (
      index < lines.length && lines[index].trim()
      && !/^(#{1,4})\s+/.test(lines[index])
      && !/^\s*([-*•]|\d+\.)\s+/.test(lines[index])
      && !lines[index].trimStart().startsWith("```")
    ) {
      paragraph.push(lines[index]);
      index += 1;
    }
    out.push(<p key={key++} className="md-p">{inline(paragraph.join("\n"))}</p>);
  }
  return out;
}

const INLINE = /(`[^`]+`|\*\*[^*]+\*\*|__[^_]+__|\*[^*\n]+\*|\[[^\]]+\]\([^)]+\))/g;

function inline(text: string): ReactNode[] {
  const out: ReactNode[] = [];
  let cursor = 0;
  let key = 0;
  for (const match of text.matchAll(INLINE)) {
    const at = match.index ?? 0;
    if (at > cursor) out.push(text.slice(cursor, at));
    const token = match[0];
    if (token.startsWith("`")) {
      out.push(<code key={key++} className="md-code">{token.slice(1, -1)}</code>);
    } else if (token.startsWith("**") || token.startsWith("__")) {
      out.push(<strong key={key++}>{token.slice(2, -2)}</strong>);
    } else if (token.startsWith("[")) {
      const link = /^\[([^\]]+)\]\(([^)]+)\)$/.exec(token);
      out.push(link
        ? <a key={key++} href={link[2]} target="_blank" rel="noreferrer noopener">{link[1]}</a>
        : token);
    } else {
      out.push(<em key={key++}>{token.slice(1, -1)}</em>);
    }
    cursor = at + token.length;
  }
  if (cursor < text.length) out.push(text.slice(cursor));
  return out;
}
