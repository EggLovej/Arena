// Just enough Markdown for a game's rules: headings, paragraphs, lists,
// **bold**, *italics* and `code`. Rules files are ours, so nothing fancier is needed.

import type { ReactNode } from "react";

function inline(text: string): ReactNode[] {
  const parts: ReactNode[] = [];
  const pattern = /(\*\*[^*]+\*\*|\*[^*]+\*|`[^`]+`)/g;
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    const index = match.index ?? 0;
    if (index > last) parts.push(text.slice(last, index));
    const token = match[0];
    if (token.startsWith("**")) parts.push(<strong key={index}>{token.slice(2, -2)}</strong>);
    else if (token.startsWith("`")) parts.push(<code key={index}>{token.slice(1, -1)}</code>);
    else parts.push(<em key={index}>{token.slice(1, -1)}</em>);
    last = index + token.length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

export function Markdown({ source }: { source: string }) {
  const blocks: ReactNode[] = [];
  const lines = source.replace(/\r\n/g, "\n").split("\n");
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index++;
      continue;
    }
    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      const level = Math.min(4, heading[1].length + 1);
      const Tag = `h${level}` as "h2" | "h3" | "h4";
      blocks.push(<Tag key={index}>{inline(heading[2])}</Tag>);
      index++;
      continue;
    }
    const ordered = /^\d+\.\s+/.test(line);
    if (ordered || /^[-*]\s+/.test(line)) {
      const items: string[] = [];
      while (index < lines.length && (ordered ? /^\d+\.\s+/ : /^[-*]\s+/).test(lines[index])) {
        let item = lines[index].replace(ordered ? /^\d+\.\s+/ : /^[-*]\s+/, "");
        index++;
        while (index < lines.length && /^\s{2,}\S/.test(lines[index])) item += " " + lines[index++].trim();
        items.push(item);
      }
      const children = items.map((item, number) => <li key={number}>{inline(item)}</li>);
      blocks.push(ordered ? <ol key={index}>{children}</ol> : <ul key={index}>{children}</ul>);
      continue;
    }
    const paragraph: string[] = [];
    while (index < lines.length && lines[index].trim() && !/^(#{1,4}\s|[-*]\s|\d+\.\s)/.test(lines[index])) {
      paragraph.push(lines[index].trim());
      index++;
    }
    blocks.push(<p key={index}>{inline(paragraph.join(" "))}</p>);
  }
  return <div className="prose">{blocks}</div>;
}
