// Draws a game's rules: blocks from lib/markdown.ts, with **bold**, *italics* and `code`.

import type { ReactNode } from "react";

import { parseMarkdown } from "@/lib/markdown.ts";

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
  const blocks = parseMarkdown(source).map((block, index) => {
    if (block.kind === "heading") {
      const Tag = `h${Math.min(4, block.level + 1)}` as "h2" | "h3" | "h4";
      return <Tag key={index}>{inline(block.text)}</Tag>;
    }
    if (block.kind === "list") {
      const children = block.items.map((item, number) => <li key={number}>{inline(item)}</li>);
      return block.ordered ? <ol key={index}>{children}</ol> : <ul key={index}>{children}</ul>;
    }
    return <p key={index}>{inline(block.text)}</p>;
  });
  return <div className="prose">{blocks}</div>;
}
