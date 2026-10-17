// Just enough Markdown for a game's rules: headings, paragraphs and lists, split into
// blocks here and drawn by components/Markdown.tsx. Rules files are ours, so nothing
// fancier is needed.

export type Block =
  | { kind: "heading"; level: number; text: string }
  | { kind: "list"; ordered: boolean; items: string[] }
  | { kind: "paragraph"; text: string };

const ORDERED = /^\d+\.\s+/;
const BULLET = /^[-*]\s+/;
const STARTS_BLOCK = /^(#{1,4}\s|[-*]\s|\d+\.\s)/;

export function parseMarkdown(source: string): Block[] {
  const blocks: Block[] = [];
  const lines = source.split(/\r\n?|\n/);
  let index = 0;
  while (index < lines.length) {
    const line = lines[index];
    if (!line.trim()) {
      index++;
      continue;
    }
    const heading = /^(#{1,4})\s+(.*)$/.exec(line);
    if (heading) {
      blocks.push({ kind: "heading", level: heading[1].length, text: heading[2] });
      index++;
      continue;
    }
    const ordered = ORDERED.test(line);
    if (ordered || BULLET.test(line)) {
      const marker = ordered ? ORDERED : BULLET;
      const items: string[] = [];
      while (index < lines.length && marker.test(lines[index])) {
        let item = lines[index].replace(marker, "");
        index++;
        while (index < lines.length && /^\s{2,}\S/.test(lines[index])) item += " " + lines[index++].trim();
        items.push(item);
      }
      blocks.push({ kind: "list", ordered, items });
      continue;
    }
    // The first line always belongs to the paragraph, so every pass moves on.
    const paragraph = [line.trim()];
    index++;
    while (index < lines.length && lines[index].trim() && !STARTS_BLOCK.test(lines[index])) {
      paragraph.push(lines[index].trim());
      index++;
    }
    blocks.push({ kind: "paragraph", text: paragraph.join(" ") });
  }
  return blocks;
}
