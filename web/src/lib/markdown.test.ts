import assert from "node:assert/strict";
import { test } from "node:test";

import { parseMarkdown } from "./markdown.ts";

test("headings, paragraphs and lists", () => {
  const source = "# Rules\n\nBases grow\nunits.\n\n## Ticks\n- one\n- two\n  continued\n1. first\n2. second\n";
  assert.deepEqual(parseMarkdown(source), [
    { kind: "heading", level: 1, text: "Rules" },
    { kind: "paragraph", text: "Bases grow units." },
    { kind: "heading", level: 2, text: "Ticks" },
    { kind: "list", ordered: false, items: ["one", "two continued"] },
    { kind: "list", ordered: true, items: ["first", "second"] },
  ]);
});

test("any line ending, and lines that only look like a block", () => {
  for (const ending of ["\r\n", "\r", "\r\r\n"]) {
    assert.deepEqual(parseMarkdown(["## Ticks", "Each tick is a tenth of a second."].join(ending)), [
      { kind: "heading", level: 2, text: "Ticks" },
      { kind: "paragraph", text: "Each tick is a tenth of a second." },
    ]);
  }
  assert.deepEqual(parseMarkdown("## Ti cks"), [{ kind: "paragraph", text: "## Ti cks" }]);
});
