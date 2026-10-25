import { readdirSync, readFileSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { describe, expect, it } from "vitest";

const SRC = join(__dirname, "..");
const ALLOWED = new Set(["theme/tokens.ts"]);   // the one file that defines colour values

function sourceFiles(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) return name === "test" ? [] : sourceFiles(p);
    return /\.(ts|tsx)$/.test(name) ? [p] : [];
  });
}

describe("design tokens", () => {
  it("defines colour values only in theme/tokens.ts", () => {
    const offenders = sourceFiles(SRC)
      .map((p) => relative(SRC, p).replace(/\\/g, "/"))
      .filter((rel) => !ALLOWED.has(rel))
      .filter((rel) => /#[0-9a-fA-F]{6}\b/.test(readFileSync(join(SRC, rel), "utf-8")));
    expect(offenders).toEqual([]);
  });
});
