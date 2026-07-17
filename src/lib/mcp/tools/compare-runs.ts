import { defineTool } from "@lovable.dev/mcp-js";
import { z } from "zod";

type AnyRec = Record<string, unknown>;

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}
function fmt(n: number | null): string {
  return n === null ? "n/a" : n.toFixed(4);
}
function diff(a: number | null, b: number | null): string {
  if (a === null || b === null) return "n/a";
  const d = b - a;
  const sign = d > 0 ? "+" : "";
  return `${sign}${d.toFixed(4)}`;
}

function extractModels(data: AnyRec) {
  const perf = (data.model_performance as AnyRec) ?? {};
  const out: Record<
    string,
    { auroc: number | null; accuracy: number | null; f1: number | null }
  > = {};
  for (const [name, m] of Object.entries(perf)) {
    const mm = (m as AnyRec) ?? {};
    out[name] = {
      auroc: num(((mm.auroc as AnyRec) ?? {}).mean),
      accuracy: num(((mm.accuracy as AnyRec) ?? {}).mean),
      f1: num(((mm.f1_score as AnyRec) ?? {}).mean),
    };
  }
  return out;
}

export default defineTool({
  name: "compare_ml_runs",
  title: "Compare two ML runs",
  description:
    "Compare two multi-ML-classifier results JSON payloads (run A vs run B). Reports per-model metric deltas (AUROC/Accuracy/F1) and selected-feature overlap.",
  inputSchema: {
    json_a: z.string().min(1).describe("Run A: full ML results JSON as a string."),
    json_b: z.string().min(1).describe("Run B: full ML results JSON as a string."),
    label_a: z.string().optional().describe("Optional label for run A (default 'A')."),
    label_b: z.string().optional().describe("Optional label for run B (default 'B')."),
  },
  annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
  handler: ({ json_a, json_b, label_a, label_b }) => {
    let a: AnyRec;
    let b: AnyRec;
    try {
      a = JSON.parse(json_a) as AnyRec;
      b = JSON.parse(json_b) as AnyRec;
    } catch (e) {
      return {
        content: [{ type: "text", text: `Invalid JSON: ${(e as Error).message}` }],
        isError: true,
      };
    }
    const A = label_a ?? "A";
    const B = label_b ?? "B";

    const ma = extractModels(a);
    const mb = extractModels(b);
    const modelNames = Array.from(new Set([...Object.keys(ma), ...Object.keys(mb)])).sort();

    const featA = new Set(Array.isArray(a.selected_features) ? (a.selected_features as string[]) : []);
    const featB = new Set(Array.isArray(b.selected_features) ? (b.selected_features as string[]) : []);
    const intersection = [...featA].filter((f) => featB.has(f));
    const onlyA = [...featA].filter((f) => !featB.has(f));
    const onlyB = [...featB].filter((f) => !featA.has(f));
    const union = new Set([...featA, ...featB]);
    const jaccard = union.size === 0 ? null : intersection.length / union.size;

    const lines: string[] = [];
    lines.push(`# Run comparison: ${A} vs ${B}`);
    lines.push("");
    lines.push(`## Model performance (mean AUROC / Accuracy / F1)`);
    lines.push(`| Model | ${A} AUROC | ${B} AUROC | Δ AUROC | ${A} Acc | ${B} Acc | Δ Acc | ${A} F1 | ${B} F1 | Δ F1 |`);
    lines.push(`|---|---|---|---|---|---|---|---|---|---|`);
    for (const name of modelNames) {
      const x = ma[name] ?? { auroc: null, accuracy: null, f1: null };
      const y = mb[name] ?? { auroc: null, accuracy: null, f1: null };
      lines.push(
        `| ${name} | ${fmt(x.auroc)} | ${fmt(y.auroc)} | ${diff(x.auroc, y.auroc)} | ${fmt(x.accuracy)} | ${fmt(y.accuracy)} | ${diff(x.accuracy, y.accuracy)} | ${fmt(x.f1)} | ${fmt(y.f1)} | ${diff(x.f1, y.f1)} |`
      );
    }
    lines.push("");
    lines.push(`## Selected feature overlap`);
    lines.push(`- ${A} features: ${featA.size}`);
    lines.push(`- ${B} features: ${featB.size}`);
    lines.push(`- Shared: ${intersection.length}`);
    lines.push(`- Only in ${A}: ${onlyA.length}`);
    lines.push(`- Only in ${B}: ${onlyB.length}`);
    lines.push(`- Jaccard: ${jaccard === null ? "n/a" : jaccard.toFixed(4)}`);

    return {
      content: [{ type: "text", text: lines.join("\n") }],
      structuredContent: {
        labels: { a: A, b: B },
        models: modelNames.map((name) => ({
          model: name,
          a: ma[name] ?? null,
          b: mb[name] ?? null,
        })),
        features: {
          a_count: featA.size,
          b_count: featB.size,
          shared: intersection,
          only_a: onlyA,
          only_b: onlyB,
          jaccard,
        },
      },
    };
  },
});