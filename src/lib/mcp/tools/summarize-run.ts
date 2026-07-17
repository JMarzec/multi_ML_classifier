import { defineTool } from "@lovable.dev/mcp-js";
import { z } from "zod";

type AnyRec = Record<string, unknown>;

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

function fmt(n: number | null): string {
  return n === null ? "n/a" : n.toFixed(4);
}

export default defineTool({
  name: "summarize_ml_run",
  title: "Summarize ML run",
  description:
    "Summarize a multi-ML-classifier results JSON: run metadata, preprocessing stats, per-model mean AUROC/accuracy/F1, top-N selected features, and best model.",
  inputSchema: {
    json: z.string().min(1).describe("Full ML results JSON as a string."),
    top_features: z
      .number()
      .int()
      .min(1)
      .max(200)
      .optional()
      .describe("How many top selected features to include (default 20)."),
  },
  annotations: { readOnlyHint: true, idempotentHint: true, openWorldHint: false },
  handler: ({ json, top_features }) => {
    let data: AnyRec;
    try {
      data = JSON.parse(json) as AnyRec;
    } catch (e) {
      return {
        content: [{ type: "text", text: `Invalid JSON: ${(e as Error).message}` }],
        isError: true,
      };
    }

    const topN = top_features ?? 20;
    const meta = (data.metadata as AnyRec) ?? {};
    const cfg = (meta.config as AnyRec) ?? {};
    const prep = (data.preprocessing as AnyRec) ?? {};
    const perf = (data.model_performance as AnyRec) ?? {};
    const features = Array.isArray(data.selected_features)
      ? (data.selected_features as string[])
      : [];

    const modelRows: Array<{
      model: string;
      auroc: number | null;
      accuracy: number | null;
      f1: number | null;
      sensitivity: number | null;
      specificity: number | null;
    }> = [];

    for (const [name, m] of Object.entries(perf)) {
      const mm = (m as AnyRec) ?? {};
      modelRows.push({
        model: name,
        auroc: num(((mm.auroc as AnyRec) ?? {}).mean),
        accuracy: num(((mm.accuracy as AnyRec) ?? {}).mean),
        f1: num(((mm.f1_score as AnyRec) ?? {}).mean),
        sensitivity: num(((mm.sensitivity as AnyRec) ?? {}).mean),
        specificity: num(((mm.specificity as AnyRec) ?? {}).mean),
      });
    }

    const bestByAuroc = modelRows
      .filter((r) => r.auroc !== null)
      .sort((a, b) => (b.auroc ?? -1) - (a.auroc ?? -1))[0];

    const lines: string[] = [];
    lines.push(`# ML Run Summary`);
    lines.push(`- Generated: ${meta.generated_at ?? "n/a"}`);
    lines.push(`- R version: ${meta.r_version ?? "n/a"}`);
    lines.push(`- Target variable: ${cfg.target_variable ?? "n/a"}`);
    lines.push(
      `- Mode: ${cfg.analysis_mode ?? (prep.full_training_mode ? "full_training" : "cv")}`
    );
    if (cfg.n_folds || cfg.n_repeats) {
      lines.push(`- CV: ${cfg.n_folds ?? "?"} folds × ${cfg.n_repeats ?? "?"} repeats`);
    }
    if (cfg.train_ratio !== undefined) lines.push(`- Train ratio: ${cfg.train_ratio}`);
    lines.push(`- Feature selection: ${cfg.feature_selection_method ?? "n/a"} (max ${cfg.max_features ?? "n/a"})`);
    lines.push("");
    lines.push(`## Preprocessing`);
    lines.push(`- Samples: ${prep.original_samples ?? "n/a"}`);
    lines.push(`- Features: ${prep.original_features ?? "n/a"}`);
    lines.push(`- Missing: ${prep.missing_values ?? "n/a"} (${prep.missing_pct ?? "n/a"}%)`);
    lines.push(`- Class distribution: ${JSON.stringify(prep.class_distribution ?? {})}`);
    lines.push("");
    lines.push(`## Model Performance (mean)`);
    lines.push(`| Model | AUROC | Accuracy | F1 | Sens | Spec |`);
    lines.push(`|---|---|---|---|---|---|`);
    for (const r of modelRows) {
      lines.push(
        `| ${r.model} | ${fmt(r.auroc)} | ${fmt(r.accuracy)} | ${fmt(r.f1)} | ${fmt(r.sensitivity)} | ${fmt(r.specificity)} |`
      );
    }
    lines.push("");
    if (bestByAuroc) {
      lines.push(`**Best model by AUROC:** ${bestByAuroc.model} (${fmt(bestByAuroc.auroc)})`);
    }
    lines.push("");
    lines.push(`## Top ${Math.min(topN, features.length)} selected features`);
    lines.push(features.slice(0, topN).map((f, i) => `${i + 1}. ${f}`).join("\n") || "(none)");

    return {
      content: [{ type: "text", text: lines.join("\n") }],
      structuredContent: {
        metadata: meta,
        preprocessing: prep,
        models: modelRows,
        best_model_by_auroc: bestByAuroc ?? null,
        selected_features_top: features.slice(0, topN),
        selected_features_count: features.length,
      },
    };
  },
});