import type { ConfusionMatrixData, KaplanMeierPoint, MLResults, ModelRiskScoreSurvival, PerGeneSurvival, ProfileRanking } from "@/types/ml-results";

export function toFiniteNumber(value: unknown): number | undefined {
  if (typeof value !== "number" && (typeof value !== "string" || !value.trim())) return undefined;
  const n = Number(value);
  return Number.isFinite(n) ? n : undefined;
}
export function formatPValue(value: unknown): string {
  const n = toFiniteNumber(value);
  return n === undefined ? "N/A" : n < 0.001 ? n.toExponential(2) : n.toFixed(4);
}
export function formatNumber(value: unknown, decimals = 3): string {
  return toFiniteNumber(value)?.toFixed(decimals) ?? "N/A";
}
function record(value: unknown): Record<string, unknown> | undefined {
  return value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : undefined;
}
function entries(raw: unknown): [string, unknown][] {
  return Array.isArray(raw) ? raw.map((v, i) => [String(i), v]) : Object.entries(record(raw) ?? {});
}
function stats(raw: unknown) {
  const value = record(raw) ?? {};
  return {
    logrank_p: toFiniteNumber(value.logrank_p) ?? NaN,
    cox_hr: toFiniteNumber(value.cox_hr) ?? NaN,
    cox_hr_lower: toFiniteNumber(value.cox_hr_lower) ?? NaN,
    cox_hr_upper: toFiniteNumber(value.cox_hr_upper) ?? NaN,
    cox_p: toFiniteNumber(value.cox_p) ?? NaN,
  };
}
export function normalizePerGene(raw: unknown): PerGeneSurvival[] {
  return entries(raw).flatMap(([key, item]) => {
    const g = record(item);
    if (!g) return [];
    const gene = typeof g.gene === "string" ? g.gene : Array.isArray(raw) ? "" : key;
    if (!gene) return [];
    return [{ ...g, gene, ...stats(g), high_median_surv: toFiniteNumber(g.high_median_surv) ?? null, low_median_surv: toFiniteNumber(g.low_median_surv) ?? null }];
  });
}
function curve(raw: unknown): KaplanMeierPoint[] {
  if (!Array.isArray(raw)) return [];
  return raw.flatMap(item => {
    const p = record(item);
    if (!p) return [];
    const time = toFiniteNumber(p.time), surv = toFiniteNumber(p.surv);
    if (time === undefined || surv === undefined) return [];
    return [{ time, surv, lower: toFiniteNumber(p.lower) ?? surv, upper: toFiniteNumber(p.upper) ?? surv,
      n_risk: toFiniteNumber(p.n_risk) ?? 0, n_event: toFiniteNumber(p.n_event) ?? 0, n_censor: toFiniteNumber(p.n_censor) ?? 0 }];
  });
}
export function normalizeModelRiskScores(raw: unknown): ModelRiskScoreSurvival[] {
  return entries(raw).flatMap(([key, item]) => {
    const m = record(item);
    if (!m) return [];
    const model = typeof m.model === "string" ? m.model : Array.isArray(raw) ? "" : key;
    if (!model) return [];
    return [{ model, stats: stats(m.stats), km_curve_high: curve(m.km_curve_high), km_curve_low: curve(m.km_curve_low),
      cutoff: toFiniteNumber(m.cutoff), high_group_rule: typeof m.high_group_rule === "string" ? m.high_group_rule : undefined }];
  });
}
/** Never infer a model's cutoff from a different model or a fixed 0.5 threshold. */
export function patientRiskCohort(data: MLResults, patient: ProfileRanking): "high" | "low" | undefined {
  const model = normalizeModelRiskScores(data.survival_analysis?.model_risk_scores).find(m => m.model === "soft_vote" || m.model === "ensemble");
  const cutoff = model?.cutoff;
  const probability = toFiniteNumber(patient.ensemble_probability);
  if (cutoff === undefined || probability === undefined) return undefined;
  const strict = model?.high_group_rule?.includes("> cutoff") && !model.high_group_rule.includes(">= cutoff");
  return (strict ? probability > cutoff : probability >= cutoff) ? "high" : "low";
}
export function confusionMetrics({tp, tn, fp, fn}: ConfusionMatrixData) {
  const ratio = (n: number, d: number) => d > 0 ? n / d : undefined;
  return { accuracy: ratio(tp + tn, tp + tn + fp + fn), precision: ratio(tp, tp + fp),
    recall: ratio(tp, tp + fn), specificity: ratio(tn, tn + fp), f1: ratio(2 * tp, 2 * tp + fp + fn), npv: ratio(tn, tn + fn) };
}
export function metricPercent(value: number | undefined): string {
  return value === undefined ? "N/A" : `${(value * 100).toFixed(1)}%`;
}