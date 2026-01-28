import type { MLResults, ModelPerformance, ModelRiskScoreSurvival } from "@/types/ml-results";
import { buildROCOverlaySVG, buildKMComparisonSVG, buildUpsetMatrixSVG, buildFeatureImportanceSVG, buildConfusionMatrixSVG } from "./chartToSvg";

const MODEL_LABELS: Record<string, string> = {
  rf: "Random Forest",
  svm: "SVM",
  xgboost: "XGBoost",
  knn: "KNN",
  mlp: "MLP",
  hard_vote: "Hard Voting",
  soft_vote: "Soft Voting",
};

export const RUN_LABELS = ["Run A", "Run B", "Run C", "Run D"];

type ModelKey = keyof ModelPerformance;

function toFiniteNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

function formatPValue(p: unknown): string {
  const n = toFiniteNumber(p);
  if (n == null) return "NA";
  return n < 0.001 ? n.toExponential(2) : n.toFixed(4);
}

function normalizeModelRiskScores(raw: unknown): ModelRiskScoreSurvival[] {
  if (Array.isArray(raw)) return raw.filter((m) => typeof (m as any)?.model === "string") as ModelRiskScoreSurvival[];
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    return Object.values(raw).filter((m) => typeof (m as any)?.model === "string") as ModelRiskScoreSurvival[];
  }
  return [];
}

export function computeFeatureIntersections(featureLists: string[][], runLabels: string[]) {
  const sets = featureLists.map((l) => new Set(l));
  const n = sets.length;
  const all = new Set<string>();
  sets.forEach((s) => s.forEach((f) => all.add(f)));

  const intersections: { label: string; count: number; features: string[]; runIndices: number[] }[] = [];

  for (let mask = 1; mask < (1 << n); mask++) {
    const runIndices: number[] = [];
    for (let i = 0; i < n; i++) if (mask & (1 << i)) runIndices.push(i);

    const features = [...all].filter((f) => {
      return sets.every((set, idx) => {
        const inSet = set.has(f);
        const shouldBe = runIndices.includes(idx);
        return inSet === shouldBe;
      });
    });

    if (features.length > 0) {
      intersections.push({
        label: runIndices.map((i) => runLabels[i]).join(" ∩ "),
        count: features.length,
        features,
        runIndices,
      });
    }
  }

  intersections.sort((a, b) => b.count - a.count);
  return intersections;
}

export function buildComparisonReportHTML(runs: { name: string; data: MLResults }[]) {
  const runLabels = RUN_LABELS.slice(0, runs.length);

  const models = Object.keys(MODEL_LABELS) as ModelKey[];
  const performanceRows = models
    .map((model) => {
      const row: Record<string, string> = { model: MODEL_LABELS[model] };
      runs.forEach((run, idx) => {
        const metrics = run.data.model_performance[model];
        row[`auroc${idx}`] = metrics?.auroc?.mean ? (metrics.auroc.mean * 100).toFixed(1) : "N/A";
        row[`accuracy${idx}`] = metrics?.accuracy?.mean ? (metrics.accuracy.mean * 100).toFixed(1) : "N/A";
        row[`sensitivity${idx}`] = metrics?.sensitivity?.mean ? (metrics.sensitivity.mean * 100).toFixed(1) : "N/A";
        row[`specificity${idx}`] = metrics?.specificity?.mean ? (metrics.specificity.mean * 100).toFixed(1) : "N/A";
        row[`f1${idx}`] = metrics?.f1_score?.mean ? (metrics.f1_score.mean * 100).toFixed(1) : "N/A";
      });
      return row;
    })
    .filter((d) => runs.some((_, idx) => d[`auroc${idx}`] !== "N/A"));

  const featureLists = runs.map((r) => r.data.selected_features ?? []);
  const intersections = computeFeatureIntersections(featureLists, runLabels);
  const commonAll = intersections.find((i) => i.runIndices.length === runs.length)?.features ?? [];

  const survivalSummary = runs.map((run) => {
    const survival = run.data.survival_analysis;
    const modelScores = normalizeModelRiskScores(survival?.model_risk_scores);
    const softVote = modelScores.find((m) => m.model === "soft_vote" || m.model === "ensemble");
    const stats = softVote?.stats;

    return {
      has: !!stats,
      logrank_p: stats ? formatPValue(stats.logrank_p) : "N/A",
      hr: stats ? (toFiniteNumber(stats.cox_hr)?.toFixed(2) ?? "N/A") : "N/A",
      hr_ci: stats
        ? `${toFiniteNumber(stats.cox_hr_lower)?.toFixed(2) ?? "?"} - ${toFiniteNumber(stats.cox_hr_upper)?.toFixed(2) ?? "?"}`
        : "N/A",
    };
  });

  const generatedAt = new Date().toLocaleString();

  const html = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ML Analysis Comparison Report (${runs.length} Runs)</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; line-height: 1.55; color: #0f172a; background: #f8fafc; padding: 2rem; }
    .container { max-width: 1200px; margin: 0 auto; }
    .header { background: linear-gradient(135deg, #0ea5e9, #6366f1); color: white; padding: 2rem; border-radius: 14px; margin-bottom: 1.5rem; }
    h1 { font-size: 1.8rem; margin-bottom: 0.4rem; }
    h2 { font-size: 1.15rem; margin: 1.6rem 0 0.75rem; padding-bottom: 0.4rem; border-bottom: 2px solid #e2e8f0; color: #334155; }
    .grid { display: grid; grid-template-columns: repeat(${Math.min(runs.length, 4)}, 1fr); gap: 0.9rem; }
    .card { background: white; border-radius: 12px; padding: 1.1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.08); border-left: 4px solid #0ea5e9; }
    .label { font-size: 0.72rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 0.35rem; color: #0ea5e9; }
    .muted { color: #64748b; font-size: 0.9rem; }
    table { width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
    th, td { padding: 0.7rem 0.9rem; border-bottom: 1px solid #e2e8f0; text-align: left; }
    th { background: #f1f5f9; font-size: 0.85rem; color: #475569; }
    td { font-size: 0.92rem; }
    .pill { display: inline-block; padding: 0.18rem 0.6rem; border-radius: 999px; background: #e0f2fe; color: #075985; font-size: 0.78rem; }
    .footer { text-align: center; margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #e2e8f0; color: #64748b; font-size: 0.85rem; }
    @media print { body { background: white; padding: 0; } table { break-inside: avoid; } }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>ML Analysis Comparison Report</h1>
      <div class="muted" style="color: rgba(255,255,255,0.9);">Comparing ${runs.length} runs • Generated: ${generatedAt}</div>
    </div>

    <div class="grid">
      ${runs
        .map(
          (run, idx) => `
        <div class="card" style="border-left-color: ${idx === 0 ? "#0ea5e9" : idx === 1 ? "#6366f1" : idx === 2 ? "#10b981" : "#f59e0b"};">
          <div class="label" style="color: ${idx === 0 ? "#0ea5e9" : idx === 1 ? "#6366f1" : idx === 2 ? "#10b981" : "#f59e0b"};">${runLabels[idx]}</div>
          <div style="font-weight: 700; word-break: break-all;">${run.name}</div>
          <div class="muted">Generated: ${new Date(run.data.metadata.generated_at).toLocaleDateString()}</div>
        </div>
      `
        )
        .join("")}
    </div>

    <h2>Model Performance Comparison</h2>
    <table>
      <thead>
        <tr>
          <th>Model</th>
          ${runs.map((_, idx) => `<th>AUROC (${runLabels[idx]})</th>`).join("")}
          ${runs.map((_, idx) => `<th>Accuracy (${runLabels[idx]})</th>`).join("")}
          ${runs.map((_, idx) => `<th>Sensitivity (${runLabels[idx]})</th>`).join("")}
          ${runs.map((_, idx) => `<th>Specificity (${runLabels[idx]})</th>`).join("")}
          ${runs.map((_, idx) => `<th>F1 (${runLabels[idx]})</th>`).join("")}
        </tr>
      </thead>
      <tbody>
        ${performanceRows
          .map(
            (row) => `
          <tr>
            <td>${row.model}</td>
            ${runs.map((_, idx) => `<td>${row[`auroc${idx}`]}%</td>`).join("")}
            ${runs.map((_, idx) => `<td>${row[`accuracy${idx}`]}%</td>`).join("")}
            ${runs.map((_, idx) => `<td>${row[`sensitivity${idx}`]}%</td>`).join("")}
            ${runs.map((_, idx) => `<td>${row[`specificity${idx}`]}%</td>`).join("")}
            ${runs.map((_, idx) => `<td>${row[`f1${idx}`]}%</td>`).join("")}
          </tr>
        `
          )
          .join("")}
      </tbody>
    </table>

    <h2>Survival Analysis (Ensemble Risk Score)</h2>
    <table>
      <thead>
        <tr>
          <th>Run</th>
          <th>Log-rank p</th>
          <th>Hazard Ratio</th>
          <th>HR 95% CI</th>
        </tr>
      </thead>
      <tbody>
        ${survivalSummary
          .map(
            (s, idx) => `
          <tr>
            <td>${runLabels[idx]}</td>
            <td>${s.logrank_p}</td>
            <td>${s.hr}</td>
            <td>${s.hr_ci}</td>
          </tr>
        `
          )
          .join("")}
      </tbody>
    </table>

    <h2>ROC Curve Comparison</h2>
    <div class="chart-container" style="text-align: center; margin: 1rem 0;">
      ${buildROCOverlaySVG(runs)}
    </div>

    <h2>Confusion Matrix Comparison (Soft Voting)</h2>
    <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin: 1rem 0;">
      ${runs.map((run, idx) => {
        const cm = run.data.model_performance.soft_vote?.confusion_matrix;
        return `
        <div style="flex: 0 0 auto;">
          <div style="text-align: center; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem; color: ${idx === 0 ? "#0ea5e9" : idx === 1 ? "#6366f1" : idx === 2 ? "#10b981" : "#f59e0b"};">${runLabels[idx]}</div>
          ${cm ? buildConfusionMatrixSVG(cm.tp, cm.tn, cm.fp, cm.fn, "Soft Vote") : '<div style="color: #64748b; text-align: center; padding: 2rem;">No data</div>'}
        </div>
      `;
      }).join("")}
    </div>

    <h2>Feature Importance Comparison</h2>
    <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin: 1rem 0;">
      ${runs.map((run, idx) => `
        <div class="chart-container" style="flex: 1; min-width: 280px; max-width: 500px;">
          <div style="text-align: center; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem; color: ${idx === 0 ? "#0ea5e9" : idx === 1 ? "#6366f1" : idx === 2 ? "#10b981" : "#f59e0b"};">${runLabels[idx]}</div>
          ${run.data.feature_importance && run.data.feature_importance.length > 0 
            ? buildFeatureImportanceSVG(run.data.feature_importance, 10, `Top 10 Features`)
            : '<div style="color: #64748b; text-align: center; padding: 2rem;">No feature data</div>'
          }
        </div>
      `).join("")}
    </div>

    <h2>Survival Analysis - Kaplan-Meier Curves</h2>
    <div style="display: flex; gap: 1rem; justify-content: center; flex-wrap: wrap; margin: 1rem 0;">
      <div class="chart-container">
        ${buildKMComparisonSVG(runs, "high")}
      </div>
      <div class="chart-container">
        ${buildKMComparisonSVG(runs, "low")}
      </div>
    </div>

    <h2>Feature Overlap (UpSet Matrix)</h2>
    <div class="chart-container" style="text-align: center; margin: 1rem 0;">
      ${buildUpsetMatrixSVG(runs)}
    </div>

    <h2>Feature Intersection Details</h2>
    <div class="muted" style="margin-bottom: 0.6rem;">Common to all runs: <span class="pill">${commonAll.length} features</span></div>
    <table>
      <thead>
        <tr>
          <th>Intersection</th>
          <th># Features</th>
          <th>Example features</th>
        </tr>
      </thead>
      <tbody>
        ${intersections
          .slice(0, 12)
          .map(
            (i) => `
          <tr>
            <td>${i.label}</td>
            <td>${i.count}</td>
            <td>${i.features.slice(0, 10).map((f) => `<span class="pill">${f}</span>`).join(" ")}${i.features.length > 10 ? ` <span class="muted">+${i.features.length - 10} more</span>` : ""}</td>
          </tr>
        `
          )
          .join("")}
      </tbody>
    </table>

    <div class="footer">
      <p>Multi-Method ML Classifier • Powered by <a href="https://accelbio.pt/" target="_blank" rel="noopener noreferrer" style="color: #0ea5e9; text-decoration: none;">AccelBio</a></p>
    </div>
  </div>
</body>
</html>`;

  return html;
}
