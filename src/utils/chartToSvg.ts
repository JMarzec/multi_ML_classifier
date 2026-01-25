import type { MLResults, ModelPerformance, ModelRiskScoreSurvival } from "@/types/ml-results";

const RUN_COLORS_HEX = ["#0ea5e9", "#8b5cf6", "#10b981", "#f59e0b"];
const RUN_LABELS = ["Run A", "Run B", "Run C", "Run D"];

type ModelKey = keyof ModelPerformance;

function toFiniteNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

function normalizeModelRiskScores(raw: unknown): ModelRiskScoreSurvival[] {
  if (Array.isArray(raw)) return raw.filter((m) => typeof (m as any)?.model === "string") as ModelRiskScoreSurvival[];
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    return Object.values(raw).filter((m) => typeof (m as any)?.model === "string") as ModelRiskScoreSurvival[];
  }
  return [];
}

// Build ROC Overlay SVG
export function buildROCOverlaySVG(runs: { name: string; data: MLResults }[], model: ModelKey = "soft_vote"): string {
  const width = 500;
  const height = 400;
  const margin = { top: 30, right: 120, bottom: 50, left: 60 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  // Interpolate ROC curves to common FPR points
  const fprPoints = Array.from({ length: 101 }, (_, i) => i / 100);

  const curvesData = runs.map((run, idx) => {
    const roc = run.data.model_performance[model]?.roc_curve;
    if (!roc || roc.length === 0) return null;

    const sorted = [...roc].sort((a, b) => a.fpr - b.fpr);

    return fprPoints.map((fpr) => {
      let lower = sorted[0];
      let upper = sorted[sorted.length - 1];

      for (let i = 0; i < sorted.length - 1; i++) {
        if (sorted[i].fpr <= fpr && sorted[i + 1].fpr >= fpr) {
          lower = sorted[i];
          upper = sorted[i + 1];
          break;
        }
      }

      const tpr = upper.fpr === lower.fpr
        ? lower.tpr
        : lower.tpr + ((fpr - lower.fpr) / (upper.fpr - lower.fpr)) * (upper.tpr - lower.tpr);

      return { fpr, tpr };
    });
  });

  const aurocs = runs.map((run) => {
    const val = run.data.model_performance[model]?.auroc?.mean;
    return val ? (val * 100).toFixed(1) : "N/A";
  });

  // Build paths
  const buildPath = (points: { fpr: number; tpr: number }[]) => {
    return points
      .map((p, i) => {
        const x = margin.left + p.fpr * plotWidth;
        const y = margin.top + (1 - p.tpr) * plotHeight;
        return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
  };

  const paths = curvesData
    .map((curve, idx) => {
      if (!curve) return "";
      return `<path d="${buildPath(curve)}" fill="none" stroke="${RUN_COLORS_HEX[idx]}" stroke-width="2.5" />`;
    })
    .join("\n");

  // Diagonal reference line
  const diagLine = `<line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${margin.left + plotWidth}" y2="${margin.top}" stroke="#94a3b8" stroke-dasharray="5 5" stroke-width="1" />`;

  // Axes
  const xAxis = `<line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${margin.left + plotWidth}" y2="${margin.top + plotHeight}" stroke="#475569" stroke-width="1" />`;
  const yAxis = `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" stroke="#475569" stroke-width="1" />`;

  // Axis labels
  const xTicks = [0, 25, 50, 75, 100].map((v) => {
    const x = margin.left + (v / 100) * plotWidth;
    return `<text x="${x}" y="${margin.top + plotHeight + 18}" text-anchor="middle" font-size="10" fill="#64748b">${v}%</text>`;
  }).join("");

  const yTicks = [0, 25, 50, 75, 100].map((v) => {
    const y = margin.top + (1 - v / 100) * plotHeight;
    return `<text x="${margin.left - 8}" y="${y + 3}" text-anchor="end" font-size="10" fill="#64748b">${v}%</text>`;
  }).join("");

  const xLabel = `<text x="${margin.left + plotWidth / 2}" y="${height - 8}" text-anchor="middle" font-size="11" fill="#475569">False Positive Rate</text>`;
  const yLabel = `<text x="14" y="${margin.top + plotHeight / 2}" text-anchor="middle" font-size="11" fill="#475569" transform="rotate(-90 14 ${margin.top + plotHeight / 2})">True Positive Rate</text>`;

  // Legend
  const legend = runs
    .map((run, idx) => {
      const y = margin.top + 15 + idx * 22;
      return `
        <rect x="${margin.left + plotWidth + 15}" y="${y - 6}" width="14" height="14" rx="2" fill="${RUN_COLORS_HEX[idx]}" />
        <text x="${margin.left + plotWidth + 34}" y="${y + 5}" font-size="10" fill="#334155">${RUN_LABELS[idx]} (${aurocs[idx]}%)</text>
      `;
    })
    .join("");

  const title = `<text x="${width / 2}" y="18" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b">ROC Overlay - Soft Voting Ensemble</text>`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <rect width="${width}" height="${height}" fill="white" />
    ${title}
    ${xAxis}
    ${yAxis}
    ${diagLine}
    ${paths}
    ${xTicks}
    ${yTicks}
    ${xLabel}
    ${yLabel}
    ${legend}
  </svg>`;
}

// Build Kaplan-Meier Comparison SVG
export function buildKMComparisonSVG(runs: { name: string; data: MLResults }[], riskLevel: "high" | "low"): string {
  const width = 450;
  const height = 320;
  const margin = { top: 40, right: 100, bottom: 50, left: 60 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  // Collect all time points and curves
  const curves = runs.map((run) => {
    const survival = run.data.survival_analysis;
    const modelScores = normalizeModelRiskScores(survival?.model_risk_scores);
    const softVote = modelScores.find((m) => m.model === "soft_vote" || m.model === "ensemble");
    const curve = riskLevel === "high" ? softVote?.km_curve_high : softVote?.km_curve_low;
    return curve || [];
  });

  const allTimes = new Set<number>();
  curves.forEach((c) => c.forEach((p) => allTimes.add(p.time)));
  const times = [...allTimes].sort((a, b) => a - b);
  const maxTime = Math.max(...times, 1);

  const buildStepPath = (curve: { time: number; surv: number }[]) => {
    if (curve.length === 0) return "";
    const sorted = [...curve].sort((a, b) => a.time - b.time);

    let path = "";
    sorted.forEach((p, i) => {
      const x = margin.left + (p.time / maxTime) * plotWidth;
      const y = margin.top + (1 - p.surv) * plotHeight;

      if (i === 0) {
        path += `M ${margin.left} ${margin.top} L ${x} ${margin.top} L ${x} ${y}`;
      } else {
        const prevY = margin.top + (1 - sorted[i - 1].surv) * plotHeight;
        path += ` L ${x} ${prevY} L ${x} ${y}`;
      }
    });
    return path;
  };

  const paths = curves
    .map((curve, idx) => {
      const d = buildStepPath(curve);
      if (!d) return "";
      return `<path d="${d}" fill="none" stroke="${RUN_COLORS_HEX[idx]}" stroke-width="2" />`;
    })
    .join("\n");

  // Axes
  const xAxis = `<line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${margin.left + plotWidth}" y2="${margin.top + plotHeight}" stroke="#475569" stroke-width="1" />`;
  const yAxis = `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" stroke="#475569" stroke-width="1" />`;

  const xTicks = [0, 0.25, 0.5, 0.75, 1].map((v) => {
    const x = margin.left + v * plotWidth;
    const label = (v * maxTime).toFixed(0);
    return `<text x="${x}" y="${margin.top + plotHeight + 18}" text-anchor="middle" font-size="10" fill="#64748b">${label}</text>`;
  }).join("");

  const yTicks = [0, 25, 50, 75, 100].map((v) => {
    const y = margin.top + (1 - v / 100) * plotHeight;
    return `<text x="${margin.left - 8}" y="${y + 3}" text-anchor="end" font-size="10" fill="#64748b">${v}%</text>`;
  }).join("");

  const xLabel = `<text x="${margin.left + plotWidth / 2}" y="${height - 8}" text-anchor="middle" font-size="11" fill="#475569">Time</text>`;
  const yLabel = `<text x="14" y="${margin.top + plotHeight / 2}" text-anchor="middle" font-size="11" fill="#475569" transform="rotate(-90 14 ${margin.top + plotHeight / 2})">Survival (%)</text>`;

  const legend = runs
    .map((_, idx) => {
      const y = margin.top + 10 + idx * 18;
      return `
        <rect x="${margin.left + plotWidth + 10}" y="${y - 5}" width="12" height="12" rx="2" fill="${RUN_COLORS_HEX[idx]}" />
        <text x="${margin.left + plotWidth + 26}" y="${y + 5}" font-size="10" fill="#334155">${RUN_LABELS[idx]}</text>
      `;
    })
    .join("");

  const titleColor = riskLevel === "high" ? "#dc2626" : "#10b981";
  const title = `<text x="${width / 2}" y="22" text-anchor="middle" font-size="13" font-weight="600" fill="${titleColor}">${riskLevel === "high" ? "High" : "Low"} Risk Group - Kaplan-Meier</text>`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <rect width="${width}" height="${height}" fill="white" />
    ${title}
    ${xAxis}
    ${yAxis}
    ${paths}
    ${xTicks}
    ${yTicks}
    ${xLabel}
    ${yLabel}
    ${legend}
  </svg>`;
}

// Build UpSet-style Feature Intersection SVG
export function buildUpsetMatrixSVG(runs: { name: string; data: MLResults }[]): string {
  const featureSets = runs.map((r) => new Set(r.data.selected_features || []));
  const n = featureSets.length;
  const allFeatures = new Set<string>();
  featureSets.forEach((s) => s.forEach((f) => allFeatures.add(f)));

  // Compute intersections
  const intersections: { runIndices: number[]; count: number; label: string }[] = [];

  for (let mask = 1; mask < (1 << n); mask++) {
    const runIndices: number[] = [];
    for (let i = 0; i < n; i++) {
      if (mask & (1 << i)) runIndices.push(i);
    }

    const features = [...allFeatures].filter((f) => {
      return featureSets.every((set, idx) => {
        const inSet = set.has(f);
        const shouldBe = runIndices.includes(idx);
        return inSet === shouldBe;
      });
    });

    if (features.length > 0) {
      intersections.push({
        runIndices,
        count: features.length,
        label: runIndices.map((i) => RUN_LABELS[i]).join(" ∩ "),
      });
    }
  }

  intersections.sort((a, b) => b.count - a.count);
  const topIntersections = intersections.slice(0, 10);

  const width = 500;
  const barHeight = 150;
  const matrixHeight = n * 24 + 20;
  const height = barHeight + matrixHeight + 60;
  const margin = { left: 80, right: 20 };
  const barWidth = (width - margin.left - margin.right) / topIntersections.length;
  const maxCount = Math.max(...topIntersections.map((i) => i.count), 1);

  // Bar chart
  const bars = topIntersections
    .map((int, idx) => {
      const x = margin.left + idx * barWidth + barWidth * 0.15;
      const h = (int.count / maxCount) * (barHeight - 30);
      const y = barHeight - h;
      const w = barWidth * 0.7;
      const color = int.runIndices.length === n ? "#10b981" : int.runIndices.length === 1 ? RUN_COLORS_HEX[int.runIndices[0]] : "#0ea5e9";
      return `
        <rect x="${x}" y="${y}" width="${w}" height="${h}" rx="3" fill="${color}" />
        <text x="${x + w / 2}" y="${y - 5}" text-anchor="middle" font-size="9" fill="#475569">${int.count}</text>
      `;
    })
    .join("");

  // Dot matrix
  const dotMatrix = topIntersections
    .map((int, colIdx) => {
      const cx = margin.left + colIdx * barWidth + barWidth / 2;
      return runs
        .map((_, rowIdx) => {
          const cy = barHeight + 30 + rowIdx * 24;
          const isIn = int.runIndices.includes(rowIdx);
          return isIn
            ? `<circle cx="${cx}" cy="${cy}" r="6" fill="#1e293b" /><circle cx="${cx}" cy="${cy}" r="3" fill="white" />`
            : `<circle cx="${cx}" cy="${cy}" r="6" fill="#e2e8f0" />`;
        })
        .join("");
    })
    .join("");

  // Run labels
  const runLabelsEl = runs
    .map((_, idx) => {
      const y = barHeight + 30 + idx * 24;
      return `
        <rect x="10" y="${y - 6}" width="12" height="12" rx="2" fill="${RUN_COLORS_HEX[idx]}" />
        <text x="28" y="${y + 4}" font-size="10" fill="#334155">${RUN_LABELS[idx]}</text>
      `;
    })
    .join("");

  const title = `<text x="${width / 2}" y="20" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b">Feature Overlap (UpSet Matrix)</text>`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <rect width="${width}" height="${height}" fill="white" />
    ${title}
    ${bars}
    ${dotMatrix}
    ${runLabelsEl}
  </svg>`;
}
