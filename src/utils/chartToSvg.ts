import type { MLResults, ModelPerformance, ModelRiskScoreSurvival } from "@/types/ml-results";

const RUN_COLORS_HEX = ["#0ea5e9", "#8b5cf6", "#10b981", "#f59e0b"];
const RUN_LABELS = ["Run A", "Run B", "Run C", "Run D"];
const MODEL_COLORS_HEX: Record<string, string> = {
  rf: "#10b981",
  svm: "#8b5cf6",
  xgboost: "#f59e0b",
  knn: "#ec4899",
  mlp: "#06b6d4",
  hard_vote: "#6366f1",
  soft_vote: "#0ea5e9",
};
const MODEL_LABELS: Record<string, string> = {
  rf: "Random Forest",
  svm: "SVM",
  xgboost: "XGBoost",
  knn: "KNN",
  mlp: "MLP",
  hard_vote: "Hard Voting",
  soft_vote: "Soft Voting",
};

type ModelKey = keyof ModelPerformance;

export function toFiniteNumber(value: unknown): number | undefined {
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

// ============================================================================
// CONFUSION MATRIX HEATMAP SVG
// ============================================================================

export function buildConfusionMatrixSVG(
  tp: number,
  tn: number,
  fp: number,
  fn: number,
  modelName: string = "Model"
): string {
  const width = 320;
  const height = 320;
  const margin = { top: 50, right: 30, bottom: 60, left: 80 };
  const cellSize = 100;
  const matrixStartX = margin.left;
  const matrixStartY = margin.top;

  // Calculate derived metrics
  const total = tp + tn + fp + fn;
  const accuracy = total > 0 ? ((tp + tn) / total * 100).toFixed(1) : "N/A";
  const sensitivity = (tp + fn) > 0 ? (tp / (tp + fn) * 100).toFixed(1) : "N/A";
  const specificity = (tn + fp) > 0 ? (tn / (tn + fp) * 100).toFixed(1) : "N/A";
  const precision = (tp + fp) > 0 ? (tp / (tp + fp) * 100).toFixed(1) : "N/A";

  // Color scale (green for correct, red for errors)
  const maxVal = Math.max(tp, tn, fp, fn, 1);
  const getColor = (val: number, isCorrect: boolean) => {
    const intensity = Math.min(val / maxVal, 1);
    if (isCorrect) {
      // Green gradient for TP/TN
      const r = Math.round(220 - intensity * 180);
      const g = Math.round(252 - intensity * 32);
      const b = Math.round(220 - intensity * 140);
      return `rgb(${r}, ${g}, ${b})`;
    } else {
      // Red gradient for FP/FN
      const r = Math.round(254 - intensity * 20);
      const g = Math.round(226 - intensity * 140);
      const b = Math.round(226 - intensity * 140);
      return `rgb(${r}, ${g}, ${b})`;
    }
  };

  // Matrix cells: [row][col] where row=Predicted, col=Actual
  // Layout:        Actual Neg | Actual Pos
  // Predicted Neg:    TN      |    FN
  // Predicted Pos:    FP      |    TP
  const cells = [
    { x: 0, y: 0, val: tn, label: "TN", isCorrect: true },
    { x: 1, y: 0, val: fn, label: "FN", isCorrect: false },
    { x: 0, y: 1, val: fp, label: "FP", isCorrect: false },
    { x: 1, y: 1, val: tp, label: "TP", isCorrect: true },
  ];

  const cellsEl = cells.map((cell) => {
    const cx = matrixStartX + cell.x * cellSize + cellSize / 2;
    const cy = matrixStartY + cell.y * cellSize + cellSize / 2;
    const color = getColor(cell.val, cell.isCorrect);
    const textColor = cell.val / maxVal > 0.5 ? "#1e293b" : "#475569";
    
    return `
      <rect x="${matrixStartX + cell.x * cellSize}" y="${matrixStartY + cell.y * cellSize}" 
            width="${cellSize}" height="${cellSize}" 
            fill="${color}" stroke="#e2e8f0" stroke-width="1" />
      <text x="${cx}" y="${cy - 8}" text-anchor="middle" font-size="11" fill="#64748b">${cell.label}</text>
      <text x="${cx}" y="${cy + 12}" text-anchor="middle" font-size="16" font-weight="600" fill="${textColor}">${cell.val}</text>
    `;
  }).join("");

  // Axis labels
  const axisLabels = `
    <text x="${matrixStartX + cellSize}" y="${margin.top - 25}" text-anchor="middle" font-size="11" font-weight="600" fill="#475569">Actual Class</text>
    <text x="${matrixStartX + cellSize / 2}" y="${margin.top - 8}" text-anchor="middle" font-size="10" fill="#64748b">Negative</text>
    <text x="${matrixStartX + cellSize * 1.5}" y="${margin.top - 8}" text-anchor="middle" font-size="10" fill="#64748b">Positive</text>
    <text x="${margin.left - 35}" y="${matrixStartY + cellSize}" text-anchor="middle" font-size="11" font-weight="600" fill="#475569" transform="rotate(-90 ${margin.left - 35} ${matrixStartY + cellSize})">Predicted</text>
    <text x="${margin.left - 8}" y="${matrixStartY + cellSize / 2 + 4}" text-anchor="end" font-size="10" fill="#64748b">Neg</text>
    <text x="${margin.left - 8}" y="${matrixStartY + cellSize * 1.5 + 4}" text-anchor="end" font-size="10" fill="#64748b">Pos</text>
  `;

  // Metrics summary below matrix
  const metricsY = matrixStartY + cellSize * 2 + 20;
  const metricsEl = `
    <text x="${matrixStartX}" y="${metricsY}" font-size="9" fill="#475569">
      <tspan font-weight="600">Acc:</tspan> ${accuracy}%  
      <tspan font-weight="600" dx="8">Sens:</tspan> ${sensitivity}%  
      <tspan font-weight="600" dx="8">Spec:</tspan> ${specificity}%  
      <tspan font-weight="600" dx="8">Prec:</tspan> ${precision}%
    </text>
  `;

  const title = `<text x="${width / 2}" y="20" text-anchor="middle" font-size="12" font-weight="600" fill="#1e293b">${modelName} - Confusion Matrix</text>`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <rect width="${width}" height="${height}" fill="white" />
    ${title}
    ${cellsEl}
    ${axisLabels}
    ${metricsEl}
  </svg>`;
}

// Build multi-model confusion matrix grid SVG
export function buildConfusionMatrixGridSVG(data: MLResults, modelsToShow?: string[]): string {
  const MODEL_LABELS: Record<string, string> = {
    rf: "Random Forest",
    svm: "SVM",
    xgboost: "XGBoost",
    knn: "KNN",
    mlp: "MLP",
    hard_vote: "Hard Voting",
    soft_vote: "Soft Voting",
  };

  const models = modelsToShow || Object.keys(data.model_performance).filter(
    (m) => data.model_performance[m as keyof typeof data.model_performance]?.confusion_matrix
  );

  const validModels = models.filter((m) => {
    const cm = data.model_performance[m as keyof typeof data.model_performance]?.confusion_matrix;
    return cm && (cm.tp !== undefined || cm.tn !== undefined);
  });

  if (validModels.length === 0) {
    return `<svg xmlns="http://www.w3.org/2000/svg" width="400" height="100" viewBox="0 0 400 100">
      <rect width="400" height="100" fill="white" />
      <text x="200" y="55" text-anchor="middle" font-size="12" fill="#64748b">No confusion matrix data available</text>
    </svg>`;
  }

  const cellWidth = 180;
  const cellHeight = 200;
  const cols = Math.min(validModels.length, 3);
  const rows = Math.ceil(validModels.length / cols);
  const width = cols * cellWidth + 40;
  const height = rows * cellHeight + 60;

  const miniMatrices = validModels.map((model, idx) => {
    const cm = data.model_performance[model as keyof typeof data.model_performance]!.confusion_matrix!;
    const col = idx % cols;
    const row = Math.floor(idx / cols);
    const offsetX = 20 + col * cellWidth;
    const offsetY = 40 + row * cellHeight;
    
    return buildMiniConfusionMatrix(
      cm.tp, cm.tn, cm.fp, cm.fn,
      MODEL_LABELS[model] || model,
      offsetX, offsetY
    );
  }).join("");

  const title = `<text x="${width / 2}" y="22" text-anchor="middle" font-size="14" font-weight="600" fill="#1e293b">Confusion Matrix Comparison</text>`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <rect width="${width}" height="${height}" fill="white" />
    ${title}
    ${miniMatrices}
  </svg>`;
}

function buildMiniConfusionMatrix(tp: number, tn: number, fp: number, fn: number, label: string, offsetX: number, offsetY: number): string {
  const cellSize = 55;
  const total = tp + tn + fp + fn;
  const maxVal = Math.max(tp, tn, fp, fn, 1);

  const getColor = (val: number, isCorrect: boolean) => {
    const intensity = Math.min(val / maxVal, 1);
    if (isCorrect) {
      return `rgb(${Math.round(220 - intensity * 180)}, ${Math.round(252 - intensity * 32)}, ${Math.round(220 - intensity * 140)})`;
    } else {
      return `rgb(${Math.round(254 - intensity * 20)}, ${Math.round(226 - intensity * 140)}, ${Math.round(226 - intensity * 140)})`;
    }
  };

  const cells = [
    { x: 0, y: 0, val: tn, abbr: "TN", isCorrect: true },
    { x: 1, y: 0, val: fn, abbr: "FN", isCorrect: false },
    { x: 0, y: 1, val: fp, abbr: "FP", isCorrect: false },
    { x: 1, y: 1, val: tp, abbr: "TP", isCorrect: true },
  ];

  const cellsEl = cells.map((c) => {
    const cx = offsetX + 30 + c.x * cellSize + cellSize / 2;
    const cy = offsetY + 25 + c.y * cellSize + cellSize / 2;
    return `
      <rect x="${offsetX + 30 + c.x * cellSize}" y="${offsetY + 25 + c.y * cellSize}" 
            width="${cellSize}" height="${cellSize}" 
            fill="${getColor(c.val, c.isCorrect)}" stroke="#e2e8f0" stroke-width="1" />
      <text x="${cx}" y="${cy - 5}" text-anchor="middle" font-size="8" fill="#64748b">${c.abbr}</text>
      <text x="${cx}" y="${cy + 10}" text-anchor="middle" font-size="12" font-weight="600" fill="#334155">${c.val}</text>
    `;
  }).join("");

  const accuracy = total > 0 ? ((tp + tn) / total * 100).toFixed(0) : "N/A";
  
  return `
    <text x="${offsetX + 30 + cellSize}" y="${offsetY + 15}" text-anchor="middle" font-size="10" font-weight="600" fill="#334155">${label}</text>
    ${cellsEl}
    <text x="${offsetX + 30 + cellSize}" y="${offsetY + 25 + cellSize * 2 + 15}" text-anchor="middle" font-size="9" fill="#475569">Accuracy: ${accuracy}%</text>
  `;
}

// ============================================================================
// SINGLE-RUN SVG BUILDERS
// ============================================================================

// Build single-run ROC Overlay SVG (all models)
export function buildSingleRunROCSVG(data: MLResults): string {
  const width = 520;
  const height = 400;
  const margin = { top: 30, right: 140, bottom: 50, left: 60 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const models = Object.keys(data.model_performance).filter(
    (m) => data.model_performance[m as ModelKey]?.roc_curve?.length
  ) as ModelKey[];

  const fprPoints = Array.from({ length: 101 }, (_, i) => i / 100);

  const curvesData = models.map((model) => {
    const roc = data.model_performance[model]?.roc_curve;
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

      const tpr =
        upper.fpr === lower.fpr
          ? lower.tpr
          : lower.tpr + ((fpr - lower.fpr) / (upper.fpr - lower.fpr)) * (upper.tpr - lower.tpr);

      return { fpr, tpr };
    });
  });

  const aurocs = models.map((model) => {
    const val = data.model_performance[model]?.auroc?.mean;
    return val ? (val * 100).toFixed(1) : "N/A";
  });

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
      const model = models[idx];
      const color = MODEL_COLORS_HEX[model] || "#64748b";
      return `<path d="${buildPath(curve)}" fill="none" stroke="${color}" stroke-width="2" />`;
    })
    .join("\n");

  const diagLine = `<line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${margin.left + plotWidth}" y2="${margin.top}" stroke="#94a3b8" stroke-dasharray="5 5" stroke-width="1" />`;
  const xAxis = `<line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${margin.left + plotWidth}" y2="${margin.top + plotHeight}" stroke="#475569" stroke-width="1" />`;
  const yAxis = `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" stroke="#475569" stroke-width="1" />`;

  const xTicks = [0, 25, 50, 75, 100]
    .map((v) => {
      const x = margin.left + (v / 100) * plotWidth;
      return `<text x="${x}" y="${margin.top + plotHeight + 18}" text-anchor="middle" font-size="10" fill="#64748b">${v}%</text>`;
    })
    .join("");

  const yTicks = [0, 25, 50, 75, 100]
    .map((v) => {
      const y = margin.top + (1 - v / 100) * plotHeight;
      return `<text x="${margin.left - 8}" y="${y + 3}" text-anchor="end" font-size="10" fill="#64748b">${v}%</text>`;
    })
    .join("");

  const xLabel = `<text x="${margin.left + plotWidth / 2}" y="${height - 8}" text-anchor="middle" font-size="11" fill="#475569">False Positive Rate</text>`;
  const yLabel = `<text x="14" y="${margin.top + plotHeight / 2}" text-anchor="middle" font-size="11" fill="#475569" transform="rotate(-90 14 ${margin.top + plotHeight / 2})">True Positive Rate</text>`;

  const legend = models
    .map((model, idx) => {
      const y = margin.top + 10 + idx * 20;
      const color = MODEL_COLORS_HEX[model] || "#64748b";
      const label = MODEL_LABELS[model] || model;
      return `
        <rect x="${margin.left + plotWidth + 10}" y="${y - 5}" width="12" height="12" rx="2" fill="${color}" />
        <text x="${margin.left + plotWidth + 26}" y="${y + 5}" font-size="9" fill="#334155">${label} (${aurocs[idx]}%)</text>
      `;
    })
    .join("");

  const title = `<text x="${width / 2}" y="18" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b">ROC Curves - All Models</text>`;

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

// Build single-run Kaplan-Meier SVG (high vs low risk)
export function buildSingleRunKMSVG(data: MLResults): string {
  const width = 500;
  const height = 350;
  const margin = { top: 40, right: 120, bottom: 50, left: 60 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const survival = data.survival_analysis;
  const modelScores = normalizeModelRiskScores(survival?.model_risk_scores);
  const softVote = modelScores.find((m) => m.model === "soft_vote" || m.model === "ensemble");

  const highCurve = softVote?.km_curve_high || [];
  const lowCurve = softVote?.km_curve_low || [];

  const allTimes = new Set<number>();
  highCurve.forEach((p) => allTimes.add(p.time));
  lowCurve.forEach((p) => allTimes.add(p.time));
  const maxTime = Math.max(...allTimes, 1);

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

  const highPath = buildStepPath(highCurve);
  const lowPath = buildStepPath(lowCurve);

  const paths = `
    ${lowPath ? `<path d="${lowPath}" fill="none" stroke="#10b981" stroke-width="2.5" />` : ""}
    ${highPath ? `<path d="${highPath}" fill="none" stroke="#ef4444" stroke-width="2.5" />` : ""}
  `;

  const xAxis = `<line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${margin.left + plotWidth}" y2="${margin.top + plotHeight}" stroke="#475569" stroke-width="1" />`;
  const yAxis = `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" stroke="#475569" stroke-width="1" />`;

  const xTicks = [0, 0.25, 0.5, 0.75, 1]
    .map((v) => {
      const x = margin.left + v * plotWidth;
      const label = (v * maxTime).toFixed(0);
      return `<text x="${x}" y="${margin.top + plotHeight + 18}" text-anchor="middle" font-size="10" fill="#64748b">${label}</text>`;
    })
    .join("");

  const yTicks = [0, 25, 50, 75, 100]
    .map((v) => {
      const y = margin.top + (1 - v / 100) * plotHeight;
      return `<text x="${margin.left - 8}" y="${y + 3}" text-anchor="end" font-size="10" fill="#64748b">${v}%</text>`;
    })
    .join("");

  const xLabel = `<text x="${margin.left + plotWidth / 2}" y="${height - 8}" text-anchor="middle" font-size="11" fill="#475569">Time</text>`;
  const yLabel = `<text x="14" y="${margin.top + plotHeight / 2}" text-anchor="middle" font-size="11" fill="#475569" transform="rotate(-90 14 ${margin.top + plotHeight / 2})">Survival Probability</text>`;

  // Stats
  const stats = softVote?.stats;
  const pValue = stats?.logrank_p !== undefined ? toFiniteNumber(stats.logrank_p) : undefined;
  const hr = stats?.cox_hr !== undefined ? toFiniteNumber(stats.cox_hr) : undefined;

  const statsText = `
    <text x="${margin.left + plotWidth + 15}" y="${margin.top + 80}" font-size="9" fill="#64748b">Log-rank p:</text>
    <text x="${margin.left + plotWidth + 15}" y="${margin.top + 94}" font-size="10" font-weight="600" fill="${pValue !== undefined && pValue < 0.05 ? '#10b981' : '#64748b'}">${pValue !== undefined ? pValue.toFixed(4) : "N/A"}</text>
    <text x="${margin.left + plotWidth + 15}" y="${margin.top + 115}" font-size="9" fill="#64748b">Hazard Ratio:</text>
    <text x="${margin.left + plotWidth + 15}" y="${margin.top + 129}" font-size="10" font-weight="600" fill="#334155">${hr !== undefined ? hr.toFixed(2) : "N/A"}</text>
  `;

  const legend = `
    <rect x="${margin.left + plotWidth + 15}" y="${margin.top + 10}" width="12" height="12" rx="2" fill="#ef4444" />
    <text x="${margin.left + plotWidth + 31}" y="${margin.top + 20}" font-size="10" fill="#334155">High Risk</text>
    <rect x="${margin.left + plotWidth + 15}" y="${margin.top + 32}" width="12" height="12" rx="2" fill="#10b981" />
    <text x="${margin.left + plotWidth + 31}" y="${margin.top + 42}" font-size="10" fill="#334155">Low Risk</text>
  `;

  const title = `<text x="${width / 2}" y="22" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b">Kaplan-Meier Survival Curves</text>`;

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
    ${statsText}
  </svg>`;
}

// Build clinical-report Kaplan-Meier SVG with patient event marker
export function buildClinicalKMSVG(
  data: MLResults,
  patientTime?: number,
  patientEvent?: boolean,
  patientRiskGroup?: "high" | "low"
): string {
  const width = 500;
  const height = 350;
  const margin = { top: 40, right: 120, bottom: 50, left: 60 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  const survival = data.survival_analysis;
  const modelScores = normalizeModelRiskScores(survival?.model_risk_scores);
  const softVote = modelScores.find((m) => m.model === "soft_vote" || m.model === "ensemble");

  const highCurve = softVote?.km_curve_high || [];
  const lowCurve = softVote?.km_curve_low || [];

  const allTimes = new Set<number>();
  highCurve.forEach((p) => allTimes.add(p.time));
  lowCurve.forEach((p) => allTimes.add(p.time));
  const maxTime = Math.max(...allTimes, 1);

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

  // Get survival probability at patient time
  const getSurvAtTime = (curve: { time: number; surv: number }[], time: number): number => {
    if (curve.length === 0) return 1;
    const sorted = [...curve].sort((a, b) => a.time - b.time);
    for (let i = sorted.length - 1; i >= 0; i--) {
      if (sorted[i].time <= time) return sorted[i].surv;
    }
    return 1;
  };

  const highPath = buildStepPath(highCurve);
  const lowPath = buildStepPath(lowCurve);

  const paths = `
    ${lowPath ? `<path d="${lowPath}" fill="none" stroke="#10b981" stroke-width="2.5" />` : ""}
    ${highPath ? `<path d="${highPath}" fill="none" stroke="#ef4444" stroke-width="2.5" />` : ""}
  `;

  // Patient marker
  let patientMarker = "";
  if (patientTime !== undefined && patientRiskGroup) {
    const curve = patientRiskGroup === "high" ? highCurve : lowCurve;
    const surv = getSurvAtTime(curve, patientTime);
    const px = margin.left + (patientTime / maxTime) * plotWidth;
    const py = margin.top + (1 - surv) * plotHeight;
    const markerColor = patientEvent ? "#dc2626" : "#0ea5e9";
    const markerSymbol = patientEvent
      ? `<polygon points="${px},${py - 10} ${px - 8},${py + 6} ${px + 8},${py + 6}" fill="${markerColor}" stroke="white" stroke-width="1.5" />`
      : `<circle cx="${px}" cy="${py}" r="7" fill="${markerColor}" stroke="white" stroke-width="2" />`;

    patientMarker = `
      ${markerSymbol}
      <text x="${px}" y="${py - 15}" text-anchor="middle" font-size="9" font-weight="600" fill="${markerColor}">Patient</text>
    `;
  }

  const xAxis = `<line x1="${margin.left}" y1="${margin.top + plotHeight}" x2="${margin.left + plotWidth}" y2="${margin.top + plotHeight}" stroke="#475569" stroke-width="1" />`;
  const yAxis = `<line x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${margin.top + plotHeight}" stroke="#475569" stroke-width="1" />`;

  const xTicks = [0, 0.25, 0.5, 0.75, 1]
    .map((v) => {
      const x = margin.left + v * plotWidth;
      const label = (v * maxTime).toFixed(0);
      return `<text x="${x}" y="${margin.top + plotHeight + 18}" text-anchor="middle" font-size="10" fill="#64748b">${label}</text>`;
    })
    .join("");

  const yTicks = [0, 25, 50, 75, 100]
    .map((v) => {
      const y = margin.top + (1 - v / 100) * plotHeight;
      return `<text x="${margin.left - 8}" y="${y + 3}" text-anchor="end" font-size="10" fill="#64748b">${v}%</text>`;
    })
    .join("");

  const xLabel = `<text x="${margin.left + plotWidth / 2}" y="${height - 8}" text-anchor="middle" font-size="11" fill="#475569">Time</text>`;
  const yLabel = `<text x="14" y="${margin.top + plotHeight / 2}" text-anchor="middle" font-size="11" fill="#475569" transform="rotate(-90 14 ${margin.top + plotHeight / 2})">Survival Probability</text>`;

  const stats = softVote?.stats;
  const pValue = stats?.logrank_p !== undefined ? toFiniteNumber(stats.logrank_p) : undefined;
  const hr = stats?.cox_hr !== undefined ? toFiniteNumber(stats.cox_hr) : undefined;

  const statsText = `
    <text x="${margin.left + plotWidth + 15}" y="${margin.top + 80}" font-size="9" fill="#64748b">Log-rank p:</text>
    <text x="${margin.left + plotWidth + 15}" y="${margin.top + 94}" font-size="10" font-weight="600" fill="${pValue !== undefined && pValue < 0.05 ? '#10b981' : '#64748b'}">${pValue !== undefined ? pValue.toFixed(4) : "N/A"}</text>
    <text x="${margin.left + plotWidth + 15}" y="${margin.top + 115}" font-size="9" fill="#64748b">Hazard Ratio:</text>
    <text x="${margin.left + plotWidth + 15}" y="${margin.top + 129}" font-size="10" font-weight="600" fill="#334155">${hr !== undefined ? hr.toFixed(2) : "N/A"}</text>
  `;

  const legend = `
    <rect x="${margin.left + plotWidth + 15}" y="${margin.top + 10}" width="12" height="12" rx="2" fill="#ef4444" />
    <text x="${margin.left + plotWidth + 31}" y="${margin.top + 20}" font-size="10" fill="#334155">High Risk</text>
    <rect x="${margin.left + plotWidth + 15}" y="${margin.top + 32}" width="12" height="12" rx="2" fill="#10b981" />
    <text x="${margin.left + plotWidth + 31}" y="${margin.top + 42}" font-size="10" fill="#334155">Low Risk</text>
  `;

  const title = `<text x="${width / 2}" y="22" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b">Kaplan-Meier Survival Curves</text>`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <rect width="${width}" height="${height}" fill="white" />
    ${title}
    ${xAxis}
    ${yAxis}
    ${paths}
    ${patientMarker}
    ${xTicks}
    ${yTicks}
    ${xLabel}
    ${yLabel}
    ${legend}
    ${statsText}
  </svg>`;
}

// ============================================================================
// FEATURE IMPORTANCE BAR CHART SVG
// ============================================================================

import type { FeatureImportance } from "@/types/ml-results";

export function buildFeatureImportanceSVG(
  features: FeatureImportance[],
  maxFeatures: number = 15,
  title?: string
): string {
  const width = 520;
  const barHeight = 22;
  const margin = { top: 40, right: 80, bottom: 30, left: 140 };
  
  const sortedFeatures = [...features]
    .sort((a, b) => b.importance - a.importance)
    .slice(0, maxFeatures);
  
  const maxImportance = Math.max(...sortedFeatures.map((f) => f.importance), 0.001);
  const plotWidth = width - margin.left - margin.right;
  const height = margin.top + sortedFeatures.length * barHeight + margin.bottom;

  const getBarColor = (index: number, total: number) => {
    const hue = 199 - (index / Math.max(total - 1, 1)) * 50;
    return `hsl(${hue}, 80%, 50%)`;
  };

  const bars = sortedFeatures
    .map((feat, idx) => {
      const normalized = (feat.importance / maxImportance) * 100;
      const y = margin.top + idx * barHeight;
      const barW = (normalized / 100) * plotWidth;
      const color = getBarColor(idx, sortedFeatures.length);

      return `
        <text x="${margin.left - 8}" y="${y + barHeight / 2 + 4}" text-anchor="end" font-size="10" fill="#334155" style="font-family: monospace;">${feat.feature.length > 18 ? feat.feature.slice(0, 16) + "…" : feat.feature}</text>
        <rect x="${margin.left}" y="${y + 3}" width="${barW}" height="${barHeight - 6}" rx="3" fill="${color}" />
        <text x="${margin.left + barW + 6}" y="${y + barHeight / 2 + 4}" font-size="9" fill="#64748b">${normalized.toFixed(1)}%</text>
      `;
    })
    .join("");

  const xAxis = `<line x1="${margin.left}" y1="${margin.top + sortedFeatures.length * barHeight}" x2="${margin.left + plotWidth}" y2="${margin.top + sortedFeatures.length * barHeight}" stroke="#e5e7eb" stroke-width="1" />`;
  
  const xTicks = [0, 25, 50, 75, 100]
    .map((v) => {
      const x = margin.left + (v / 100) * plotWidth;
      return `
        <line x1="${x}" y1="${margin.top - 5}" x2="${x}" y2="${margin.top + sortedFeatures.length * barHeight}" stroke="#f1f5f9" stroke-width="1" />
        <text x="${x}" y="${margin.top + sortedFeatures.length * barHeight + 15}" text-anchor="middle" font-size="9" fill="#64748b">${v}%</text>
      `;
    })
    .join("");

  const titleText = title || `Top ${sortedFeatures.length} Features by Importance`;
  const titleEl = `<text x="${width / 2}" y="22" text-anchor="middle" font-size="13" font-weight="600" fill="#1e293b">${titleText}</text>`;

  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
    <rect width="${width}" height="${height}" fill="white" />
    ${titleEl}
    ${xTicks}
    ${xAxis}
    ${bars}
  </svg>`;
}
