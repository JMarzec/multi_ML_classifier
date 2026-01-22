import { useMemo, useState } from "react";
import {
  Line,
  LineChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { MLResults, ModelPerformance } from "@/types/ml-results";

interface ROCRunsOverlayChartProps {
  runs: { name: string; data: MLResults }[];
  runColors: { fill: string; text: string; bg: string; border: string }[];
  runLabels: string[];
}

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

function getModelsWithROC(runs: { data: MLResults }[]): ModelKey[] {
  const present = new Set<ModelKey>();
  runs.forEach((r) => {
    (Object.keys(r.data.model_performance) as ModelKey[]).forEach((model) => {
      const roc = r.data.model_performance[model]?.roc_curve;
      if (roc && roc.length > 0) present.add(model);
    });
  });
  return (Object.keys(MODEL_LABELS) as ModelKey[]).filter((m) => present.has(m));
}

export function ROCRunsOverlayChart({ runs, runColors, runLabels }: ROCRunsOverlayChartProps) {
  const models = useMemo(() => getModelsWithROC(runs), [runs]);
  const [selectedModel, setSelectedModel] = useState<ModelKey>(models.includes("soft_vote") ? "soft_vote" : (models[0] ?? "rf"));

  const chartData = useMemo(() => {
    const fprPoints = Array.from({ length: 101 }, (_, i) => i / 100);

    return fprPoints.map((fpr) => {
      const point: Record<string, number> = { fpr };

      runs.forEach((run, idx) => {
        const roc = run.data.model_performance[selectedModel]?.roc_curve;
        if (!roc || roc.length === 0) return;

        const sorted = [...roc].sort((a, b) => a.fpr - b.fpr);
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

        point[`run${idx}`] = tpr;
      });

      return point;
    });
  }, [runs, selectedModel]);

  const anyROCForModel = useMemo(() => {
    return runs.some((r) => (r.data.model_performance[selectedModel]?.roc_curve?.length ?? 0) > 0);
  }, [runs, selectedModel]);

  if (models.length === 0) {
    return (
      <div className="bg-card rounded-xl p-8 border border-border text-center">
        <p className="text-sm text-muted-foreground">
          No ROC curve data was found in the uploaded runs.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-card rounded-xl p-6 border border-border space-y-4">
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <h3 className="text-lg font-semibold">ROC Overlay Across Runs</h3>
        <div className="flex items-center gap-2">
          <span className="text-sm text-muted-foreground">Model</span>
          <Select value={selectedModel} onValueChange={(v) => setSelectedModel(v as ModelKey)}>
            <SelectTrigger className="w-[220px]">
              <SelectValue placeholder="Select model" />
            </SelectTrigger>
            <SelectContent>
              {models.map((m) => (
                <SelectItem key={m} value={m}>
                  {MODEL_LABELS[m] || m}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      {!anyROCForModel ? (
        <p className="text-sm text-muted-foreground">
          This model has no ROC data in the uploaded runs.
        </p>
      ) : (
        <div className="h-[420px]">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 50 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis
                dataKey="fpr"
                type="number"
                domain={[0, 1]}
                tickFormatter={(v) => (v * 100).toFixed(0) + "%"}
                stroke="hsl(var(--muted-foreground))"
                label={{
                  value: "False Positive Rate (1 - Specificity)",
                  position: "insideBottom",
                  offset: -10,
                  style: { textAnchor: "middle", fill: "hsl(var(--muted-foreground))" },
                }}
              />
              <YAxis
                type="number"
                domain={[0, 1]}
                tickFormatter={(v) => (v * 100).toFixed(0) + "%"}
                stroke="hsl(var(--muted-foreground))"
                label={{ value: "True Positive Rate (Sensitivity)", angle: -90, position: "insideLeft" }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                }}
                labelFormatter={(value) => `FPR: ${(Number(value) * 100).toFixed(1)}%`}
                formatter={(value: number, name: string) => {
                  const idx = parseInt(name.replace("run", ""));
                  return [`${(value * 100).toFixed(1)}%`, runLabels[idx] ?? name];
                }}
              />
              <Legend verticalAlign="top" align="right" wrapperStyle={{ paddingBottom: "16px" }} />
              <ReferenceLine
                segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                stroke="hsl(var(--muted-foreground))"
                strokeDasharray="5 5"
              />

              {runs.map((_, idx) => (
                <Line
                  key={idx}
                  type="monotone"
                  dataKey={`run${idx}`}
                  name={runLabels[idx]}
                  stroke={runColors[idx].fill}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                  connectNulls
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      <p className="text-xs text-muted-foreground">
        Overlays ROC curves for the selected model across runs to compare discrimination directly.
      </p>
    </div>
  );
}
