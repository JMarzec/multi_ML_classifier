import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import type { ModelPerformance } from "@/types/ml-results";

interface ModelComparisonChartProps {
  performance: ModelPerformance;
  metric: "accuracy" | "auroc" | "f1_score" | "balanced_accuracy";
}

const MODEL_COLORS: Record<string, string> = {
  rf: "hsl(142, 71%, 45%)",
  svm: "hsl(262, 83%, 58%)",
  xgboost: "hsl(38, 92%, 50%)",
  knn: "hsl(199, 89%, 48%)",
  mlp: "hsl(340, 82%, 52%)",
  hard_vote: "hsl(172, 66%, 50%)",
  soft_vote: "hsl(280, 70%, 60%)",
};

const MODEL_LABELS: Record<string, string> = {
  rf: "Random Forest",
  svm: "SVM",
  xgboost: "XGBoost",
  knn: "KNN",
  mlp: "MLP",
  hard_vote: "Hard Vote",
  soft_vote: "Soft Vote",
};

export function ModelComparisonChart({ performance, metric }: ModelComparisonChartProps) {
  const data = useMemo(() => {
    return Object.entries(performance)
      .filter(([, metrics]) => metrics?.[metric])
      .map(([model, metrics]) => ({
        model: MODEL_LABELS[model] || model,
        mean: (metrics![metric]!.mean * 100),
        sd: (metrics![metric]!.sd * 100),
        fill: MODEL_COLORS[model] || "#888",
      }))
      .sort((a, b) => b.mean - a.mean);
  }, [performance, metric]);

  const metricLabel = metric.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());

  return (
    <div className="bg-card rounded-xl p-6 border border-border">
      <h3 className="text-lg font-semibold mb-4">{metricLabel} by Model</h3>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={data} layout="vertical" margin={{ left: 100, right: 20, top: 10, bottom: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis 
              type="number" 
              domain={[0, 100]} 
              tickFormatter={(v) => `${v}%`}
              stroke="hsl(var(--muted-foreground))"
            />
            <YAxis 
              type="category" 
              dataKey="model" 
              stroke="hsl(var(--muted-foreground))"
              tick={{ fill: "hsl(var(--foreground))" }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "8px",
              }}
              formatter={(value: number, _name, props) => [
                `${value.toFixed(1)}% ± ${props.payload.sd.toFixed(1)}%`,
                metricLabel
              ]}
            />
            <Legend />
            <Bar 
              dataKey="mean" 
              name={metricLabel}
              radius={[0, 4, 4, 0]}
              fill="hsl(var(--primary))"
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
