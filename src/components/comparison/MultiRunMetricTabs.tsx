import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import type { MLResults, ModelPerformance, ModelMetrics } from "@/types/ml-results";

interface MultiRunMetricTabsProps {
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

type MetricKey = keyof Pick<NonNullable<ModelMetrics>, "accuracy" | "auroc" | "sensitivity" | "specificity" | "f1_score">;

const METRIC_TABS: { key: MetricKey; label: string }[] = [
  { key: "auroc", label: "AUROC" },
  { key: "accuracy", label: "Accuracy" },
  { key: "sensitivity", label: "Sensitivity" },
  { key: "specificity", label: "Specificity" },
  { key: "f1_score", label: "F1" },
];

function metricValue(metrics: ModelMetrics | undefined, metric: MetricKey): number {
  const mean = metrics?.[metric]?.mean;
  return typeof mean === "number" && Number.isFinite(mean) ? mean * 100 : 0;
}

function MetricChart({
  runs,
  runColors,
  runLabels,
  metric,
}: {
  runs: { name: string; data: MLResults }[];
  runColors: { fill: string; text: string; bg: string; border: string }[];
  runLabels: string[];
  metric: MetricKey;
}) {
  const data = useMemo(() => {
    const models = Object.keys(MODEL_LABELS) as (keyof ModelPerformance)[];
    return models
      .map((model) => {
        const entry: Record<string, string | number> = { model: MODEL_LABELS[model] };
        runs.forEach((run, idx) => {
          const m = run.data.model_performance[model];
          entry[`v${idx}`] = metricValue(m, metric);
        });
        return entry;
      })
      .filter((row) => runs.some((_, idx) => (row[`v${idx}`] as number) > 0));
  }, [runs, metric]);

  const title = METRIC_TABS.find((t) => t.key === metric)?.label ?? metric;

  return (
    <div className="bg-card rounded-xl p-6 border border-border">
      <h3 className="text-lg font-semibold mb-4">{title} Comparison by Model</h3>
      <ResponsiveContainer width="100%" height={350}>
        <BarChart data={data} layout="vertical" margin={{ left: 110, right: 10 }}>
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
            tick={{ fontSize: 12 }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--popover))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "8px",
            }}
            formatter={(value: number) => [`${value.toFixed(1)}%`, ""]}
          />
          <Legend />
          {runs.map((_, idx) => (
            <Bar
              key={idx}
              dataKey={`v${idx}`}
              name={runLabels[idx]}
              fill={runColors[idx].fill}
              radius={[0, 4, 4, 0]}
            />
          ))}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export function MultiRunMetricTabs({ runs, runColors, runLabels }: MultiRunMetricTabsProps) {
  return (
    <Tabs defaultValue="auroc" className="w-full">
      <div className="flex items-center justify-between gap-3 flex-wrap mb-4">
        <h3 className="text-lg font-semibold">Performance Metrics</h3>
        <TabsList>
          {METRIC_TABS.map((t) => (
            <TabsTrigger key={t.key} value={t.key}>
              {t.label}
            </TabsTrigger>
          ))}
        </TabsList>
      </div>

      {METRIC_TABS.map((t) => (
        <TabsContent key={t.key} value={t.key}>
          <MetricChart runs={runs} runColors={runColors} runLabels={runLabels} metric={t.key} />
        </TabsContent>
      ))}
    </Tabs>
  );
}
