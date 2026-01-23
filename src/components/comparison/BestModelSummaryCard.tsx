import { useMemo } from "react";
import { Trophy, TrendingUp, Target, Activity, Percent } from "lucide-react";
import { cn } from "@/lib/utils";
import type { MLResults, ModelPerformance, ModelMetrics } from "@/types/ml-results";

interface BestModelSummaryCardProps {
  runs: { name: string; data: MLResults }[];
  runColors: { fill: string; text: string; bg: string; border: string }[];
  runLabels: string[];
}

const METRICS = [
  { key: "auroc", label: "AUROC", icon: TrendingUp },
  { key: "accuracy", label: "Accuracy", icon: Target },
  { key: "sensitivity", label: "Sensitivity", icon: Activity },
  { key: "specificity", label: "Specificity", icon: Percent },
  { key: "f1_score", label: "F1 Score", icon: Trophy },
] as const;

type MetricKey = (typeof METRICS)[number]["key"];

interface BestRunInfo {
  runIdx: number;
  model: string;
  value: number;
  sd: number;
  isSignificant: boolean; // > 2 SD difference from next best
}

function getBestRunForMetric(
  runs: { name: string; data: MLResults }[],
  metric: MetricKey
): BestRunInfo | null {
  let best: { runIdx: number; model: string; value: number; sd: number } | null = null;
  let secondBest: { value: number } | null = null;

  runs.forEach((run, runIdx) => {
    const models = Object.entries(run.data.model_performance) as [keyof ModelPerformance, ModelMetrics | undefined][];
    models.forEach(([model, metrics]) => {
      const val = metrics?.[metric]?.mean;
      const sd = metrics?.[metric]?.sd ?? 0;
      if (typeof val === "number" && Number.isFinite(val)) {
        if (!best || val > best.value) {
          secondBest = best ? { value: best.value } : null;
          best = { runIdx, model, value: val, sd };
        } else if (!secondBest || val > secondBest.value) {
          secondBest = { value: val };
        }
      }
    });
  });

  if (!best) return null;

  // Statistical significance: difference > 2 * pooled SD
  const isSignificant = secondBest 
    ? (best.value - secondBest.value) > 2 * best.sd
    : true;

  return { ...best, isSignificant };
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

export function BestModelSummaryCard({ runs, runColors, runLabels }: BestModelSummaryCardProps) {
  const bestByMetric = useMemo(() => {
    return METRICS.map(({ key, label, icon }) => ({
      metric: key,
      label,
      icon,
      best: getBestRunForMetric(runs, key),
    }));
  }, [runs]);

  return (
    <div className="bg-card rounded-xl p-6 border border-border">
      <div className="flex items-center gap-2 mb-4">
        <Trophy className="w-5 h-5 text-accent" />
        <h3 className="text-lg font-semibold">Best Performing Run by Metric</h3>
      </div>
      
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        {bestByMetric.map(({ metric, label, icon: Icon, best }) => {
          if (!best) return null;
          const colors = runColors[best.runIdx];
          
          return (
            <div 
              key={metric} 
              className={cn(
                "rounded-lg p-4 border transition-all",
                colors.bg,
                colors.border
              )}
            >
              <div className="flex items-center gap-2 mb-2">
                <Icon className="w-4 h-4 text-muted-foreground" />
                <span className="text-sm font-medium text-muted-foreground">{label}</span>
              </div>
              
              <div className="flex items-baseline gap-2">
                <span className={cn("text-2xl font-bold", colors.text)}>
                  {(best.value * 100).toFixed(1)}%
                </span>
                {best.isSignificant && (
                  <span className="text-xs px-1.5 py-0.5 rounded bg-accent/20 text-accent font-medium">
                    sig.
                  </span>
                )}
              </div>
              
              <div className="mt-2 space-y-1">
                <p className={cn("text-sm font-medium", colors.text)}>
                  {runLabels[best.runIdx]}
                </p>
                <p className="text-xs text-muted-foreground">
                  {MODEL_LABELS[best.model] || best.model}
                </p>
                <p className="text-xs text-muted-foreground">
                  ±{(best.sd * 100).toFixed(1)}% SD
                </p>
              </div>
            </div>
          );
        })}
      </div>
      
      <p className="text-xs text-muted-foreground mt-4">
        <span className="px-1.5 py-0.5 rounded bg-accent/20 text-accent font-medium mr-1">sig.</span>
        indicates statistically meaningful difference (&gt;2 SD) from next best result
      </p>
    </div>
  );
}
