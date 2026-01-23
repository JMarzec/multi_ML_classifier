import { useMemo, useState } from "react";
import { ArrowUp, ArrowDown, Minus, Medal } from "lucide-react";
import { cn } from "@/lib/utils";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import type { MLResults, ModelPerformance } from "@/types/ml-results";

interface RunRankingSummaryProps {
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

const METRICS = ["auroc", "accuracy", "sensitivity", "specificity", "f1_score"] as const;
type MetricKey = (typeof METRICS)[number];

const METRIC_LABELS: Record<MetricKey, string> = {
  auroc: "AUROC",
  accuracy: "Accuracy",
  sensitivity: "Sens.",
  specificity: "Spec.",
  f1_score: "F1",
};

interface ModelRankingRow {
  model: string;
  modelLabel: string;
  metrics: {
    [K in MetricKey]: {
      bestRunIdx: number;
      bestValue: number;
      deltas: { runIdx: number; delta: number; isSignificant: boolean }[];
    } | null;
  };
}

export function RunRankingSummary({ runs, runColors, runLabels }: RunRankingSummaryProps) {
  const [baselineRunIdx, setBaselineRunIdx] = useState(0);

  const rankingData = useMemo(() => {
    const models = Object.keys(MODEL_LABELS) as (keyof ModelPerformance)[];
    
    return models.map((model) => {
      const metrics = {} as ModelRankingRow["metrics"];
      
      METRICS.forEach((metric) => {
        let bestRunIdx = -1;
        let bestValue = -1;
        let bestSd = 0;
        
        // Find best run for this model/metric
        runs.forEach((run, idx) => {
          const m = run.data.model_performance[model];
          const val = m?.[metric]?.mean;
          const sd = m?.[metric]?.sd ?? 0;
          if (typeof val === "number" && Number.isFinite(val) && val > bestValue) {
            bestValue = val;
            bestRunIdx = idx;
            bestSd = sd;
          }
        });
        
        if (bestRunIdx === -1) {
          metrics[metric] = null;
          return;
        }
        
        // Calculate deltas vs all runs
        const deltas = runs.map((run, idx) => {
          const runMetrics = run.data.model_performance[model];
          const runValue = runMetrics?.[metric]?.mean ?? 0;
          const runSd = runMetrics?.[metric]?.sd ?? 0;
          const delta = (bestValue - runValue) * 100; // percentage points
          const isSignificant = Math.abs(delta / 100) > 2 * Math.max(bestSd, runSd);
          return { runIdx: idx, delta, isSignificant };
        });
        
        metrics[metric] = {
          bestRunIdx,
          bestValue,
          deltas,
        };
      });
      
      return {
        model,
        modelLabel: MODEL_LABELS[model],
        metrics,
      };
    }).filter((row) => 
      // Only include models that have data in at least one run
      METRICS.some((m) => row.metrics[m] !== null)
    );
  }, [runs]);

  // Get only the runs that are available (for tabs)
  const availableTabs = runs.map((_, idx) => ({
    idx,
    label: runLabels[idx],
    color: runColors[idx],
  }));

  return (
    <div className="bg-card rounded-xl p-6 border border-border">
      <div className="flex items-center gap-2 mb-4">
        <Medal className="w-5 h-5 text-primary" />
        <h3 className="text-lg font-semibold">Per-Model Best Run Ranking</h3>
      </div>

      <Tabs value={String(baselineRunIdx)} onValueChange={(v) => setBaselineRunIdx(Number(v))}>
        <TabsList className="mb-4">
          {availableTabs.map(({ idx, label, color }) => (
            <TabsTrigger
              key={idx}
              value={String(idx)}
              className={cn(
                "data-[state=active]:bg-opacity-20",
                baselineRunIdx === idx && color.bg
              )}
            >
              Delta vs {label}
            </TabsTrigger>
          ))}
        </TabsList>

        {availableTabs.map(({ idx }) => (
          <TabsContent key={idx} value={String(idx)} className="mt-0">
            <RankingTable
              rankingData={rankingData}
              baselineRunIdx={idx}
              runs={runs}
              runColors={runColors}
              runLabels={runLabels}
            />
          </TabsContent>
        ))}
      </Tabs>
      
      <p className="text-xs text-muted-foreground mt-4">
        Showing best performing run for each model/metric combination. 
        Delta values are percentage points vs the selected baseline run. 
        <span className="text-accent ml-1">*</span> indicates statistically significant difference (&gt;2 SD).
      </p>
    </div>
  );
}

interface RankingTableProps {
  rankingData: ModelRankingRow[];
  baselineRunIdx: number;
  runs: { name: string; data: MLResults }[];
  runColors: { fill: string; text: string; bg: string; border: string }[];
  runLabels: string[];
}

function RankingTable({ rankingData, baselineRunIdx, runColors, runLabels }: RankingTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-border">
            <th className="text-left py-2 px-3">Model</th>
            {METRICS.map((m) => (
              <th key={m} className="text-center py-2 px-3 min-w-[100px]">
                {METRIC_LABELS[m]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rankingData.map(({ model, modelLabel, metrics }) => (
            <tr key={model} className="border-b border-border/50">
              <td className="py-3 px-3 font-medium">{modelLabel}</td>
              {METRICS.map((metric) => {
                const data = metrics[metric];
                if (!data) {
                  return <td key={metric} className="py-3 px-3 text-center text-muted-foreground">—</td>;
                }
                
                const colors = runColors[data.bestRunIdx];
                const deltaData = data.deltas.find(d => d.runIdx === baselineRunIdx);
                const delta = deltaData?.delta ?? 0;
                const isSignificant = deltaData?.isSignificant ?? false;
                const showDelta = data.bestRunIdx !== baselineRunIdx;
                
                return (
                  <td key={metric} className="py-3 px-3">
                    <div className="flex flex-col items-center gap-1">
                      <div className={cn(
                        "text-xs font-semibold px-2 py-0.5 rounded",
                        colors.bg,
                        colors.text
                      )}>
                        {runLabels[data.bestRunIdx]}
                      </div>
                      <span className="font-mono text-sm">
                        {(data.bestValue * 100).toFixed(1)}%
                      </span>
                      {showDelta && (
                        <div className={cn(
                          "flex items-center gap-0.5 text-xs font-mono",
                          delta > 0.5 && "text-accent",
                          delta < -0.5 && "text-destructive",
                          Math.abs(delta) <= 0.5 && "text-muted-foreground"
                        )}>
                          {delta > 0.5 ? (
                            <ArrowUp className="w-3 h-3" />
                          ) : delta < -0.5 ? (
                            <ArrowDown className="w-3 h-3" />
                          ) : (
                            <Minus className="w-3 h-3" />
                          )}
                          <span>
                            {delta >= 0 ? "+" : ""}
                            {delta.toFixed(1)}
                          </span>
                          {isSignificant && (
                            <span className="ml-1 text-accent">*</span>
                          )}
                        </div>
                      )}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
