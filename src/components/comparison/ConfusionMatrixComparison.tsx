import { useMemo, useState } from "react";
import { cn } from "@/lib/utils";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import type { MLResults, ConfusionMatrixData, ModelPerformance } from "@/types/ml-results";

interface ConfusionMatrixComparisonProps {
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

function calculateMetrics(data: ConfusionMatrixData) {
  const { tp, tn, fp, fn } = data;
  const total = tp + tn + fp + fn;
  
  return {
    accuracy: total > 0 ? ((tp + tn) / total * 100) : 0,
    precision: tp + fp > 0 ? (tp / (tp + fp) * 100) : 0,
    sensitivity: tp + fn > 0 ? (tp / (tp + fn) * 100) : 0,
    specificity: tn + fp > 0 ? (tn / (tn + fp) * 100) : 0,
    f1: (() => {
      const prec = tp + fp > 0 ? tp / (tp + fp) : 0;
      const rec = tp + fn > 0 ? tp / (tp + fn) : 0;
      return prec + rec > 0 ? (2 * prec * rec / (prec + rec) * 100) : 0;
    })(),
    npv: tn + fn > 0 ? (tn / (tn + fn) * 100) : 0,
  };
}

export function ConfusionMatrixComparison({
  runs,
  runColors,
  runLabels,
}: ConfusionMatrixComparisonProps) {
  // Get available models across all runs
  const availableModels = useMemo(() => {
    const modelsSet = new Set<string>();
    runs.forEach(run => {
      Object.entries(run.data.model_performance).forEach(([model, metrics]) => {
        if (metrics?.confusion_matrix) {
          modelsSet.add(model);
        }
      });
    });
    return Array.from(modelsSet);
  }, [runs]);

  const [selectedModel, setSelectedModel] = useState<string>(
    availableModels.includes("soft_vote") ? "soft_vote" : availableModels[0] || "rf"
  );

  // Get confusion matrices for each run
  const matrices = useMemo(() => {
    return runs.map(run => {
      const metrics = run.data.model_performance[selectedModel as keyof ModelPerformance];
      return metrics?.confusion_matrix || null;
    });
  }, [runs, selectedModel]);

  if (availableModels.length === 0) {
    return (
      <div className="bg-card rounded-xl p-6 border border-border">
        <h3 className="text-lg font-semibold mb-4">Confusion Matrix Comparison</h3>
        <p className="text-muted-foreground text-center py-8">
          No confusion matrix data available in the uploaded runs.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-card rounded-xl p-6 border border-border">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold">Confusion Matrix Comparison</h3>
        <Select value={selectedModel} onValueChange={setSelectedModel}>
          <SelectTrigger className="w-[200px]">
            <SelectValue placeholder="Select model" />
          </SelectTrigger>
          <SelectContent>
            {availableModels.map(model => (
              <SelectItem key={model} value={model}>
                {MODEL_LABELS[model] || model}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      {/* Matrix grid */}
      <div className={cn(
        "grid gap-4",
        runs.length === 2 && "grid-cols-1 md:grid-cols-2",
        runs.length === 3 && "grid-cols-1 md:grid-cols-3",
        runs.length === 4 && "grid-cols-1 md:grid-cols-2 lg:grid-cols-4"
      )}>
        {runs.map((run, idx) => {
          const matrix = matrices[idx];
          const colors = runColors[idx];
          
          if (!matrix) {
            return (
              <div key={run.name} className={cn("rounded-lg p-4 border", colors.border, colors.bg)}>
                <div className="flex items-center gap-2 mb-3">
                  <div className={cn("w-3 h-3 rounded-full", colors.text.replace("text-", "bg-"))} />
                  <span className={cn("text-sm font-medium", colors.text)}>{runLabels[idx]}</span>
                </div>
                <p className="text-muted-foreground text-center py-8 text-sm">
                  No data available
                </p>
              </div>
            );
          }

          const metrics = calculateMetrics(matrix);

          return (
            <div key={run.name} className={cn("rounded-lg p-4 border", colors.border, colors.bg)}>
              {/* Run header */}
              <div className="flex items-center gap-2 mb-3">
                <div className={cn("w-3 h-3 rounded-full", colors.text.replace("text-", "bg-"))} />
                <span className={cn("text-sm font-medium", colors.text)}>{runLabels[idx]}</span>
              </div>

              {/* Compact matrix */}
              <div className="grid grid-cols-2 gap-1 mb-3">
                <div className="aspect-square bg-warning/20 rounded flex flex-col items-center justify-center text-center p-1">
                  <span className="text-lg font-bold">{matrix.fn}</span>
                  <span className="text-[10px] text-muted-foreground">FN</span>
                </div>
                <div className="aspect-square bg-success/20 rounded flex flex-col items-center justify-center text-center p-1">
                  <span className="text-lg font-bold text-success">{matrix.tp}</span>
                  <span className="text-[10px] text-muted-foreground">TP</span>
                </div>
                <div className="aspect-square bg-success/20 rounded flex flex-col items-center justify-center text-center p-1">
                  <span className="text-lg font-bold text-success">{matrix.tn}</span>
                  <span className="text-[10px] text-muted-foreground">TN</span>
                </div>
                <div className="aspect-square bg-warning/20 rounded flex flex-col items-center justify-center text-center p-1">
                  <span className="text-lg font-bold">{matrix.fp}</span>
                  <span className="text-[10px] text-muted-foreground">FP</span>
                </div>
              </div>

              {/* Derived metrics */}
              <div className="grid grid-cols-2 gap-x-3 gap-y-1 text-xs border-t border-border/50 pt-2">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Accuracy</span>
                  <span className="font-mono font-semibold">{metrics.accuracy.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Precision</span>
                  <span className="font-mono font-semibold">{metrics.precision.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Sensitivity</span>
                  <span className="font-mono font-semibold text-success">{metrics.sensitivity.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Specificity</span>
                  <span className="font-mono font-semibold text-info">{metrics.specificity.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">F1 Score</span>
                  <span className="font-mono font-semibold">{metrics.f1.toFixed(1)}%</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">NPV</span>
                  <span className="font-mono font-semibold">{metrics.npv.toFixed(1)}%</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Comparison summary table */}
      <div className="mt-6 pt-4 border-t border-border">
        <h4 className="text-sm font-medium mb-3">Metrics Summary</h4>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-2 px-2">Metric</th>
                {runs.map((_, idx) => (
                  <th key={idx} className={cn("text-center py-2 px-2", runColors[idx].text)}>
                    {runLabels[idx]}
                  </th>
                ))}
                <th className="text-center py-2 px-2 text-muted-foreground">Best</th>
              </tr>
            </thead>
            <tbody>
              {["accuracy", "precision", "sensitivity", "specificity", "f1", "npv"].map(metric => {
                const values = matrices.map(m => m ? calculateMetrics(m)[metric as keyof ReturnType<typeof calculateMetrics>] : null);
                const validValues = values.filter((v): v is number => v !== null);
                const maxVal = validValues.length > 0 ? Math.max(...validValues) : null;
                const bestIdx = maxVal !== null ? values.findIndex(v => v === maxVal) : -1;
                
                return (
                  <tr key={metric} className="border-b border-border/50">
                    <td className="py-2 px-2 capitalize text-muted-foreground">{metric}</td>
                    {values.map((val, idx) => (
                      <td
                        key={idx}
                        className={cn(
                          "py-2 px-2 text-center font-mono",
                          val !== null && val === maxVal && "font-bold",
                          runColors[idx].text
                        )}
                      >
                        {val !== null ? `${val.toFixed(1)}%` : "—"}
                      </td>
                    ))}
                    <td className={cn("py-2 px-2 text-center font-medium", bestIdx >= 0 && runColors[bestIdx].text)}>
                      {bestIdx >= 0 ? runLabels[bestIdx] : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
