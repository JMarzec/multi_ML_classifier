import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { Button } from "@/components/ui/button";
import { ThresholdSlider } from "./ThresholdSlider";
import type { ModelPerformance, ROCPoint } from "@/types/ml-results";

interface ROCCurveChartProps {
  performance: ModelPerformance;
}

const MODEL_COLORS: Record<string, string> = {
  rf: "hsl(var(--chart-rf))",
  svm: "hsl(var(--chart-svm))",
  xgboost: "hsl(var(--chart-xgb))",
  knn: "hsl(var(--chart-knn))",
  mlp: "hsl(var(--chart-mlp))",
  hard_vote: "hsl(var(--chart-ensemble))",
  soft_vote: "hsl(var(--primary))",
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

export function ROCCurveChart({ performance }: ROCCurveChartProps) {
  const [visibleModels, setVisibleModels] = useState<Set<string>>(
    new Set(Object.keys(performance).filter((k) => performance[k as keyof ModelPerformance]?.roc_curve))
  );
  const [thresholds, setThresholds] = useState<Record<string, number>>({});

  // Get models with ROC data
  const modelsWithROC = useMemo(() => {
    return Object.entries(performance)
      .filter(([, metrics]) => metrics?.roc_curve && metrics.roc_curve.length > 0)
      .map(([model]) => model);
  }, [performance]);

  // Combine all ROC curves into a single dataset for the chart
  const chartData = useMemo(() => {
    // Create unified x-axis points (FPR from 0 to 1)
    const fprPoints = Array.from({ length: 101 }, (_, i) => i / 100);
    
    return fprPoints.map((fpr) => {
      const point: Record<string, number> = { fpr };
      
      modelsWithROC.forEach((model) => {
        const modelKey = model as keyof ModelPerformance;
        const rocCurve = performance[modelKey]?.roc_curve;
        
        if (rocCurve && visibleModels.has(model)) {
          // Find closest FPR point and interpolate TPR
          const sortedCurve = [...rocCurve].sort((a, b) => a.fpr - b.fpr);
          
          // Binary search for closest point
          let lower = sortedCurve[0];
          let upper = sortedCurve[sortedCurve.length - 1];
          
          for (let i = 0; i < sortedCurve.length - 1; i++) {
            if (sortedCurve[i].fpr <= fpr && sortedCurve[i + 1].fpr >= fpr) {
              lower = sortedCurve[i];
              upper = sortedCurve[i + 1];
              break;
            }
          }
          
          // Linear interpolation
          if (upper.fpr === lower.fpr) {
            point[model] = lower.tpr;
          } else {
            const t = (fpr - lower.fpr) / (upper.fpr - lower.fpr);
            point[model] = lower.tpr + t * (upper.tpr - lower.tpr);
          }
        }
      });
      
      return point;
    });
  }, [performance, modelsWithROC, visibleModels]);

  const toggleModel = (model: string) => {
    setVisibleModels((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(model)) {
        newSet.delete(model);
      } else {
        newSet.add(model);
      }
      return newSet;
    });
  };

  // Calculate AUC values for display
  const aucValues = useMemo(() => {
    return modelsWithROC.map((model) => {
      const modelKey = model as keyof ModelPerformance;
      const auroc = performance[modelKey]?.auroc?.mean;
      return { model, auc: auroc ? (auroc * 100).toFixed(1) : "N/A" };
    });
  }, [modelsWithROC, performance]);

  if (modelsWithROC.length === 0) {
    return (
      <div className="bg-card rounded-xl p-12 border border-border text-center">
        <p className="text-muted-foreground">
          No ROC curve data available. Run the R script with ROC curve export enabled.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Model toggles */}
      <div className="flex flex-wrap gap-2">
        {modelsWithROC.map((model) => {
          const isVisible = visibleModels.has(model);
          const auc = aucValues.find((a) => a.model === model)?.auc;
          
          return (
            <Button
              key={model}
              variant={isVisible ? "default" : "outline"}
              size="sm"
              onClick={() => toggleModel(model)}
              style={{
                borderColor: MODEL_COLORS[model],
                backgroundColor: isVisible ? MODEL_COLORS[model] : "transparent",
                color: isVisible ? "hsl(var(--background))" : MODEL_COLORS[model],
              }}
            >
              {MODEL_LABELS[model]} (AUC: {auc}%)
            </Button>
          );
        })}
      </div>

      {/* ROC Chart */}
      <div className="bg-card rounded-xl p-6 border border-border">
        <h3 className="text-lg font-semibold mb-4">ROC Curves Comparison</h3>
        <ResponsiveContainer width="100%" height={450}>
          <LineChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis
              dataKey="fpr"
              type="number"
              domain={[0, 1]}
              tickFormatter={(v) => (v * 100).toFixed(0) + "%"}
              label={{ value: "False Positive Rate (1 - Specificity)", position: "bottom", offset: 0 }}
              stroke="hsl(var(--muted-foreground))"
            />
            <YAxis
              type="number"
              domain={[0, 1]}
              tickFormatter={(v) => (v * 100).toFixed(0) + "%"}
              label={{ value: "True Positive Rate (Sensitivity)", angle: -90, position: "insideLeft" }}
              stroke="hsl(var(--muted-foreground))"
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--popover))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "8px",
              }}
              labelFormatter={(value) => `FPR: ${(Number(value) * 100).toFixed(1)}%`}
              formatter={(value: number, name: string) => [
                `${(value * 100).toFixed(1)}%`,
                MODEL_LABELS[name] || name,
              ]}
            />
            <Legend />
            
            {/* Random classifier line */}
            <ReferenceLine
              segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
              stroke="hsl(var(--muted-foreground))"
              strokeDasharray="5 5"
              label={{ value: "Random", position: "insideBottomRight" }}
            />
            
            {/* Model curves */}
            {modelsWithROC.map((model) =>
              visibleModels.has(model) ? (
                <Line
                  key={model}
                  type="monotone"
                  dataKey={model}
                  name={MODEL_LABELS[model]}
                  stroke={MODEL_COLORS[model]}
                  strokeWidth={2}
                  dot={false}
                  activeDot={{ r: 4 }}
                />
              ) : null
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* AUC Summary Table */}
      <div className="bg-card rounded-xl p-5 border border-border">
        <h4 className="font-semibold mb-3">Area Under Curve (AUC) Summary</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-4">
          {aucValues.map(({ model, auc }) => (
            <div key={model} className="text-center p-3 rounded-lg bg-muted/30">
              <div
                className="w-3 h-3 rounded-full mx-auto mb-2"
                style={{ backgroundColor: MODEL_COLORS[model] }}
              />
              <p className="text-xs text-muted-foreground">{MODEL_LABELS[model]}</p>
              <p className="font-mono font-bold text-lg">{auc}%</p>
            </div>
          ))}
        </div>
      </div>

      {/* Threshold Sliders */}
      <div className="bg-card rounded-xl p-5 border border-border">
        <h4 className="font-semibold mb-4">Interactive Threshold Analysis</h4>
        <p className="text-sm text-muted-foreground mb-4">
          Adjust the decision threshold for each model to see how sensitivity and specificity trade off at different cutoff points.
        </p>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {modelsWithROC
            .filter((model) => visibleModels.has(model))
            .map((model) => {
              const modelKey = model as keyof ModelPerformance;
              const rocCurve = performance[modelKey]?.roc_curve;
              if (!rocCurve) return null;
              
              return (
                <ThresholdSlider
                  key={model}
                  threshold={thresholds[model] ?? 0.5}
                  onThresholdChange={(value) => setThresholds((prev) => ({ ...prev, [model]: value }))}
                  rocCurve={rocCurve}
                  modelName={MODEL_LABELS[model]}
                  color={MODEL_COLORS[model]}
                />
              );
            })}
        </div>
      </div>
    </div>
  );
}
