import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
  Legend,
} from "recharts";
import type { MLResults } from "@/types/ml-results";

interface PermutationDistributionTabProps {
  data: MLResults;
}

const MODEL_COLORS: Record<string, string> = {
  rf: "hsl(var(--primary))",
  svm: "hsl(var(--secondary))",
  xgboost: "hsl(var(--accent))",
  knn: "hsl(var(--info))",
  mlp: "hsl(var(--warning))",
  soft_vote: "hsl(var(--success))",
  hard_vote: "hsl(var(--muted-foreground))",
};

const MODEL_LABELS: Record<string, string> = {
  rf: "Random Forest",
  svm: "SVM",
  xgboost: "XGBoost",
  knn: "KNN",
  mlp: "MLP",
  soft_vote: "Soft Voting",
  hard_vote: "Hard Voting",
};

export function PermutationDistributionTab({ data }: PermutationDistributionTabProps) {
  const permDist = data.permutation_distributions;
  const [selectedMetric, setSelectedMetric] = useState<"auroc" | "accuracy">("auroc");
  const [selectedModel, setSelectedModel] = useState<string>("rf");

  if (!permDist || Object.keys(permDist).length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Permutation Distribution Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No permutation distribution data available. Re-run the R script with the updated exporter
            to include per-model AUROC and Accuracy distributions from actual vs permuted data.
          </p>
        </CardContent>
      </Card>
    );
  }

  const availableModels = Object.keys(permDist);
  const currentModelData = permDist[selectedModel];

  // Build histogram data for the selected model and metric
  // The R script exports arrays directly, not objects with actual_values/permuted_values
  const buildHistogramData = () => {
    if (!currentModelData) return [];

    const permMetricData = currentModelData[selectedMetric];
    const actualMetricData = data.actual_distributions?.[selectedModel as keyof typeof data.actual_distributions]?.[selectedMetric];
    
    const permuted_values = Array.isArray(permMetricData) ? permMetricData : [];
    const actual_values = Array.isArray(actualMetricData) ? actualMetricData : [];

    // Combine all values for binning
    const allValues = [...(actual_values || []), ...(permuted_values || [])];
    if (allValues.length === 0) return [];

    const min = Math.min(...allValues);
    const max = Math.max(...allValues);
    const binCount = 20;
    const binWidth = (max - min) / binCount || 0.05;

    const bins: { bin: string; binCenter: number; actual: number; permuted: number }[] = [];

    for (let i = 0; i < binCount; i++) {
      const binStart = min + i * binWidth;
      const binEnd = binStart + binWidth;
      const binCenter = (binStart + binEnd) / 2;

      const actualCount = (actual_values || []).filter((v) => v >= binStart && v < binEnd).length;
      const permutedCount = (permuted_values || []).filter((v) => v >= binStart && v < binEnd).length;

      bins.push({
        bin: `${(binStart * 100).toFixed(0)}-${(binEnd * 100).toFixed(0)}%`,
        binCenter,
        actual: actualCount,
        permuted: permutedCount,
      });
    }

    return bins;
  };

  const histogramData = buildHistogramData();

  // Calculate p-value approximation
  const calculatePValue = () => {
    const permMetricData = currentModelData?.[selectedMetric];
    const actualMetricData = data.actual_distributions?.[selectedModel as keyof typeof data.actual_distributions]?.[selectedMetric];
    
    const permuted_values = Array.isArray(permMetricData) ? permMetricData : [];
    const actual_values = Array.isArray(actualMetricData) ? actualMetricData : [];
    
    if (!actual_values.length || !permuted_values.length) return null;

    const actualMean = actual_values.reduce((a, b) => a + b, 0) / actual_values.length;
    const extremeCount = permuted_values.filter((v) => v >= actualMean).length;
    return extremeCount / permuted_values.length;
  };

  const pValue = calculatePValue();
  
  const actualMetricVals = data.actual_distributions?.[selectedModel as keyof typeof data.actual_distributions]?.[selectedMetric];
  const permMetricVals = currentModelData?.[selectedMetric];
  
  const actualMean = Array.isArray(actualMetricVals) && actualMetricVals.length
    ? actualMetricVals.reduce((a, b) => a + b, 0) / actualMetricVals.length
    : null;
  const permutedMean = Array.isArray(permMetricVals) && permMetricVals.length
    ? permMetricVals.reduce((a, b) => a + b, 0) / permMetricVals.length
    : null;

  // Summary across all models for the selected metric
  const modelSummary = availableModels.map((model) => {
    const permMetricData = permDist[model]?.[selectedMetric];
    const actualMetricData = data.actual_distributions?.[model as keyof typeof data.actual_distributions]?.[selectedMetric];

    const actualVals = Array.isArray(actualMetricData) ? actualMetricData : [];
    const permVals = Array.isArray(permMetricData) ? permMetricData : [];

    const actualM = actualVals.length ? actualVals.reduce((a, b) => a + b, 0) / actualVals.length : 0;
    const permM = permVals.length ? permVals.reduce((a, b) => a + b, 0) / permVals.length : 0;
    const extreme = permVals.filter((v) => v >= actualM).length;
    const pVal = permVals.length ? extreme / permVals.length : 1;

    return {
      model,
      label: MODEL_LABELS[model] || model,
      actualMean: actualM,
      permutedMean: permM,
      pValue: pVal,
      significant: pVal < 0.05,
    };
  }).filter(Boolean) as { model: string; label: string; actualMean: number; permutedMean: number; pValue: number; significant: boolean }[];

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="flex flex-wrap gap-4">
        <div className="space-y-2">
          <label className="text-sm font-medium">Metric</label>
          <div className="flex gap-2">
            {(["auroc", "accuracy"] as const).map((m) => (
              <Button
                key={m}
                variant={selectedMetric === m ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedMetric(m)}
              >
                {m.toUpperCase()}
              </Button>
            ))}
          </div>
        </div>

        <div className="space-y-2">
          <label className="text-sm font-medium">Model</label>
          <div className="flex flex-wrap gap-2">
            {availableModels.map((m) => (
              <Button
                key={m}
                variant={selectedModel === m ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedModel(m)}
              >
                {MODEL_LABELS[m] || m}
              </Button>
            ))}
          </div>
        </div>
      </div>

      {/* Distribution histogram */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            {MODEL_LABELS[selectedModel] || selectedModel} - {selectedMetric.toUpperCase()} Distribution
            {pValue !== null && (
              <Badge variant={pValue < 0.05 ? "default" : "secondary"}>
                p = {pValue.toFixed(3)}
              </Badge>
            )}
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Comparison of actual CV-fold {selectedMetric.toUpperCase()} values vs permuted (random chance) distribution.
          </p>
        </CardHeader>
        <CardContent>
          {histogramData.length > 0 ? (
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={histogramData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="bin" stroke="hsl(var(--muted-foreground))" tick={{ fontSize: 10 }} />
                <YAxis stroke="hsl(var(--muted-foreground))" label={{ value: "Frequency", angle: -90, position: "insideLeft" }} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--popover))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                  }}
                />
                <Legend />
                <Bar dataKey="actual" name="Actual Data" fill="hsl(var(--success))" opacity={0.8} />
                <Bar dataKey="permuted" name="Permuted Data" fill="hsl(var(--destructive))" opacity={0.5} />
                {actualMean !== null && (
                  <ReferenceLine
                    x={histogramData.find((b) => b.binCenter >= actualMean)?.bin}
                    stroke="hsl(var(--success))"
                    strokeWidth={2}
                    strokeDasharray="5 5"
                    label={{ value: "Actual μ", position: "top", fill: "hsl(var(--success))" }}
                  />
                )}
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <p className="text-muted-foreground text-center py-8">No distribution data for this selection.</p>
          )}

          {/* Stats summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
            <div className="bg-muted/30 rounded-lg p-3 text-center">
              <p className="text-xs text-muted-foreground">Actual Mean</p>
              <p className="text-xl font-bold text-success">
                {actualMean !== null ? (actualMean * 100).toFixed(1) + "%" : "-"}
              </p>
            </div>
            <div className="bg-muted/30 rounded-lg p-3 text-center">
              <p className="text-xs text-muted-foreground">Permuted Mean</p>
              <p className="text-xl font-bold text-destructive">
                {permutedMean !== null ? (permutedMean * 100).toFixed(1) + "%" : "-"}
              </p>
            </div>
            <div className="bg-muted/30 rounded-lg p-3 text-center">
              <p className="text-xs text-muted-foreground">Effect Size</p>
              <p className="text-xl font-bold">
                {actualMean !== null && permutedMean !== null
                  ? ((actualMean - permutedMean) * 100).toFixed(1) + "%"
                  : "-"}
              </p>
            </div>
            <div className="bg-muted/30 rounded-lg p-3 text-center">
              <p className="text-xs text-muted-foreground">P-Value</p>
              <p className={`text-xl font-bold ${pValue !== null && pValue < 0.05 ? "text-success" : "text-destructive"}`}>
                {pValue !== null ? pValue.toFixed(4) : "-"}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* All models comparison */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">All Models - {selectedMetric.toUpperCase()} Comparison</CardTitle>
          <p className="text-sm text-muted-foreground">
            Actual vs permuted mean performance across all models.
          </p>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelSummary} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis dataKey="label" stroke="hsl(var(--muted-foreground))" />
              <YAxis stroke="hsl(var(--muted-foreground))" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                }}
                formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
              />
              <Legend />
              <Bar dataKey="actualMean" name="Actual" fill="hsl(var(--success))" />
              <Bar dataKey="permutedMean" name="Permuted" fill="hsl(var(--destructive))" opacity={0.5} />
            </BarChart>
          </ResponsiveContainer>

          {/* Significance table */}
          <div className="mt-4 overflow-auto rounded-lg border border-border">
            <table className="w-full text-sm">
              <thead className="bg-muted/50">
                <tr>
                  <th className="text-left p-2">Model</th>
                  <th className="text-right p-2">Actual Mean</th>
                  <th className="text-right p-2">Permuted Mean</th>
                  <th className="text-right p-2">P-Value</th>
                  <th className="text-center p-2">Significant</th>
                </tr>
              </thead>
              <tbody>
                {modelSummary.map((m) => (
                  <tr key={m.model} className="border-t border-border">
                    <td className="p-2 font-medium">{m.label}</td>
                    <td className="text-right p-2 font-mono text-success">{(m.actualMean * 100).toFixed(1)}%</td>
                    <td className="text-right p-2 font-mono text-destructive">{(m.permutedMean * 100).toFixed(1)}%</td>
                    <td className="text-right p-2 font-mono">{m.pValue.toFixed(4)}</td>
                    <td className="text-center p-2">
                      {m.significant ? (
                        <Badge variant="default" className="bg-success text-success-foreground">Yes</Badge>
                      ) : (
                        <Badge variant="secondary">No</Badge>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </CardContent>
      </Card>

      {/* Interpretation */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Interpretation Guide</CardTitle>
        </CardHeader>
        <CardContent className="space-y-3 text-sm text-muted-foreground">
          <p>
            <strong className="text-foreground">Permutation testing</strong> validates whether your model's
            performance is significantly better than random chance by comparing actual results against
            results from shuffled (permuted) class labels.
          </p>
          <p>
            <strong className="text-foreground">P-value {"<"} 0.05:</strong> The model performs significantly
            better than random — the learned patterns are meaningful.
          </p>
          <p>
            <strong className="text-foreground">P-value {"≥"} 0.05:</strong> Performance may be due to chance.
            Consider collecting more data or reviewing feature quality.
          </p>
        </CardContent>
      </Card>
    </div>
  );
}
