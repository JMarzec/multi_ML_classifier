import { useMemo } from "react";
import {
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { ArrowUp, ArrowDown, Minus, Users } from "lucide-react";
import { cn } from "@/lib/utils";
import { ComparisonReportExport } from "./ComparisonReportExport";
import { FeatureUpsetPlot } from "./comparison/FeatureUpsetPlot";
import { SurvivalComparisonSection } from "./comparison/SurvivalComparisonSection";
import { FeatureDetailsSection } from "./comparison/FeatureDetailsSection";
import { MultiRunMetricTabs } from "./comparison/MultiRunMetricTabs";
import { ROCRunsOverlayChart } from "./comparison/ROCRunsOverlayChart";
import { ConfusionMatrixComparison } from "./comparison/ConfusionMatrixComparison";
import { ModelSignaturesSection } from "./comparison/ModelSignaturesSection";
import { BestModelSummaryCard } from "./comparison/BestModelSummaryCard";
import { RunRankingSummary } from "./comparison/RunRankingSummary";
import { FeatureStabilityPanel } from "./comparison/FeatureStabilityPanel";
import type { MLResults, ModelPerformance } from "@/types/ml-results";

export interface ComparisonRun {
  name: string;
  data: MLResults;
}

interface ComparisonDashboardProps {
  runs: ComparisonRun[];
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

export const RUN_COLORS = [
  { fill: "hsl(var(--primary))", text: "text-primary", bg: "bg-primary/10", border: "border-primary/30" },
  { fill: "hsl(var(--secondary))", text: "text-secondary", bg: "bg-secondary/10", border: "border-secondary/30" },
  { fill: "hsl(var(--accent))", text: "text-accent", bg: "bg-accent/10", border: "border-accent/30" },
  { fill: "hsl(var(--warning))", text: "text-warning", bg: "bg-warning/10", border: "border-warning/30" },
];

export const RUN_LABELS = ["Run A", "Run B", "Run C", "Run D"];

const METRICS = ["accuracy", "auroc", "sensitivity", "specificity", "f1_score"] as const;

export function ComparisonDashboard({ runs }: ComparisonDashboardProps) {
  // Best model per run
  const bestModels = useMemo(() => {
    return runs.map(run => {
      let best = { model: "", auroc: 0 };
      Object.entries(run.data.model_performance).forEach(([model, metrics]) => {
        if (metrics?.auroc?.mean && metrics.auroc.mean > best.auroc) {
          best = { model: MODEL_LABELS[model] || model, auroc: metrics.auroc.mean };
        }
      });
      return best;
    });
  }, [runs]);

  // Radar chart data for soft_vote comparison
  const radarData = useMemo(() => {
    return METRICS.map(metric => {
      const entry: Record<string, string | number> = {
        metric: metric.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()),
      };
      
      runs.forEach((run, idx) => {
        const metrics = run.data.model_performance.soft_vote;
        entry[`run${idx}`] = metrics?.[metric]?.mean ? metrics[metric].mean * 100 : 0;
      });
      
      return entry;
    });
  }, [runs]);

  // Feature overlap analysis (legacy for basic stats)
  const featureComparison = useMemo(() => {
    const featureSets = runs.map(run => new Set(run.data.selected_features || []));
    
    let common: string[] = [];
    if (featureSets.length > 0) {
      common = [...featureSets[0]].filter(f => 
        featureSets.every(set => set.has(f))
      );
    }
    
    const uniquePerRun = featureSets.map((set, idx) => 
      [...set].filter(f => 
        featureSets.every((otherSet, otherIdx) => 
          otherIdx === idx || !otherSet.has(f)
        )
      )
    );
    
    return { common, uniquePerRun, totals: featureSets.map(s => s.size) };
  }, [runs]);

  // Feature sets for UpSet plot
  const featureSetsForUpset = useMemo(() => {
    return runs.map(run => ({
      name: run.name,
      features: run.data.selected_features || [],
    }));
  }, [runs]);

  // Performance difference table (comparing to first run)
  const diffData = useMemo(() => {
    if (runs.length < 2) return [];
    
    const models = Object.keys(MODEL_LABELS) as (keyof ModelPerformance)[];
    
    return models.map(model => {
      const baseMetrics = runs[0].data.model_performance[model];
      const baseAuroc = baseMetrics?.auroc?.mean || 0;
      
      const diffs = runs.slice(1).map((run, idx) => {
        const metrics = run.data.model_performance[model];
        const auroc = metrics?.auroc?.mean || 0;
        return { idx: idx + 1, diff: (auroc - baseAuroc) * 100 };
      });
      
      return { model: MODEL_LABELS[model], diffs };
    }).filter(d => runs[0].data.model_performance[d.model as keyof ModelPerformance] !== undefined);
  }, [runs]);

  return (
    <div className="space-y-6">
      {/* Header with export */}
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-bold">Analysis Comparison ({runs.length} Runs)</h2>
        <ComparisonReportExport runs={runs} />
      </div>

      {/* Run summary cards */}
      <div className={cn(
        "grid gap-4",
        runs.length === 2 && "grid-cols-1 md:grid-cols-2",
        runs.length === 3 && "grid-cols-1 md:grid-cols-3",
        runs.length === 4 && "grid-cols-1 md:grid-cols-2 lg:grid-cols-4"
      )}>
        {runs.map((run, idx) => {
          const colors = RUN_COLORS[idx];
          const sampleCount = run.data.preprocessing?.original_samples 
            ?? run.data.profile_ranking?.all_rankings?.length 
            ?? 0;
          return (
            <div key={run.name} className={cn("rounded-xl p-5 border", colors.bg, colors.border)}>
              <div className="flex items-center gap-2 mb-2">
                <div className={cn("w-3 h-3 rounded-full", colors.text.replace("text-", "bg-"))} />
                <span className={cn("text-sm font-medium", colors.text)}>{RUN_LABELS[idx]}</span>
              </div>
              <h3 className="font-semibold truncate">{run.name}</h3>
              <p className="text-sm text-muted-foreground">
                Generated: {new Date(run.data.metadata.generated_at).toLocaleDateString()}
              </p>
              
              {/* Sample count */}
              <div className="flex items-center gap-1.5 mt-2 text-sm text-muted-foreground">
                <Users className="w-3.5 h-3.5" />
                <span>{sampleCount} samples</span>
              </div>
              
              <div className="mt-3 pt-3 border-t border-border/30">
                <p className="text-sm text-muted-foreground">Best Model</p>
                <p className="font-semibold">{bestModels[idx].model}</p>
                <p className={cn("text-2xl font-bold", colors.text)}>
                  {(bestModels[idx].auroc * 100).toFixed(1)}% AUROC
                </p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Best Model Summary Card */}
      <BestModelSummaryCard runs={runs} runColors={RUN_COLORS} runLabels={RUN_LABELS} />

      {/* Metric comparison tabs */}
      <MultiRunMetricTabs runs={runs} runColors={RUN_COLORS} runLabels={RUN_LABELS} />

      {/* Confusion Matrix Comparison */}
      <ConfusionMatrixComparison
        runs={runs}
        runColors={RUN_COLORS}
        runLabels={RUN_LABELS}
      />

      {/* ROC overlay across runs */}
      <ROCRunsOverlayChart runs={runs} runColors={RUN_COLORS} runLabels={RUN_LABELS} />

      {/* Difference Table (vs Run A) */}
      {runs.length >= 2 && (
        <div className="bg-card rounded-xl p-6 border border-border">
          <h3 className="text-lg font-semibold mb-4">Performance Difference vs Run A</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 px-3">Model</th>
                  {runs.slice(1).map((_, idx) => (
                    <th key={idx} className={cn("text-center py-2 px-3", RUN_COLORS[idx + 1].text)}>
                      {RUN_LABELS[idx + 1]} - A
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {diffData.map(({ model, diffs }) => (
                  <tr key={model} className="border-b border-border/50">
                    <td className="py-2 px-3 text-muted-foreground">{model}</td>
                    {diffs.map(({ idx, diff }) => (
                      <td key={idx} className="py-2 px-3 text-center">
                        <div className={cn(
                          "flex items-center justify-center gap-1 font-mono font-bold",
                          diff > 0.5 && "text-accent",
                          diff < -0.5 && "text-destructive",
                          Math.abs(diff) <= 0.5 && "text-muted-foreground"
                        )}>
                          {diff > 0.5 ? <ArrowUp className="w-3 h-3" /> : 
                           diff < -0.5 ? <ArrowDown className="w-3 h-3" /> : 
                           <Minus className="w-3 h-3" />}
                          <span>{diff >= 0 ? "+" : ""}{diff.toFixed(1)}%</span>
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Per-Model Run Ranking */}
      <RunRankingSummary runs={runs} runColors={RUN_COLORS} runLabels={RUN_LABELS} />

      {/* Radar Chart - Ensemble Comparison */}
      <div className="bg-card rounded-xl p-6 border border-border">
        <h3 className="text-lg font-semibold mb-4">Soft Voting Ensemble - Metrics Comparison</h3>
        <ResponsiveContainer width="100%" height={400}>
          <RadarChart data={radarData}>
            <PolarGrid stroke="hsl(var(--border))" />
            <PolarAngleAxis dataKey="metric" stroke="hsl(var(--muted-foreground))" />
            <PolarRadiusAxis
              angle={30}
              domain={[0, 100]}
              stroke="hsl(var(--muted-foreground))"
              tickFormatter={(v) => `${v}%`}
            />
            {runs.map((_, idx) => (
              <Radar
                key={idx}
                name={RUN_LABELS[idx]}
                dataKey={`run${idx}`}
                stroke={RUN_COLORS[idx].fill}
                fill={RUN_COLORS[idx].fill}
                fillOpacity={0.2}
              />
            ))}
            <Legend />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--popover))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "8px",
              }}
              formatter={(value: number) => [`${value.toFixed(1)}%`, ""]}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Survival Analysis Comparison */}
      <SurvivalComparisonSection
        runs={runs}
        runColors={RUN_COLORS}
        runLabels={RUN_LABELS}
      />

      {/* Feature Comparison with UpSet Plot */}
      <div className="bg-card rounded-xl p-6 border border-border">
        <h3 className="text-lg font-semibold mb-4">Selected Features Comparison</h3>
        
        {/* Summary cards */}
        <div className={cn(
          "grid gap-4 mb-6",
          runs.length <= 3 ? `grid-cols-1 md:grid-cols-${runs.length + 1}` : "grid-cols-2 md:grid-cols-5"
        )}>
          <div className="bg-muted/30 rounded-lg p-4 text-center">
            <p className="text-sm text-muted-foreground mb-1">Common to All</p>
            <p className="text-3xl font-bold text-accent">{featureComparison.common.length}</p>
          </div>
          {runs.map((run, idx) => (
            <div key={run.name} className={cn("rounded-lg p-4 text-center", RUN_COLORS[idx].bg)}>
              <p className="text-sm text-muted-foreground mb-1">Unique to {RUN_LABELS[idx]}</p>
              <p className={cn("text-3xl font-bold", RUN_COLORS[idx].text)}>
                {featureComparison.uniquePerRun[idx].length}
              </p>
              <p className="text-xs text-muted-foreground mt-1">
                Total: {featureComparison.totals[idx]}
              </p>
            </div>
          ))}
        </div>

        {/* UpSet-style visualization */}
        <div className="mt-6">
          <h4 className="text-sm font-medium mb-4">Feature Intersection Matrix</h4>
          <FeatureUpsetPlot
            featureSets={featureSetsForUpset}
            runColors={RUN_COLORS}
            runLabels={RUN_LABELS}
          />
        </div>
      </div>

      {/* Model Signatures (Feature Importance with CSV export) */}
      <ModelSignaturesSection
        runs={runs}
        runColors={RUN_COLORS}
        runLabels={RUN_LABELS}
      />

      {/* Feature Stability Panel */}
      <FeatureStabilityPanel
        runs={runs}
        runColors={RUN_COLORS}
        runLabels={RUN_LABELS}
      />

      {/* Feature Importance Rankings Details */}
      <FeatureDetailsSection
        runs={runs}
        runColors={RUN_COLORS}
        runLabels={RUN_LABELS}
      />

      {/* Configuration Comparison */}
      <div className="bg-card rounded-xl p-6 border border-border">
        <h3 className="text-lg font-semibold mb-4">Configuration Comparison</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-2 px-3">Setting</th>
                {runs.map((_, idx) => (
                  <th key={idx} className={cn("text-center py-2 px-3", RUN_COLORS[idx].text)}>
                    {RUN_LABELS[idx]}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                { label: "Feature Selection", key: "feature_selection_method" },
                { label: "CV Folds", key: "n_folds" },
                { label: "CV Repeats", key: "n_repeats" },
                { label: "Permutations", key: "n_permutations" },
                { label: "Max Features", key: "max_features" },
                { label: "RF Trees", key: "rf_ntree" },
              ].map(({ label, key }) => {
                const values = runs.map(run => run.data.metadata.config[key]);
                const allSame = values.every(v => v === values[0]);
                
                return (
                  <tr key={label} className="border-b border-border/50">
                    <td className="py-2 px-3 text-muted-foreground">{label}</td>
                    {runs.map((run, idx) => (
                      <td
                        key={idx}
                        className={cn(
                          "py-2 px-3 text-center font-mono",
                          !allSame && RUN_COLORS[idx].bg
                        )}
                      >
                        {String(run.data.metadata.config[key])}
                      </td>
                    ))}
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
