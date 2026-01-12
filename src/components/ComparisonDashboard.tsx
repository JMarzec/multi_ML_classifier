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
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
} from "recharts";
import { ArrowUp, ArrowDown, Minus } from "lucide-react";
import { cn } from "@/lib/utils";
import type { MLResults, ModelPerformance } from "@/types/ml-results";

interface ComparisonDashboardProps {
  runA: { name: string; data: MLResults };
  runB: { name: string; data: MLResults };
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

const METRICS = ["accuracy", "auroc", "sensitivity", "specificity", "f1_score"] as const;

export function ComparisonDashboard({ runA, runB }: ComparisonDashboardProps) {
  // Compare metrics across models
  const comparisonData = useMemo(() => {
    const models = Object.keys(MODEL_LABELS) as (keyof ModelPerformance)[];
    
    return models.map(model => {
      const metricsA = runA.data.model_performance[model];
      const metricsB = runB.data.model_performance[model];
      
      return {
        model: MODEL_LABELS[model],
        aurocA: metricsA?.auroc?.mean ? metricsA.auroc.mean * 100 : 0,
        aurocB: metricsB?.auroc?.mean ? metricsB.auroc.mean * 100 : 0,
        accuracyA: metricsA?.accuracy?.mean ? metricsA.accuracy.mean * 100 : 0,
        accuracyB: metricsB?.accuracy?.mean ? metricsB.accuracy.mean * 100 : 0,
        diff: metricsA?.auroc?.mean && metricsB?.auroc?.mean
          ? (metricsB.auroc.mean - metricsA.auroc.mean) * 100
          : 0,
      };
    }).filter(d => d.aurocA > 0 || d.aurocB > 0);
  }, [runA, runB]);

  // Best model comparison
  const bestModels = useMemo(() => {
    const getBest = (data: MLResults) => {
      let best = { model: "", auroc: 0 };
      Object.entries(data.model_performance).forEach(([model, metrics]) => {
        if (metrics?.auroc?.mean && metrics.auroc.mean > best.auroc) {
          best = { model: MODEL_LABELS[model] || model, auroc: metrics.auroc.mean };
        }
      });
      return best;
    };
    
    return {
      runA: getBest(runA.data),
      runB: getBest(runB.data),
    };
  }, [runA, runB]);

  // Radar chart data for soft_vote comparison
  const radarData = useMemo(() => {
    const metricsA = runA.data.model_performance.soft_vote;
    const metricsB = runB.data.model_performance.soft_vote;
    
    return METRICS.map(metric => ({
      metric: metric.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase()),
      runA: metricsA?.[metric]?.mean ? metricsA[metric].mean * 100 : 0,
      runB: metricsB?.[metric]?.mean ? metricsB[metric].mean * 100 : 0,
    }));
  }, [runA, runB]);

  // Feature overlap
  const featureComparison = useMemo(() => {
    const featuresA = new Set(runA.data.selected_features || []);
    const featuresB = new Set(runB.data.selected_features || []);
    
    const common = [...featuresA].filter(f => featuresB.has(f));
    const onlyA = [...featuresA].filter(f => !featuresB.has(f));
    const onlyB = [...featuresB].filter(f => !featuresA.has(f));
    
    return { common, onlyA, onlyB, totalA: featuresA.size, totalB: featuresB.size };
  }, [runA, runB]);

  return (
    <div className="space-y-6">
      {/* Header comparison */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="bg-primary/10 rounded-xl p-5 border border-primary/30">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-3 h-3 rounded-full bg-primary" />
            <span className="text-sm font-medium text-primary">Run A</span>
          </div>
          <h3 className="font-semibold truncate">{runA.name}</h3>
          <p className="text-sm text-muted-foreground">
            Generated: {new Date(runA.data.metadata.generated_at).toLocaleDateString()}
          </p>
          <div className="mt-3 pt-3 border-t border-primary/20">
            <p className="text-sm text-muted-foreground">Best Model</p>
            <p className="font-semibold">{bestModels.runA.model}</p>
            <p className="text-2xl font-bold text-primary">
              {(bestModels.runA.auroc * 100).toFixed(1)}% AUROC
            </p>
          </div>
        </div>

        <div className="bg-secondary/10 rounded-xl p-5 border border-secondary/30">
          <div className="flex items-center gap-2 mb-2">
            <div className="w-3 h-3 rounded-full bg-secondary" />
            <span className="text-sm font-medium text-secondary">Run B</span>
          </div>
          <h3 className="font-semibold truncate">{runB.name}</h3>
          <p className="text-sm text-muted-foreground">
            Generated: {new Date(runB.data.metadata.generated_at).toLocaleDateString()}
          </p>
          <div className="mt-3 pt-3 border-t border-secondary/20">
            <p className="text-sm text-muted-foreground">Best Model</p>
            <p className="font-semibold">{bestModels.runB.model}</p>
            <p className="text-2xl font-bold text-secondary">
              {(bestModels.runB.auroc * 100).toFixed(1)}% AUROC
            </p>
          </div>
        </div>
      </div>

      {/* AUROC Comparison Chart */}
      <div className="bg-card rounded-xl p-6 border border-border">
        <h3 className="text-lg font-semibold mb-4">AUROC Comparison by Model</h3>
        <ResponsiveContainer width="100%" height={350}>
          <BarChart data={comparisonData} layout="vertical" margin={{ left: 100 }}>
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
            <Bar dataKey="aurocA" name="Run A" fill="hsl(var(--primary))" radius={[0, 4, 4, 0]} />
            <Bar dataKey="aurocB" name="Run B" fill="hsl(var(--secondary))" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Difference Table */}
      <div className="bg-card rounded-xl p-6 border border-border">
        <h3 className="text-lg font-semibold mb-4">Performance Difference (Run B - Run A)</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
          {comparisonData.map(({ model, diff }) => (
            <div key={model} className="bg-muted/30 rounded-lg p-3 text-center">
              <p className="text-xs text-muted-foreground mb-1 truncate">{model}</p>
              <div className={cn(
                "flex items-center justify-center gap-1 font-mono font-bold",
                diff > 0.5 && "text-accent",
                diff < -0.5 && "text-destructive",
                Math.abs(diff) <= 0.5 && "text-muted-foreground"
              )}>
                {diff > 0.5 ? <ArrowUp className="w-4 h-4" /> : 
                 diff < -0.5 ? <ArrowDown className="w-4 h-4" /> : 
                 <Minus className="w-4 h-4" />}
                <span>{Math.abs(diff).toFixed(1)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>

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
            <Radar
              name="Run A"
              dataKey="runA"
              stroke="hsl(var(--primary))"
              fill="hsl(var(--primary))"
              fillOpacity={0.3}
            />
            <Radar
              name="Run B"
              dataKey="runB"
              stroke="hsl(var(--secondary))"
              fill="hsl(var(--secondary))"
              fillOpacity={0.3}
            />
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

      {/* Feature Comparison */}
      <div className="bg-card rounded-xl p-6 border border-border">
        <h3 className="text-lg font-semibold mb-4">Selected Features Comparison</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
          <div className="bg-muted/30 rounded-lg p-4 text-center">
            <p className="text-sm text-muted-foreground mb-1">Common Features</p>
            <p className="text-3xl font-bold text-accent">{featureComparison.common.length}</p>
          </div>
          <div className="bg-primary/10 rounded-lg p-4 text-center">
            <p className="text-sm text-muted-foreground mb-1">Only in Run A</p>
            <p className="text-3xl font-bold text-primary">{featureComparison.onlyA.length}</p>
          </div>
          <div className="bg-secondary/10 rounded-lg p-4 text-center">
            <p className="text-sm text-muted-foreground mb-1">Only in Run B</p>
            <p className="text-3xl font-bold text-secondary">{featureComparison.onlyB.length}</p>
          </div>
        </div>

        {featureComparison.common.length > 0 && (
          <div className="mt-4">
            <p className="text-sm font-medium mb-2">Common features:</p>
            <div className="flex flex-wrap gap-2">
              {featureComparison.common.slice(0, 15).map(f => (
                <span key={f} className="px-2 py-1 bg-accent/20 text-accent text-xs rounded-full">
                  {f}
                </span>
              ))}
              {featureComparison.common.length > 15 && (
                <span className="px-2 py-1 bg-muted text-muted-foreground text-xs rounded-full">
                  +{featureComparison.common.length - 15} more
                </span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Configuration Comparison */}
      <div className="bg-card rounded-xl p-6 border border-border">
        <h3 className="text-lg font-semibold mb-4">Configuration Comparison</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-2 px-3">Setting</th>
                <th className="text-center py-2 px-3 text-primary">Run A</th>
                <th className="text-center py-2 px-3 text-secondary">Run B</th>
              </tr>
            </thead>
            <tbody>
              {[
                ["Feature Selection", runA.data.metadata.config.feature_selection_method, runB.data.metadata.config.feature_selection_method],
                ["CV Folds", runA.data.metadata.config.n_folds, runB.data.metadata.config.n_folds],
                ["CV Repeats", runA.data.metadata.config.n_repeats, runB.data.metadata.config.n_repeats],
                ["Permutations", runA.data.metadata.config.n_permutations, runB.data.metadata.config.n_permutations],
                ["Max Features", runA.data.metadata.config.max_features, runB.data.metadata.config.max_features],
                ["RF Trees", runA.data.metadata.config.rf_ntree, runB.data.metadata.config.rf_ntree],
              ].map(([label, valA, valB]) => (
                <tr key={label} className="border-b border-border/50">
                  <td className="py-2 px-3 text-muted-foreground">{label}</td>
                  <td className={cn(
                    "py-2 px-3 text-center font-mono",
                    valA !== valB && "bg-primary/10"
                  )}>{String(valA)}</td>
                  <td className={cn(
                    "py-2 px-3 text-center font-mono",
                    valA !== valB && "bg-secondary/10"
                  )}>{String(valB)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
