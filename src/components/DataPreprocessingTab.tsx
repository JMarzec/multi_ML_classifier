import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from "recharts";
import { Database, Users, Layers, GitBranch, AlertTriangle, CheckCircle2, FileText, Hash } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { MLResults } from "@/types/ml-results";

interface DataPreprocessingTabProps {
  data: MLResults;
}

const CLASS_COLORS = ["hsl(var(--primary))", "hsl(var(--secondary))", "hsl(var(--accent))", "hsl(var(--info))"];

export function DataPreprocessingTab({ data }: DataPreprocessingTabProps) {
  // Use preprocessing stats from R script if available
  const preprocessing = data.preprocessing;
  
  const stats = useMemo(() => {
    const rankings = data.profile_ranking?.all_rankings || [];
    const totalSamples = preprocessing?.original_samples || rankings.length;
    
    // Class distribution from preprocessing or rankings
    let classData: { name: string; value: number; percentage: string }[] = [];
    
    if (preprocessing?.class_distribution) {
      const classDistribution = preprocessing.class_distribution;
      const total = Object.values(classDistribution).reduce((a, b) => a + b, 0);
      classData = Object.entries(classDistribution).map(([cls, count]) => ({
        name: `Class ${cls}`,
        value: count,
        percentage: total > 0 ? ((count / total) * 100).toFixed(1) : "0",
      }));
    } else if (rankings.length > 0) {
      const classDistribution = rankings.reduce((acc, r) => {
        acc[r.actual_class] = (acc[r.actual_class] || 0) + 1;
        return acc;
      }, {} as Record<string, number>);
      classData = Object.entries(classDistribution).map(([cls, count]) => ({
        name: `Class ${cls}`,
        value: count,
        percentage: totalSamples > 0 ? ((count / totalSamples) * 100).toFixed(1) : "0",
      }));
    }
    
    const numFeatures = data.selected_features?.length || 0;
    const totalFeatures = preprocessing?.original_features || data.feature_importance?.length || numFeatures;
    const topFeatures = data.feature_importance?.slice(0, 10) || [];
    
    const correctPredictions = rankings.filter(r => r.correct).length;
    const incorrectPredictions = rankings.filter(r => !r.correct).length;
    
    return {
      totalSamples,
      classData,
      numSelectedFeatures: numFeatures,
      totalFeatures,
      topFeatures,
      correctPredictions,
      incorrectPredictions,
      accuracyFromRankings: rankings.length > 0 ? (correctPredictions / rankings.length) * 100 : 0,
    };
  }, [data, preprocessing]);

  const predictionData = [
    { name: "Correct", value: stats.correctPredictions, fill: "hsl(var(--accent))" },
    { name: "Incorrect", value: stats.incorrectPredictions, fill: "hsl(var(--destructive))" },
  ];

  // Calculate class imbalance ratio
  const classImbalanceRatio = useMemo(() => {
    if (stats.classData.length < 2) return null;
    const counts = stats.classData.map(c => c.value);
    const maxCount = Math.max(...counts);
    const minCount = Math.min(...counts);
    return minCount > 0 ? (maxCount / minCount).toFixed(2) : "∞";
  }, [stats.classData]);

  return (
    <div className="space-y-6">
      {/* Preprocessing Statistics from R Script */}
      {preprocessing && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <FileText className="w-5 h-5" />
              Preprocessing Statistics
              <Badge variant="outline" className="ml-2">From R Script</Badge>
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-4">
              <div className="bg-muted/30 rounded-lg p-4 text-center">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <Users className="w-4 h-4 text-primary" />
                </div>
                <p className="text-xs text-muted-foreground">Original Samples</p>
                <p className="text-2xl font-bold">{preprocessing.original_samples}</p>
              </div>
              
              <div className="bg-muted/30 rounded-lg p-4 text-center">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <Hash className="w-4 h-4 text-secondary" />
                </div>
                <p className="text-xs text-muted-foreground">Original Features</p>
                <p className="text-2xl font-bold">{preprocessing.original_features}</p>
              </div>
              
              <div className={`rounded-lg p-4 text-center ${preprocessing.missing_values > 0 ? 'bg-warning/20' : 'bg-success/20'}`}>
                <div className="flex items-center justify-center gap-2 mb-2">
                  {preprocessing.missing_values > 0 ? (
                    <AlertTriangle className="w-4 h-4 text-warning" />
                  ) : (
                    <CheckCircle2 className="w-4 h-4 text-success" />
                  )}
                </div>
                <p className="text-xs text-muted-foreground">Missing Values</p>
                <p className="text-2xl font-bold">
                  {preprocessing.missing_values}
                  <span className="text-sm font-normal text-muted-foreground ml-1">
                    ({preprocessing.missing_pct}%)
                  </span>
                </p>
              </div>
              
              <div className={`rounded-lg p-4 text-center ${preprocessing.constant_features_removed > 0 ? 'bg-warning/20' : 'bg-muted/30'}`}>
                <div className="flex items-center justify-center gap-2 mb-2">
                  <Layers className="w-4 h-4 text-accent" />
                </div>
                <p className="text-xs text-muted-foreground">Constant Features Removed</p>
                <p className="text-2xl font-bold">{preprocessing.constant_features_removed}</p>
              </div>
              
              <div className="bg-muted/30 rounded-lg p-4 text-center">
                <div className="flex items-center justify-center gap-2 mb-2">
                  <Database className="w-4 h-4 text-info" />
                </div>
                <p className="text-xs text-muted-foreground">Classes</p>
                <p className="text-2xl font-bold">{Object.keys(preprocessing.class_distribution).length}</p>
              </div>
              
              <div className={`rounded-lg p-4 text-center ${classImbalanceRatio && parseFloat(classImbalanceRatio) > 2 ? 'bg-warning/20' : 'bg-muted/30'}`}>
                <div className="flex items-center justify-center gap-2 mb-2">
                  <GitBranch className="w-4 h-4 text-warning" />
                </div>
                <p className="text-xs text-muted-foreground">Class Imbalance Ratio</p>
                <p className="text-2xl font-bold">{classImbalanceRatio || "N/A"}</p>
              </div>
            </div>

            {/* Train/Test Split Summary Card */}
            {!preprocessing.full_training_mode && preprocessing.cv_folds && (
              <div className="bg-gradient-to-r from-primary/10 via-secondary/10 to-accent/10 rounded-xl p-6 border border-primary/20">
                <h4 className="font-semibold flex items-center gap-2 mb-3">
                  <GitBranch className="w-4 h-4 text-primary" />
                  Training vs Testing Data Split Summary
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-primary/20 rounded-lg p-4 text-center">
                    <p className="text-xs text-muted-foreground mb-1">Training Data</p>
                    <p className="text-3xl font-bold text-primary">
                      {(((preprocessing.cv_folds - 1) / preprocessing.cv_folds) * 100).toFixed(0)}%
                    </p>
                    <p className="text-sm text-muted-foreground">
                      ~{preprocessing.train_samples_per_fold || Math.floor(preprocessing.original_samples * (preprocessing.cv_folds - 1) / preprocessing.cv_folds)} samples/fold
                    </p>
                  </div>
                  <div className="bg-secondary/20 rounded-lg p-4 text-center">
                    <p className="text-xs text-muted-foreground mb-1">Testing Data</p>
                    <p className="text-3xl font-bold text-secondary">
                      {((1 / preprocessing.cv_folds) * 100).toFixed(0)}%
                    </p>
                    <p className="text-sm text-muted-foreground">
                      ~{preprocessing.test_samples_per_fold || Math.ceil(preprocessing.original_samples / preprocessing.cv_folds)} samples/fold
                    </p>
                  </div>
                  <div className="bg-accent/20 rounded-lg p-4 text-center">
                    <p className="text-xs text-muted-foreground mb-1">Total CV Iterations</p>
                    <p className="text-3xl font-bold text-accent">
                      {preprocessing.cv_folds * (preprocessing.cv_repeats || 1)}
                    </p>
                    <p className="text-sm text-muted-foreground">
                      {preprocessing.cv_folds} folds × {preprocessing.cv_repeats || 1} repeats
                    </p>
                  </div>
                  <div className="bg-info/20 rounded-lg p-4 text-center">
                    <p className="text-xs text-muted-foreground mb-1">Validation Ratio</p>
                    <p className="text-3xl font-bold text-info">
                      {(preprocessing.cv_folds - 1)}:1
                    </p>
                    <p className="text-sm text-muted-foreground">
                      Train : Test per fold
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* Train/Test Split Information */}
            {preprocessing.full_training_mode ? (
              <div className="bg-warning/10 border border-warning/30 rounded-lg p-4">
                <h4 className="font-semibold text-warning flex items-center gap-2 mb-2">
                  <AlertTriangle className="w-4 h-4" />
                  Full Training Mode
                </h4>
                <p className="text-sm text-muted-foreground">
                  All {preprocessing.original_samples} samples were used for training (100% training set). 
                  No holdout test set was created. <strong>External validation is required</strong> to assess true model performance.
                </p>
                <div className="mt-3 grid grid-cols-2 gap-4">
                  <div className="bg-primary/10 rounded-lg p-3 text-center">
                    <p className="text-xs text-muted-foreground">Training Samples</p>
                    <p className="text-xl font-bold text-primary">{preprocessing.original_samples}</p>
                    <p className="text-xs text-muted-foreground">(100%)</p>
                  </div>
                  <div className="bg-muted/30 rounded-lg p-3 text-center">
                    <p className="text-xs text-muted-foreground">Test Samples</p>
                    <p className="text-xl font-bold text-muted-foreground">0</p>
                    <p className="text-xs text-muted-foreground">(External validation needed)</p>
                  </div>
                </div>
              </div>
            ) : preprocessing.cv_folds ? (
              <div className="bg-accent/10 border border-accent/30 rounded-lg p-4">
                <h4 className="font-semibold text-accent flex items-center gap-2 mb-2">
                  <GitBranch className="w-4 h-4" />
                  Cross-Validation Split ({preprocessing.cv_folds}-Fold × {preprocessing.cv_repeats || 1} Repeats)
                </h4>
                <p className="text-sm text-muted-foreground mb-3">
                  Data is split into {preprocessing.cv_folds} folds for cross-validation. 
                  Each fold serves as a test set once while the remaining folds form the training set.
                </p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-primary/10 rounded-lg p-3 text-center">
                    <p className="text-xs text-muted-foreground">Training per Fold</p>
                    <p className="text-xl font-bold text-primary">{preprocessing.train_samples_per_fold || Math.floor(preprocessing.original_samples * (preprocessing.cv_folds - 1) / preprocessing.cv_folds)}</p>
                    <p className="text-xs text-muted-foreground">
                      (~{(((preprocessing.cv_folds - 1) / preprocessing.cv_folds) * 100).toFixed(0)}%)
                    </p>
                  </div>
                  <div className="bg-secondary/10 rounded-lg p-3 text-center">
                    <p className="text-xs text-muted-foreground">Testing per Fold</p>
                    <p className="text-xl font-bold text-secondary">{preprocessing.test_samples_per_fold || Math.ceil(preprocessing.original_samples / preprocessing.cv_folds)}</p>
                    <p className="text-xs text-muted-foreground">
                      (~{((1 / preprocessing.cv_folds) * 100).toFixed(0)}%)
                    </p>
                  </div>
                  {preprocessing.train_class_distribution && (
                    <div className="bg-muted/30 rounded-lg p-3 text-center">
                      <p className="text-xs text-muted-foreground">Train Class Dist.</p>
                      <p className="text-sm font-mono">
                        {Object.entries(preprocessing.train_class_distribution).map(([cls, count]) => 
                          `${cls}:${count}`
                        ).join(' / ')}
                      </p>
                    </div>
                  )}
                  {preprocessing.test_class_distribution && (
                    <div className="bg-muted/30 rounded-lg p-3 text-center">
                      <p className="text-xs text-muted-foreground">Test Class Dist.</p>
                      <p className="text-sm font-mono">
                        {Object.entries(preprocessing.test_class_distribution).map(([cls, count]) => 
                          `${cls}:${count}`
                        ).join(' / ')}
                      </p>
                    </div>
                  )}
                </div>
              </div>
            ) : null}
          </CardContent>
        </Card>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="bg-card rounded-xl p-5 border border-border">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 rounded-lg bg-primary/20">
              <Users className="w-5 h-5 text-primary" />
            </div>
            <span className="text-sm text-muted-foreground">Total Samples</span>
          </div>
          <p className="text-3xl font-bold">{stats.totalSamples}</p>
        </div>

        <div className="bg-card rounded-xl p-5 border border-border">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 rounded-lg bg-secondary/20">
              <Layers className="w-5 h-5 text-secondary" />
            </div>
            <span className="text-sm text-muted-foreground">Selected Features</span>
          </div>
          <p className="text-3xl font-bold">
            {stats.numSelectedFeatures}
            <span className="text-lg text-muted-foreground font-normal">
              {" "}/ {stats.totalFeatures}
            </span>
          </p>
        </div>

        <div className="bg-card rounded-xl p-5 border border-border">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 rounded-lg bg-accent/20">
              <Database className="w-5 h-5 text-accent" />
            </div>
            <span className="text-sm text-muted-foreground">Classes</span>
          </div>
          <p className="text-3xl font-bold">{stats.classData.length}</p>
        </div>

        <div className="bg-card rounded-xl p-5 border border-border">
          <div className="flex items-center gap-3 mb-3">
            <div className="p-2 rounded-lg bg-info/20">
              <GitBranch className="w-5 h-5 text-info" />
            </div>
            <span className="text-sm text-muted-foreground">Feature Selection</span>
          </div>
          <p className="text-lg font-bold capitalize">
            {data.metadata.config.feature_selection_method}
          </p>
        </div>
      </div>

      {/* Charts Row */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Class Distribution */}
        <div className="bg-card rounded-xl p-6 border border-border">
          <h3 className="text-lg font-semibold mb-4">Class Distribution</h3>
          {stats.classData.length > 0 ? (
            <div className="flex items-center gap-8">
              <ResponsiveContainer width="50%" height={200}>
                <PieChart>
                  <Pie
                    data={stats.classData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {stats.classData.map((_, index) => (
                      <Cell key={`cell-${index}`} fill={CLASS_COLORS[index % CLASS_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--popover))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex-1 space-y-3">
                {stats.classData.map((item, index) => (
                  <div key={item.name} className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <div
                        className="w-3 h-3 rounded-full"
                        style={{ backgroundColor: CLASS_COLORS[index % CLASS_COLORS.length] }}
                      />
                      <span className="text-sm">{item.name}</span>
                    </div>
                    <div className="text-right">
                      <span className="font-mono font-semibold">{item.value}</span>
                      <span className="text-muted-foreground text-sm ml-2">({item.percentage}%)</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <p className="text-muted-foreground text-center py-8">No class distribution data available</p>
          )}
        </div>

        {/* Prediction Accuracy */}
        <div className="bg-card rounded-xl p-6 border border-border">
          <h3 className="text-lg font-semibold mb-4">Prediction Results</h3>
          {stats.totalSamples > 0 && (stats.correctPredictions > 0 || stats.incorrectPredictions > 0) ? (
            <div className="flex items-center gap-8">
              <ResponsiveContainer width="50%" height={200}>
                <PieChart>
                  <Pie
                    data={predictionData}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={2}
                    dataKey="value"
                  >
                    {predictionData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.fill} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--popover))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex-1 space-y-4">
                <div>
                  <p className="text-sm text-muted-foreground">Overall Accuracy</p>
                  <p className="text-3xl font-bold text-accent">
                    {stats.accuracyFromRankings.toFixed(1)}%
                  </p>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-accent">Correct</span>
                    <span className="font-mono">{stats.correctPredictions}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-destructive">Incorrect</span>
                    <span className="font-mono">{stats.incorrectPredictions}</span>
                  </div>
                </div>
              </div>
            </div>
          ) : (
            <p className="text-muted-foreground text-center py-8">No prediction data available</p>
          )}
        </div>
      </div>

      {/* Top Features */}
      <div className="bg-card rounded-xl p-6 border border-border">
        <h3 className="text-lg font-semibold mb-4">Top 10 Selected Features</h3>
        {stats.topFeatures.length > 0 ? (
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={stats.topFeatures}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis
                type="number"
                stroke="hsl(var(--muted-foreground))"
                tickFormatter={(v) => v.toFixed(3)}
              />
              <YAxis
                type="category"
                dataKey="feature"
                stroke="hsl(var(--muted-foreground))"
                tick={{ fontSize: 12 }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                }}
                formatter={(value: number) => [value.toFixed(4), "Importance"]}
              />
              <Bar
                dataKey="importance"
                fill="hsl(var(--primary))"
                radius={[0, 4, 4, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-muted-foreground text-center py-8">No feature importance data available</p>
        )}
      </div>

      {/* Configuration Summary */}
      <div className="bg-card rounded-xl p-6 border border-border">
        <h3 className="text-lg font-semibold mb-4">Analysis Configuration</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-muted/30 rounded-lg p-3">
            <p className="text-xs text-muted-foreground">Cross-Validation</p>
            <p className="font-mono font-semibold">
              {data.metadata.config.n_folds}-fold × {data.metadata.config.n_repeats} repeats
            </p>
          </div>
          <div className="bg-muted/30 rounded-lg p-3">
            <p className="text-xs text-muted-foreground">Permutations</p>
            <p className="font-mono font-semibold">{data.metadata.config.n_permutations}</p>
          </div>
          <div className="bg-muted/30 rounded-lg p-3">
            <p className="text-xs text-muted-foreground">Top Profiles</p>
            <p className="font-mono font-semibold">{data.metadata.config.top_percent}%</p>
          </div>
          <div className="bg-muted/30 rounded-lg p-3">
            <p className="text-xs text-muted-foreground">Random Seed</p>
            <p className="font-mono font-semibold">{data.metadata.config.seed}</p>
          </div>
        </div>
      </div>
    </div>
  );
}
