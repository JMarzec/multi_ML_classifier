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
import { Database, Users, Layers, GitBranch } from "lucide-react";
import type { MLResults } from "@/types/ml-results";

interface DataPreprocessingTabProps {
  data: MLResults;
}

const CLASS_COLORS = ["hsl(var(--primary))", "hsl(var(--secondary))"];

export function DataPreprocessingTab({ data }: DataPreprocessingTabProps) {
  const stats = useMemo(() => {
    // Extract statistics from the data
    const rankings = data.profile_ranking?.all_rankings || [];
    const totalSamples = rankings.length;
    
    // Class distribution from rankings
    const classDistribution = rankings.reduce((acc, r) => {
      acc[r.actual_class] = (acc[r.actual_class] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);
    
    const classData = Object.entries(classDistribution).map(([cls, count]) => ({
      name: `Class ${cls}`,
      value: count,
      percentage: totalSamples > 0 ? ((count / totalSamples) * 100).toFixed(1) : "0",
    }));
    
    // Feature statistics
    const numFeatures = data.selected_features?.length || 0;
    const totalFeatures = data.feature_importance?.length || numFeatures;
    
    // Top features summary
    const topFeatures = data.feature_importance?.slice(0, 10) || [];
    
    // Correct/incorrect predictions
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
      accuracyFromRankings: totalSamples > 0 ? (correctPredictions / totalSamples) * 100 : 0,
    };
  }, [data]);

  const predictionData = [
    { name: "Correct", value: stats.correctPredictions, fill: "hsl(var(--accent))" },
    { name: "Incorrect", value: stats.incorrectPredictions, fill: "hsl(var(--destructive))" },
  ];

  return (
    <div className="space-y-6">
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
          {stats.totalSamples > 0 ? (
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
