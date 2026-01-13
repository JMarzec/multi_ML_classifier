import { useState, useMemo } from "react";
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
  Cell,
  ErrorBar,
} from "recharts";
import type { MLResults, FeatureBoxplotClassStats } from "@/types/ml-results";

interface FeatureExpressionBoxplotTabProps {
  data: MLResults;
}

const CLASS_COLORS: Record<string, string> = {
  "0": "hsl(var(--primary))",
  "1": "hsl(var(--secondary))",
};

interface FeatureBoxplotData {
  feature: string;
  by_class: Record<string, { mean: number; median: number; q25: number; q75: number; min: number; max: number }>;
}

export function FeatureExpressionBoxplotTab({ data }: FeatureExpressionBoxplotTabProps) {
  // Convert the R script's format to a usable format
  const boxplotData = useMemo<FeatureBoxplotData[]>(() => {
    const rawStats = data.feature_boxplot_stats;
    if (!rawStats) return [];
    
    return Object.entries(rawStats).map(([feature, classStats]) => {
      const byClass: Record<string, { mean: number; median: number; q25: number; q75: number; min: number; max: number }> = {};
      classStats.forEach((cs: FeatureBoxplotClassStats) => {
        byClass[cs.class] = {
          mean: cs.mean,
          median: cs.median,
          q25: cs.q1,
          q75: cs.q3,
          min: cs.min,
          max: cs.max,
        };
      });
      return { feature, by_class: byClass };
    });
  }, [data.feature_boxplot_stats]);

  const [selectedFeatureIdx, setSelectedFeatureIdx] = useState(0);

  if (!boxplotData || boxplotData.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Feature Expression Box Plots</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-sm text-muted-foreground">
            No feature expression boxplot data available. Re-run the R script with the updated exporter
            to include per-feature expression statistics for top-N features.
          </p>
        </CardContent>
      </Card>
    );
  }

  const selectedFeature = boxplotData[selectedFeatureIdx];
  const classes = Object.keys(selectedFeature.by_class);

  // Prepare chart data for the selected feature
  const chartData = classes.map((cls) => {
    const stats = selectedFeature.by_class[cls];
    return {
      class: `Class ${cls}`,
      classKey: cls,
      mean: stats.mean,
      median: stats.median,
      q25: stats.q25,
      q75: stats.q75,
      min: stats.min,
      max: stats.max,
      // For error bars: show IQR range
      iqrLower: stats.median - stats.q25,
      iqrUpper: stats.q75 - stats.median,
    };
  });

  // Summary comparison across all features
  const summaryData = boxplotData.slice(0, 20).map((feat) => {
    const classKeys = Object.keys(feat.by_class);
    const means = classKeys.map((c) => feat.by_class[c].mean);
    const diff = Math.abs(means[0] - (means[1] || means[0]));
    return {
      feature: feat.feature.length > 15 ? feat.feature.slice(0, 12) + "..." : feat.feature,
      fullName: feat.feature,
      diff,
      mean0: means[0],
      mean1: means[1] || 0,
    };
  });

  return (
    <div className="space-y-6">
      {/* Feature selector */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            Feature Expression Box Plots
            <Badge variant="outline">Top {boxplotData.length} Features</Badge>
          </CardTitle>
          <p className="text-sm text-muted-foreground">
            Expression distribution of top important features across diagnostic/prognostic classes.
          </p>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2 mb-6">
            {boxplotData.slice(0, 15).map((feat, idx) => (
              <Button
                key={feat.feature}
                variant={selectedFeatureIdx === idx ? "default" : "outline"}
                size="sm"
                onClick={() => setSelectedFeatureIdx(idx)}
                className="text-xs"
              >
                {feat.feature.length > 12 ? feat.feature.slice(0, 10) + "..." : feat.feature}
              </Button>
            ))}
            {boxplotData.length > 15 && (
              <span className="text-xs text-muted-foreground self-center">
                +{boxplotData.length - 15} more
              </span>
            )}
          </div>

          {/* Selected feature boxplot visualization */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Bar chart with error bars (IQR) */}
            <div>
              <h4 className="font-semibold mb-2 text-sm">
                {selectedFeature.feature} - Expression by Class
              </h4>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={chartData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="class" stroke="hsl(var(--muted-foreground))" />
                  <YAxis stroke="hsl(var(--muted-foreground))" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--popover))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                    }}
                    content={({ active, payload }) => {
                      if (active && payload && payload.length) {
                        const d = payload[0].payload;
                        return (
                          <div className="bg-card p-3 rounded-lg border border-border shadow-lg text-sm">
                            <p className="font-semibold mb-1">{d.class}</p>
                            <p>Mean: {d.mean.toFixed(3)}</p>
                            <p>Median: {d.median.toFixed(3)}</p>
                            <p>Q25: {d.q25.toFixed(3)}</p>
                            <p>Q75: {d.q75.toFixed(3)}</p>
                            <p>Min: {d.min.toFixed(3)}</p>
                            <p>Max: {d.max.toFixed(3)}</p>
                          </div>
                        );
                      }
                      return null;
                    }}
                  />
                  <Bar dataKey="median" name="Median" radius={[4, 4, 0, 0]}>
                    {chartData.map((entry) => (
                      <Cell
                        key={entry.classKey}
                        fill={CLASS_COLORS[entry.classKey as "0" | "1"] || "hsl(var(--accent))"}
                      />
                    ))}
                    <ErrorBar dataKey="iqrUpper" width={4} strokeWidth={2} stroke="hsl(var(--foreground))" direction="y" />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Statistics table */}
            <div>
              <h4 className="font-semibold mb-2 text-sm">Statistics Summary</h4>
              <div className="overflow-auto rounded-lg border border-border">
                <table className="w-full text-sm">
                  <thead className="bg-muted/50">
                    <tr>
                      <th className="text-left p-2">Statistic</th>
                      {classes.map((cls) => (
                        <th key={cls} className="text-right p-2">Class {cls}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {(["mean", "median", "q25", "q75", "min", "max"] as const).map((stat) => (
                      <tr key={stat} className="border-t border-border">
                        <td className="p-2 capitalize">{stat === "q25" ? "Q25" : stat === "q75" ? "Q75" : stat}</td>
                        {classes.map((cls) => (
                          <td key={cls} className="text-right p-2 font-mono">
                            {selectedFeature.by_class[cls][stat].toFixed(3)}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Feature comparison summary */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Class Separation by Feature</CardTitle>
          <p className="text-sm text-muted-foreground">
            Absolute difference in mean expression between classes for top features.
          </p>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart
              data={summaryData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 100, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
              <XAxis type="number" stroke="hsl(var(--muted-foreground))" />
              <YAxis
                type="category"
                dataKey="feature"
                stroke="hsl(var(--muted-foreground))"
                tick={{ fontSize: 11 }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(var(--popover))",
                  border: "1px solid hsl(var(--border))",
                  borderRadius: "8px",
                }}
                formatter={(value: number, name: string, props: { payload: { fullName: string } }) => [
                  value.toFixed(3),
                  `Mean Diff (${props.payload.fullName})`,
                ]}
              />
              <Bar dataKey="diff" fill="hsl(var(--primary))" radius={[0, 4, 4, 0]} name="Mean Difference" />
            </BarChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>
    </div>
  );
}
