import { useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import type { FeatureImportance } from "@/types/ml-results";

interface FeatureImportanceChartProps {
  features: FeatureImportance[];
  maxFeatures?: number;
}

export function FeatureImportanceChart({ features, maxFeatures = 20 }: FeatureImportanceChartProps) {
  const data = useMemo(() => {
    const sorted = [...features]
      .sort((a, b) => b.importance - a.importance)
      .slice(0, maxFeatures);
    
    const maxImportance = Math.max(...sorted.map(f => f.importance));
    
    return sorted.map(f => ({
      ...f,
      normalized: (f.importance / maxImportance) * 100,
    }));
  }, [features, maxFeatures]);

  const getBarColor = (index: number, total: number) => {
    const hue = 199 - (index / total) * 50; // Gradient from primary to secondary
    return `hsl(${hue}, 80%, 50%)`;
  };

  return (
    <div className="bg-card rounded-xl p-6 border border-border">
      <h3 className="text-lg font-semibold mb-4">Top {maxFeatures} Features by Importance</h3>
      <div className="h-[500px]">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart 
            data={data} 
            layout="vertical" 
            margin={{ left: 150, right: 20, top: 10, bottom: 10 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
            <XAxis 
              type="number" 
              domain={[0, 100]}
              tickFormatter={(v) => `${v.toFixed(0)}%`}
              stroke="hsl(var(--muted-foreground))"
            />
            <YAxis 
              type="category" 
              dataKey="feature" 
              stroke="hsl(var(--muted-foreground))"
              tick={{ fill: "hsl(var(--foreground))", fontSize: 12 }}
              width={140}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(var(--card))",
                border: "1px solid hsl(var(--border))",
                borderRadius: "8px",
              }}
              formatter={(value: number) => [`${value.toFixed(1)}%`, "Importance"]}
            />
            <Bar dataKey="normalized" radius={[0, 4, 4, 0]}>
              {data.map((_, index) => (
                <Cell key={`cell-${index}`} fill={getBarColor(index, data.length)} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
