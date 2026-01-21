import { useMemo } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { cn } from "@/lib/utils";

interface FeatureUpsetPlotProps {
  featureSets: { name: string; features: string[] }[];
  runColors: { fill: string; text: string; bg: string }[];
  runLabels: string[];
}

interface IntersectionData {
  id: string;
  features: string[];
  count: number;
  runIndices: number[];
  label: string;
}

export function FeatureUpsetPlot({ featureSets, runColors, runLabels }: FeatureUpsetPlotProps) {
  const intersections = useMemo(() => {
    const sets = featureSets.map(s => new Set(s.features));
    const n = sets.length;
    const allFeatures = new Set<string>();
    sets.forEach(s => s.forEach(f => allFeatures.add(f)));

    const result: IntersectionData[] = [];

    // Generate all possible non-empty subsets
    for (let mask = 1; mask < (1 << n); mask++) {
      const runIndices: number[] = [];
      for (let i = 0; i < n; i++) {
        if (mask & (1 << i)) runIndices.push(i);
      }

      // Find features that are in exactly these runs
      const features = [...allFeatures].filter(f => {
        return sets.every((set, idx) => {
          const inSet = set.has(f);
          const shouldBeInSet = runIndices.includes(idx);
          return inSet === shouldBeInSet;
        });
      });

      if (features.length > 0) {
        const label = runIndices.map(i => runLabels[i]).join(" ∩ ");
        result.push({
          id: runIndices.join("-"),
          features,
          count: features.length,
          runIndices,
          label,
        });
      }
    }

    // Sort by count descending
    return result.sort((a, b) => b.count - a.count);
  }, [featureSets, runLabels]);

  const maxCount = Math.max(...intersections.map(i => i.count), 1);
  const numRuns = featureSets.length;

  const getBarColor = (runIndices: number[]) => {
    if (runIndices.length === numRuns) return "hsl(var(--accent))";
    if (runIndices.length === 1) return runColors[runIndices[0]]?.fill || "hsl(var(--muted))";
    // Mix for multiple but not all
    return "hsl(var(--primary))";
  };

  return (
    <div className="space-y-4">
      {/* Matrix dots + bar chart */}
      <div className="flex gap-4">
        {/* Run labels and dot matrix */}
        <div className="flex flex-col">
          <div className="h-[200px]" /> {/* Spacer for bar chart */}
          <div className="space-y-1 pt-2">
            {featureSets.map((set, idx) => (
              <div key={set.name} className="flex items-center gap-2 h-6">
                <div className={cn("w-3 h-3 rounded-full", runColors[idx]?.text.replace("text-", "bg-"))} />
                <span className="text-xs font-medium truncate max-w-[80px]">{runLabels[idx]}</span>
                <span className="text-xs text-muted-foreground">({set.features.length})</span>
              </div>
            ))}
          </div>
        </div>

        {/* Bar chart and dot matrix columns */}
        <div className="flex-1 overflow-x-auto">
          <div className="flex flex-col min-w-max">
            {/* Bar chart */}
            <div className="h-[200px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={intersections} margin={{ top: 10, right: 10, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="hsl(var(--border))" />
                  <XAxis dataKey="id" hide />
                  <YAxis
                    stroke="hsl(var(--muted-foreground))"
                    fontSize={10}
                    tickLine={false}
                    axisLine={false}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(var(--popover))",
                      border: "1px solid hsl(var(--border))",
                      borderRadius: "8px",
                      fontSize: "12px",
                    }}
                    formatter={(value: number, _, props) => [
                      `${value} features`,
                      props.payload.label,
                    ]}
                  />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {intersections.map((entry) => (
                      <Cell key={entry.id} fill={getBarColor(entry.runIndices)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Dot matrix */}
            <div className="flex gap-0 pt-2 border-t border-border">
              {intersections.map((intersection) => (
                <div
                  key={intersection.id}
                  className="flex flex-col items-center"
                  style={{ width: `${100 / Math.max(intersections.length, 1)}%`, minWidth: "28px" }}
                >
                  {featureSets.map((_, runIdx) => {
                    const isIncluded = intersection.runIndices.includes(runIdx);
                    return (
                      <div
                        key={runIdx}
                        className={cn(
                          "w-4 h-4 rounded-full flex items-center justify-center mb-1",
                          isIncluded
                            ? "bg-foreground"
                            : "bg-muted"
                        )}
                      >
                        {isIncluded && (
                          <div className="w-2 h-2 rounded-full bg-background" />
                        )}
                      </div>
                    );
                  })}
                  {/* Connecting line for multi-run intersections */}
                  {intersection.runIndices.length > 1 && (
                    <div
                      className="absolute w-0.5 bg-foreground"
                      style={{
                        height: `${(Math.max(...intersection.runIndices) - Math.min(...intersection.runIndices)) * 20}px`,
                        marginTop: `${Math.min(...intersection.runIndices) * 20 + 8}px`,
                      }}
                    />
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Legend / Feature lists */}
      <div className="grid gap-2 mt-4">
        {intersections.slice(0, 6).map((intersection) => (
          <div key={intersection.id} className="text-sm">
            <div className="flex items-center gap-2 mb-1">
              <div
                className="w-3 h-3 rounded"
                style={{ backgroundColor: getBarColor(intersection.runIndices) }}
              />
              <span className="font-medium">{intersection.label}</span>
              <span className="text-muted-foreground">({intersection.count} features)</span>
            </div>
            <div className="flex flex-wrap gap-1 ml-5">
              {intersection.features.slice(0, 8).map(f => (
                <span key={f} className="px-2 py-0.5 bg-muted text-xs rounded-full">
                  {f}
                </span>
              ))}
              {intersection.features.length > 8 && (
                <span className="px-2 py-0.5 bg-muted text-xs rounded-full text-muted-foreground">
                  +{intersection.features.length - 8} more
                </span>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
