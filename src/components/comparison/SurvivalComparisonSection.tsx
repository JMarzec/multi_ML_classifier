import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { Heart, AlertTriangle } from "lucide-react";
import type { MLResults, ModelRiskScoreSurvival } from "@/types/ml-results";

interface SurvivalComparisonSectionProps {
  runs: { name: string; data: MLResults }[];
  runColors: { fill: string; text: string; bg: string; border: string }[];
  runLabels: string[];
}

interface ComparisonPoint {
  time: number;
  [key: string]: number;
}

function toFiniteNumber(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const n = Number(value);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

function formatPValue(p: unknown): string {
  const n = toFiniteNumber(p);
  if (n == null) return "NA";
  return n < 0.001 ? n.toExponential(2) : n.toFixed(4);
}

function normalizeModelRiskScores(raw: unknown): ModelRiskScoreSurvival[] {
  if (Array.isArray(raw)) {
    return raw.filter((m) => typeof (m as any)?.model === "string");
  }
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    return Object.values(raw).filter((m) => typeof (m as any)?.model === "string") as ModelRiskScoreSurvival[];
  }
  return [];
}

export function SurvivalComparisonSection({
  runs,
  runColors,
  runLabels,
}: SurvivalComparisonSectionProps) {
  // Check which runs have survival data
  const runsWithSurvival = useMemo(() => {
    return runs.map((run, idx) => {
      const survival = run.data.survival_analysis;
      const hasData = survival && 
        typeof survival === 'object' && 
        Object.keys(survival).length > 0 &&
        (survival.per_gene || survival.model_risk_scores);
      
      const modelData = hasData ? normalizeModelRiskScores(survival?.model_risk_scores) : [];
      const softVote = modelData.find(m => m.model === "soft_vote" || m.model === "ensemble");
      
      return {
        ...run,
        idx,
        hasData,
        modelData,
        softVote,
        stats: softVote?.stats,
      };
    });
  }, [runs]);

  const anyHasSurvival = runsWithSurvival.some(r => r.hasData);

  // Build combined KM curve data for soft_vote across runs
  const combinedKMData = useMemo(() => {
    if (!anyHasSurvival) return { high: [], low: [] };

    const timeSet = new Set<number>();
    
    runsWithSurvival.forEach(run => {
      if (run.softVote) {
        run.softVote.km_curve_high?.forEach(p => {
          if (typeof p.time === 'number') timeSet.add(p.time);
        });
        run.softVote.km_curve_low?.forEach(p => {
          if (typeof p.time === 'number') timeSet.add(p.time);
        });
      }
    });

    const times = [...timeSet].sort((a, b) => a - b);

    const buildCurve = (riskLevel: 'high' | 'low'): ComparisonPoint[] => {
      return times.map(time => {
        const point: ComparisonPoint = { time };
        
        runsWithSurvival.forEach((run, idx) => {
          if (run.softVote) {
            const curve = riskLevel === 'high' 
              ? run.softVote.km_curve_high 
              : run.softVote.km_curve_low;
            
            // Find closest point at or before this time
            const validPoints = curve?.filter(p => p.time <= time) || [];
            const closest = validPoints.length > 0 
              ? validPoints[validPoints.length - 1]
              : curve?.[0];
            
            if (closest && typeof closest.surv === 'number') {
              point[`run${idx}`] = closest.surv * 100;
            }
          }
        });
        
        return point;
      });
    };

    return {
      high: buildCurve('high'),
      low: buildCurve('low'),
    };
  }, [runsWithSurvival, anyHasSurvival]);

  if (!anyHasSurvival) {
    return (
      <div className="bg-card rounded-xl p-8 border border-border text-center">
        <Heart className="w-10 h-10 text-muted-foreground mx-auto mb-3" />
        <h3 className="text-lg font-semibold mb-2">No Survival Data Available</h3>
        <p className="text-sm text-muted-foreground max-w-md mx-auto">
          None of the uploaded runs contain survival analysis results. 
          Configure the R script with <code className="bg-muted px-1 rounded">--time</code> and{" "}
          <code className="bg-muted px-1 rounded">--event</code> parameters.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-card rounded-xl p-6 border border-border space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Heart className="w-5 h-5" />
          Survival Analysis Comparison
        </h3>
        <Badge variant="outline">Ensemble Risk Scores</Badge>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {runsWithSurvival.map((run, idx) => {
          const colors = runColors[idx];
          const stats = run.stats;
          
          return (
            <div
              key={run.name}
              className={`rounded-lg p-4 border ${colors.bg} ${colors.border}`}
            >
              <div className="flex items-center gap-2 mb-2">
                <div className={`w-2 h-2 rounded-full ${colors.text.replace("text-", "bg-")}`} />
                <span className="text-xs font-medium">{runLabels[idx]}</span>
              </div>
              
              {run.hasData && stats ? (
                <div className="space-y-1">
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">Log-rank p:</span>
                    <span className={stats.logrank_p < 0.05 ? "text-accent font-medium" : ""}>
                      {formatPValue(stats.logrank_p)}
                    </span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">Hazard Ratio:</span>
                    <span>{toFiniteNumber(stats.cox_hr)?.toFixed(2) || "NA"}</span>
                  </div>
                  <div className="flex justify-between text-xs">
                    <span className="text-muted-foreground">HR 95% CI:</span>
                    <span>
                      {toFiniteNumber(stats.cox_hr_lower)?.toFixed(2) || "?"} - {toFiniteNumber(stats.cox_hr_upper)?.toFixed(2) || "?"}
                    </span>
                  </div>
                </div>
              ) : (
                <div className="flex items-center gap-1 text-xs text-muted-foreground">
                  <AlertTriangle className="w-3 h-3" />
                  No survival data
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* KM Curves side by side: High Risk vs Low Risk */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* High Risk Comparison */}
        <div>
          <h4 className="text-sm font-medium mb-3 text-destructive">High Risk Group</h4>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={combinedKMData.high} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="time"
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={10}
                  label={{ value: "Time", position: "bottom", offset: -5, fontSize: 10 }}
                />
                <YAxis
                  domain={[0, 100]}
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={10}
                  tickFormatter={(v) => `${v}%`}
                  label={{ value: "Survival %", angle: -90, position: "insideLeft", fontSize: 10 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--popover))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                  formatter={(value: number, name: string) => {
                    const idx = parseInt(name.replace("run", ""));
                    return [`${value.toFixed(1)}%`, `${runLabels[idx]} High Risk`];
                  }}
                />
                <Legend />
                {runsWithSurvival.map((run, idx) => run.hasData && (
                  <Line
                    key={idx}
                    type="stepAfter"
                    dataKey={`run${idx}`}
                    name={runLabels[idx]}
                    stroke={runColors[idx].fill}
                    strokeWidth={2}
                    dot={false}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Low Risk Comparison */}
        <div>
          <h4 className="text-sm font-medium mb-3 text-accent">Low Risk Group</h4>
          <div className="h-[250px]">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={combinedKMData.low} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis
                  dataKey="time"
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={10}
                  label={{ value: "Time", position: "bottom", offset: -5, fontSize: 10 }}
                />
                <YAxis
                  domain={[0, 100]}
                  stroke="hsl(var(--muted-foreground))"
                  fontSize={10}
                  tickFormatter={(v) => `${v}%`}
                  label={{ value: "Survival %", angle: -90, position: "insideLeft", fontSize: 10 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "hsl(var(--popover))",
                    border: "1px solid hsl(var(--border))",
                    borderRadius: "8px",
                    fontSize: "12px",
                  }}
                  formatter={(value: number, name: string) => {
                    const idx = parseInt(name.replace("run", ""));
                    return [`${value.toFixed(1)}%`, `${runLabels[idx]} Low Risk`];
                  }}
                />
                <Legend />
                {runsWithSurvival.map((run, idx) => run.hasData && (
                  <Line
                    key={idx}
                    type="stepAfter"
                    dataKey={`run${idx}`}
                    name={runLabels[idx]}
                    stroke={runColors[idx].fill}
                    strokeWidth={2}
                    dot={false}
                    connectNulls
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Interpretation */}
      <div className="bg-muted/30 rounded-lg p-4 text-sm">
        <p className="text-muted-foreground">
          <strong>Interpretation:</strong> Comparing Kaplan-Meier survival curves for high-risk and low-risk groups 
          stratified by ensemble model predictions across runs. Lower log-rank p-values indicate better 
          discrimination between risk groups. Hazard Ratios &gt;1 suggest the high-risk group has increased 
          risk of the event.
        </p>
      </div>
    </div>
  );
}
