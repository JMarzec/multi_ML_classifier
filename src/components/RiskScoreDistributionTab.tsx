import { useMemo } from "react";
import type { ProfileRanking } from "@/types/ml-results";

interface RiskScoreDistributionTabProps {
  rankings: ProfileRanking[];
}

interface BoxplotStats {
  class: string;
  classLabel: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  mean: number;
  n: number;
  scoreType: "class_0" | "class_1";
}

export function RiskScoreDistributionTab({ rankings }: RiskScoreDistributionTabProps) {
  // Calculate box plot statistics from the rankings
  const boxplotData = useMemo(() => {
    if (!rankings || rankings.length === 0) return [];

    const classes = [...new Set(rankings.map(r => r.actual_class))].sort();
    const stats: BoxplotStats[] = [];

    // For each class, compute risk score distributions for both risk_score_class_0 and risk_score_class_1
    classes.forEach(cls => {
      const classRankings = rankings.filter(r => r.actual_class === cls);
      
      // Risk Score for Class 0 (Negative)
      const scores0 = classRankings
        .map(r => r.risk_score_class_0)
        .filter((s): s is number => s !== undefined && s !== null)
        .sort((a, b) => a - b);
      
      if (scores0.length > 0) {
        stats.push(computeBoxplotStats(scores0, cls, "class_0"));
      }

      // Risk Score for Class 1 (Positive)
      const scores1 = classRankings
        .map(r => r.risk_score_class_1)
        .filter((s): s is number => s !== undefined && s !== null)
        .sort((a, b) => a - b);
      
      if (scores1.length > 0) {
        stats.push(computeBoxplotStats(scores1, cls, "class_1"));
      }
    });

    return stats;
  }, [rankings]);

  // Group data by score type for side-by-side visualization
  const groupedByScoreType = useMemo(() => {
    const class0Stats = boxplotData.filter(s => s.scoreType === "class_0");
    const class1Stats = boxplotData.filter(s => s.scoreType === "class_1");
    return { class0Stats, class1Stats };
  }, [boxplotData]);

  if (boxplotData.length === 0) {
    return (
      <div className="bg-card rounded-xl border border-border p-8 text-center">
        <p className="text-muted-foreground">No risk score data available for visualization.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="bg-card rounded-xl border border-border p-6">
        <h3 className="text-lg font-semibold mb-2">Risk Score Distributions by Class</h3>
        <p className="text-sm text-muted-foreground mb-6">
          Box plots showing the distribution of model-predicted risk scores (0-100) for each actual class.
          Higher scores indicate higher confidence for that prediction.
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Risk Score Class 0 (Negative) Distribution */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-center border-b border-border pb-2">
              Negative Class Risk Score Distribution
            </h4>
            <BoxplotSVG 
              data={groupedByScoreType.class0Stats} 
              colorScheme="negative"
            />
          </div>

          {/* Risk Score Class 1 (Positive) Distribution */}
          <div className="space-y-4">
            <h4 className="text-sm font-semibold text-center border-b border-border pb-2">
              Positive Class Risk Score Distribution
            </h4>
            <BoxplotSVG 
              data={groupedByScoreType.class1Stats} 
              colorScheme="positive"
            />
          </div>
        </div>
      </div>

      {/* Summary Statistics Table */}
      <div className="bg-card rounded-xl border border-border p-6">
        <h3 className="text-lg font-semibold mb-4">Risk Score Summary Statistics</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left py-2 px-3 font-medium">Actual Class</th>
                <th className="text-left py-2 px-3 font-medium">Score Type</th>
                <th className="text-right py-2 px-3 font-medium">N</th>
                <th className="text-right py-2 px-3 font-medium">Min</th>
                <th className="text-right py-2 px-3 font-medium">Q1</th>
                <th className="text-right py-2 px-3 font-medium">Median</th>
                <th className="text-right py-2 px-3 font-medium">Mean</th>
                <th className="text-right py-2 px-3 font-medium">Q3</th>
                <th className="text-right py-2 px-3 font-medium">Max</th>
              </tr>
            </thead>
            <tbody>
              {boxplotData.map((stat, idx) => (
                <tr key={idx} className="border-b border-border/50 hover:bg-muted/30">
                  <td className="py-2 px-3 font-medium">{stat.classLabel}</td>
                  <td className="py-2 px-3">
                    <span className={`text-xs px-2 py-0.5 rounded ${
                      stat.scoreType === "class_0" 
                        ? "bg-primary/20 text-primary" 
                        : "bg-destructive/20 text-destructive"
                    }`}>
                      {stat.scoreType === "class_0" ? "Negative Risk" : "Positive Risk"}
                    </span>
                  </td>
                  <td className="py-2 px-3 text-right font-mono">{stat.n}</td>
                  <td className="py-2 px-3 text-right font-mono">{stat.min.toFixed(1)}</td>
                  <td className="py-2 px-3 text-right font-mono">{stat.q1.toFixed(1)}</td>
                  <td className="py-2 px-3 text-right font-mono font-semibold">{stat.median.toFixed(1)}</td>
                  <td className="py-2 px-3 text-right font-mono">{stat.mean.toFixed(1)}</td>
                  <td className="py-2 px-3 text-right font-mono">{stat.q3.toFixed(1)}</td>
                  <td className="py-2 px-3 text-right font-mono">{stat.max.toFixed(1)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Risk Score Explanation */}
      <div className="bg-primary/5 rounded-xl border border-primary/20 p-6">
        <h3 className="text-lg font-semibold mb-3">Understanding Risk Scores</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm">
          <div className="space-y-3">
            <div className="p-4 bg-primary/10 rounded-lg border border-primary/20">
              <h4 className="font-semibold text-primary mb-2">Negative Risk Score (Class 0)</h4>
              <p className="text-muted-foreground">
                Represents the model's predicted probability (0-100%) that a sample belongs to the 
                <strong className="text-foreground"> negative/control class</strong>. A high Negative Risk score 
                indicates the model is confident the sample does NOT have the condition or outcome of interest.
              </p>
            </div>
          </div>
          <div className="space-y-3">
            <div className="p-4 bg-destructive/10 rounded-lg border border-destructive/20">
              <h4 className="font-semibold text-destructive mb-2">Positive Risk Score (Class 1)</h4>
              <p className="text-muted-foreground">
                Represents the model's predicted probability (0-100%) that a sample belongs to the 
                <strong className="text-foreground"> positive/case class</strong>. A high Positive Risk score 
                indicates the model is confident the sample HAS the condition or outcome of interest.
              </p>
            </div>
          </div>
        </div>
        <div className="mt-4 p-3 bg-muted/50 rounded-lg text-xs text-muted-foreground">
          <strong>Note:</strong> For each sample, Negative Risk + Positive Risk = 100%. These scores are derived from 
          the ensemble model's probability outputs averaged across all base learners (RF, SVM, XGBoost, KNN, MLP).
        </div>
      </div>

      {/* Clinical Interpretation */}
      <div className="bg-muted/30 rounded-xl border border-border p-6">
        <h3 className="text-lg font-semibold mb-3">Clinical Interpretation</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-muted-foreground">
          <div>
            <h4 className="font-medium text-foreground mb-2">Expected Patterns</h4>
            <ul className="space-y-1 list-disc list-inside">
              <li><strong>True Negative samples:</strong> Should have high Negative Risk scores, low Positive Risk scores</li>
              <li><strong>True Positive samples:</strong> Should have high Positive Risk scores, low Negative Risk scores</li>
              <li>Clear separation between groups indicates strong model discrimination</li>
            </ul>
          </div>
          <div>
            <h4 className="font-medium text-foreground mb-2">Warning Signs</h4>
            <ul className="space-y-1 list-disc list-inside">
              <li>Overlapping distributions suggest model uncertainty</li>
              <li>Wide IQR (box height) indicates inconsistent predictions</li>
              <li>Many outliers may suggest specific sample subgroups</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
}

// Helper function to compute boxplot statistics
function computeBoxplotStats(
  sortedValues: number[], 
  cls: string, 
  scoreType: "class_0" | "class_1"
): BoxplotStats {
  const n = sortedValues.length;
  const q1Idx = Math.floor(n * 0.25);
  const q3Idx = Math.floor(n * 0.75);
  const medIdx = Math.floor(n * 0.5);
  
  const classLabels: Record<string, string> = {
    "0": "Negative",
    "1": "Positive",
  };

  return {
    class: cls,
    classLabel: classLabels[cls] || `Class ${cls}`,
    min: sortedValues[0],
    q1: sortedValues[q1Idx],
    median: n % 2 === 0 
      ? (sortedValues[medIdx - 1] + sortedValues[medIdx]) / 2 
      : sortedValues[medIdx],
    q3: sortedValues[q3Idx],
    max: sortedValues[n - 1],
    mean: sortedValues.reduce((a, b) => a + b, 0) / n,
    n,
    scoreType
  };
}

// SVG Box Plot Component
interface BoxplotSVGProps {
  data: BoxplotStats[];
  colorScheme: "negative" | "positive";
}

function BoxplotSVG({ data, colorScheme }: BoxplotSVGProps) {
  if (data.length === 0) {
    return (
      <div className="h-64 flex items-center justify-center text-muted-foreground">
        No data available
      </div>
    );
  }

  const width = 400;
  const height = 300;
  const margin = { top: 20, right: 30, bottom: 50, left: 50 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  
  // Y-axis scale (0-100 for risk scores)
  const yMin = 0;
  const yMax = 100;
  const yScale = (val: number) => plotHeight - ((val - yMin) / (yMax - yMin)) * plotHeight;
  
  // X positions for each class
  const boxWidth = Math.min(60, plotWidth / (data.length + 1));
  const spacing = plotWidth / (data.length + 1);
  
  const colors = colorScheme === "negative" 
    ? { fill: "hsl(var(--primary) / 0.3)", stroke: "hsl(var(--primary))", mean: "hsl(var(--accent))" }
    : { fill: "hsl(var(--destructive) / 0.3)", stroke: "hsl(var(--destructive))", mean: "hsl(var(--accent))" };

  return (
    <div className="flex justify-center">
      <svg 
        viewBox={`0 0 ${width} ${height}`} 
        className="w-full max-w-md"
        style={{ height: "auto" }}
      >
        {/* Y-axis */}
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          <line 
            x1={0} y1={0} 
            x2={0} y2={plotHeight} 
            stroke="currentColor" 
            strokeOpacity={0.3}
          />
          {[0, 25, 50, 75, 100].map(tick => (
            <g key={tick} transform={`translate(0, ${yScale(tick)})`}>
              <line x1={-5} x2={0} stroke="currentColor" strokeOpacity={0.5} />
              <text 
                x={-10} 
                textAnchor="end" 
                dominantBaseline="middle"
                className="text-xs fill-muted-foreground"
              >
                {tick}
              </text>
            </g>
          ))}
          <text 
            transform={`translate(-35, ${plotHeight / 2}) rotate(-90)`}
            textAnchor="middle"
            className="text-xs fill-muted-foreground font-medium"
          >
            Risk Score
          </text>
        </g>

        {/* Box plots */}
        <g transform={`translate(${margin.left}, ${margin.top})`}>
          {data.map((stat, idx) => {
            const xCenter = spacing * (idx + 1);
            const boxHalfWidth = boxWidth / 2;
            
            // Calculate whisker bounds (using 1.5 * IQR)
            const iqr = stat.q3 - stat.q1;
            const whiskerLow = Math.max(stat.min, stat.q1 - 1.5 * iqr);
            const whiskerHigh = Math.min(stat.max, stat.q3 + 1.5 * iqr);

            return (
              <g key={idx}>
                {/* Whisker line (vertical) */}
                <line
                  x1={xCenter}
                  x2={xCenter}
                  y1={yScale(whiskerHigh)}
                  y2={yScale(whiskerLow)}
                  stroke={colors.stroke}
                  strokeWidth={1.5}
                />
                
                {/* Whisker caps */}
                <line
                  x1={xCenter - boxHalfWidth / 2}
                  x2={xCenter + boxHalfWidth / 2}
                  y1={yScale(whiskerHigh)}
                  y2={yScale(whiskerHigh)}
                  stroke={colors.stroke}
                  strokeWidth={1.5}
                />
                <line
                  x1={xCenter - boxHalfWidth / 2}
                  x2={xCenter + boxHalfWidth / 2}
                  y1={yScale(whiskerLow)}
                  y2={yScale(whiskerLow)}
                  stroke={colors.stroke}
                  strokeWidth={1.5}
                />
                
                {/* Box (IQR) */}
                <rect
                  x={xCenter - boxHalfWidth}
                  y={yScale(stat.q3)}
                  width={boxWidth}
                  height={yScale(stat.q1) - yScale(stat.q3)}
                  fill={colors.fill}
                  stroke={colors.stroke}
                  strokeWidth={2}
                  rx={2}
                />
                
                {/* Median line */}
                <line
                  x1={xCenter - boxHalfWidth}
                  x2={xCenter + boxHalfWidth}
                  y1={yScale(stat.median)}
                  y2={yScale(stat.median)}
                  stroke={colors.stroke}
                  strokeWidth={2.5}
                />
                
                {/* Mean diamond */}
                <polygon
                  points={`
                    ${xCenter},${yScale(stat.mean) - 5}
                    ${xCenter + 5},${yScale(stat.mean)}
                    ${xCenter},${yScale(stat.mean) + 5}
                    ${xCenter - 5},${yScale(stat.mean)}
                  `}
                  fill={colors.mean}
                  stroke={colors.mean}
                  strokeWidth={1}
                />
                
                {/* X-axis label */}
                <text
                  x={xCenter}
                  y={plotHeight + 20}
                  textAnchor="middle"
                  className="text-xs fill-foreground font-medium"
                >
                  {stat.classLabel}
                </text>
                <text
                  x={xCenter}
                  y={plotHeight + 35}
                  textAnchor="middle"
                  className="text-xs fill-muted-foreground"
                >
                  (n={stat.n})
                </text>
              </g>
            );
          })}
        </g>

        {/* Legend */}
        <g transform={`translate(${width - margin.right - 80}, ${margin.top})`}>
          <line x1={0} x2={15} y1={5} y2={5} stroke={colors.stroke} strokeWidth={2.5} />
          <text x={20} y={8} className="text-xs fill-muted-foreground">Median</text>
          
          <polygon
            points="7,20 12,25 7,30 2,25"
            fill={colors.mean}
          />
          <text x={20} y={28} className="text-xs fill-muted-foreground">Mean</text>
        </g>
      </svg>
    </div>
  );
}
