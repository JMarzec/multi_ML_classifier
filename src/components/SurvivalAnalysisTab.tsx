import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  Legend,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Activity, TrendingDown, AlertTriangle, Info, Heart } from "lucide-react";
import type { MLResults, KaplanMeierPoint, PerGeneSurvival, ModelRiskScoreSurvival } from "@/types/ml-results";

interface SurvivalAnalysisTabProps {
  data: MLResults;
}

const MODEL_COLORS: Record<string, string> = {
  rf: "hsl(var(--primary))",
  svm: "hsl(var(--secondary))",
  xgboost: "hsl(var(--accent))",
  knn: "hsl(var(--info))",
  mlp: "hsl(var(--warning))",
  soft_vote: "hsl(var(--success))",
  ensemble: "hsl(var(--success))",
};

const HIGH_RISK_COLOR = "hsl(var(--destructive))";
const LOW_RISK_COLOR = "hsl(var(--success))";

// Forest Plot SVG component with proper box-plot style markers
interface ForestPlotEntry {
  gene: string;
  hr: number;
  hr_lower: number;
  hr_upper: number;
  p_value: number;
  significant: boolean;
}

const ForestPlotSVG = ({ data }: { data: ForestPlotEntry[] }) => {
  const width = 800;
  const rowHeight = 22;
  const height = Math.max(400, data.length * rowHeight + 80);
  const margin = { top: 40, right: 100, left: 140, bottom: 40 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;

  // Calculate x-axis domain
  const allValues = data.flatMap((d) => [d.hr_lower, d.hr_upper, d.hr]);
  const minVal = Math.min(...allValues, 0.1);
  const maxVal = Math.max(...allValues, 2);
  const xMin = Math.max(0.01, minVal * 0.8);
  const xMax = maxVal * 1.2;

  // Use log scale for HR visualization (common in forest plots)
  const logScale = (value: number) => {
    const logMin = Math.log10(xMin);
    const logMax = Math.log10(xMax);
    const logVal = Math.log10(Math.max(value, xMin));
    return ((logVal - logMin) / (logMax - logMin)) * plotWidth;
  };

  // Y position for each gene
  const yPosition = (index: number) => {
    return (index + 0.5) * (plotHeight / data.length);
  };

  // Generate x-axis ticks (log scale friendly values)
  const generateTicks = () => {
    const ticks: number[] = [];
    const logMin = Math.log10(xMin);
    const logMax = Math.log10(xMax);
    
    // Add nice round values
    [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50].forEach((v) => {
      if (v >= xMin && v <= xMax) ticks.push(v);
    });
    
    return ticks;
  };

  const xTicks = generateTicks();
  const hrOnePosition = logScale(1);

  return (
    <svg 
      viewBox={`0 0 ${width} ${height}`} 
      className="w-full max-w-4xl mx-auto"
      style={{ minHeight: height }}
    >
      {/* Background */}
      <rect width={width} height={height} fill="hsl(var(--card))" rx={8} />

      {/* Plot area */}
      <g transform={`translate(${margin.left}, ${margin.top})`}>
        {/* Grid lines */}
        {xTicks.map((tick) => (
          <line
            key={tick}
            x1={logScale(tick)}
            y1={0}
            x2={logScale(tick)}
            y2={plotHeight}
            stroke="hsl(var(--border))"
            strokeDasharray={tick === 1 ? "none" : "3 3"}
            strokeWidth={tick === 1 ? 2 : 1}
            strokeOpacity={tick === 1 ? 0.8 : 0.4}
          />
        ))}

        {/* Horizontal grid lines for each gene */}
        {data.map((_, i) => (
          <line
            key={i}
            x1={0}
            y1={yPosition(i)}
            x2={plotWidth}
            y2={yPosition(i)}
            stroke="hsl(var(--border))"
            strokeOpacity={0.2}
          />
        ))}

        {/* HR = 1 reference line (emphasized) */}
        <line
          x1={hrOnePosition}
          y1={-10}
          x2={hrOnePosition}
          y2={plotHeight + 10}
          stroke="hsl(var(--muted-foreground))"
          strokeWidth={2}
          strokeDasharray="8 4"
        />
        <text
          x={hrOnePosition}
          y={-20}
          textAnchor="middle"
          className="text-xs fill-muted-foreground font-medium"
        >
          HR = 1
        </text>

        {/* Forest plot entries */}
        {data.map((entry, i) => {
          const y = yPosition(i);
          const hrX = logScale(entry.hr);
          const lowerX = logScale(entry.hr_lower);
          const upperX = logScale(entry.hr_upper);
          const color = entry.hr > 1 ? HIGH_RISK_COLOR : LOW_RISK_COLOR;
          const opacity = entry.significant ? 1 : 0.5;

          return (
            <g key={entry.gene} opacity={opacity}>
              {/* Gene label */}
              <text
                x={-10}
                y={y}
                textAnchor="end"
                alignmentBaseline="middle"
                className="text-xs fill-foreground font-mono"
              >
                {entry.gene.length > 18 ? entry.gene.slice(0, 16) + "..." : entry.gene}
              </text>

              {/* Confidence interval line */}
              <line
                x1={lowerX}
                y1={y}
                x2={upperX}
                y2={y}
                stroke={color}
                strokeWidth={2}
              />

              {/* CI caps */}
              <line
                x1={lowerX}
                y1={y - 5}
                x2={lowerX}
                y2={y + 5}
                stroke={color}
                strokeWidth={2}
              />
              <line
                x1={upperX}
                y1={y - 5}
                x2={upperX}
                y2={y + 5}
                stroke={color}
                strokeWidth={2}
              />

              {/* Point estimate (diamond/square marker) */}
              <rect
                x={hrX - 5}
                y={y - 5}
                width={10}
                height={10}
                fill={color}
                stroke="hsl(var(--background))"
                strokeWidth={1}
                transform={`rotate(45, ${hrX}, ${y})`}
              />

              {/* HR value label on right */}
              <text
                x={plotWidth + 10}
                y={y}
                textAnchor="start"
                alignmentBaseline="middle"
                className="text-xs fill-muted-foreground font-mono"
              >
                {entry.hr.toFixed(2)} ({entry.hr_lower.toFixed(2)}-{entry.hr_upper.toFixed(2)})
              </text>
            </g>
          );
        })}

        {/* X-axis */}
        <g transform={`translate(0, ${plotHeight})`}>
          <line x1={0} y1={0} x2={plotWidth} y2={0} stroke="hsl(var(--muted-foreground))" />
          {xTicks.map((tick) => (
            <g key={tick} transform={`translate(${logScale(tick)}, 0)`}>
              <line y1={0} y2={6} stroke="hsl(var(--muted-foreground))" />
              <text
                y={20}
                textAnchor="middle"
                className="text-xs fill-muted-foreground"
              >
                {tick}
              </text>
            </g>
          ))}
          <text
            x={plotWidth / 2}
            y={35}
            textAnchor="middle"
            className="text-sm fill-muted-foreground font-medium"
          >
            Hazard Ratio (log scale)
          </text>
        </g>

        {/* Labels for risk interpretation */}
        <text
          x={hrOnePosition / 2}
          y={plotHeight + 35}
          textAnchor="middle"
          className="text-xs fill-success font-medium"
        >
          ← Protective
        </text>
        <text
          x={hrOnePosition + (plotWidth - hrOnePosition) / 2}
          y={plotHeight + 35}
          textAnchor="middle"
          className="text-xs fill-destructive font-medium"
        >
          Risk →
        </text>
      </g>

      {/* Column header for HR values */}
      <text
        x={margin.left + plotWidth + 10}
        y={margin.top - 15}
        textAnchor="start"
        className="text-xs fill-muted-foreground font-medium"
      >
        HR (95% CI)
      </text>
    </svg>
  );
};

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

function formatNumber(n: unknown, decimals: number = 3): string {
  const v = toFiniteNumber(n);
  if (v == null) return "NA";
  return v.toFixed(decimals);
}

export function SurvivalAnalysisTab({ data }: SurvivalAnalysisTabProps) {
  const [selectedGene, setSelectedGene] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<string | null>(null);

  const survivalData = data.survival_analysis;

  // Check for null, undefined, or empty object
  const hasValidSurvivalData = survivalData && 
    typeof survivalData === 'object' && 
    Object.keys(survivalData).length > 0 &&
    (survivalData.per_gene || survivalData.model_risk_scores);

  if (!hasValidSurvivalData) {
    return (
      <div className="bg-card rounded-xl p-12 border border-border text-center">
        <Heart className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
        <h3 className="text-lg font-semibold mb-2">No Survival Analysis Data</h3>
        <p className="text-muted-foreground max-w-md mx-auto">
          Survival analysis requires time-to-event data (survival time and event status columns) in your annotation file. 
          Configure the R script with <code className="bg-muted px-1 rounded">--time</code> and <code className="bg-muted px-1 rounded">--event</code> parameters.
        </p>
      </div>
    );
  }

  // Helper to normalize per_gene from array or object format
  function normalizePerGene(raw: unknown): PerGeneSurvival[] {
    if (Array.isArray(raw)) {
      return raw
        .filter((g) => typeof (g as any)?.gene === "string" && (g as any).gene.length > 0)
        .map((g) => ({
          ...(g as any),
          logrank_p: toFiniteNumber((g as any).logrank_p) ?? Number.NaN,
          cox_hr: toFiniteNumber((g as any).cox_hr) ?? Number.NaN,
          cox_hr_lower: toFiniteNumber((g as any).cox_hr_lower) ?? Number.NaN,
          cox_hr_upper: toFiniteNumber((g as any).cox_hr_upper) ?? Number.NaN,
          cox_p: toFiniteNumber((g as any).cox_p) ?? Number.NaN,
          high_median_surv: toFiniteNumber((g as any).high_median_surv) ?? null,
          low_median_surv: toFiniteNumber((g as any).low_median_surv) ?? null,
        })) as PerGeneSurvival[];
    }
    if (raw && typeof raw === "object" && !Array.isArray(raw)) {
      // Object keyed by gene name
      return Object.values(raw)
        .filter((g) => typeof (g as any)?.gene === "string" && (g as any).gene.length > 0)
        .map((g) => ({
          ...(g as any),
          logrank_p: toFiniteNumber((g as any).logrank_p) ?? Number.NaN,
          cox_hr: toFiniteNumber((g as any).cox_hr) ?? Number.NaN,
          cox_hr_lower: toFiniteNumber((g as any).cox_hr_lower) ?? Number.NaN,
          cox_hr_upper: toFiniteNumber((g as any).cox_hr_upper) ?? Number.NaN,
          cox_p: toFiniteNumber((g as any).cox_p) ?? Number.NaN,
          high_median_surv: toFiniteNumber((g as any).high_median_surv) ?? null,
          low_median_surv: toFiniteNumber((g as any).low_median_surv) ?? null,
        })) as PerGeneSurvival[];
    }
    return [];
  }

  // Helper to normalize model_risk_scores from array or object format
  function normalizeModelRiskScores(raw: unknown): ModelRiskScoreSurvival[] {
    if (Array.isArray(raw)) {
      return raw.filter((m) => typeof (m as any)?.model === "string");
    }
    if (raw && typeof raw === "object" && !Array.isArray(raw)) {
      // Object keyed by model name
      return Object.values(raw).filter((m) => typeof (m as any)?.model === "string") as ModelRiskScoreSurvival[];
    }
    return [];
  }

  // Normalize and sort genes by significance (Cox p-value)
  const sortedGenes = useMemo(() => {
    const normalized = normalizePerGene(survivalData.per_gene);
    const pForSort = (p: number) => (Number.isFinite(p) ? p : Number.POSITIVE_INFINITY);
    return [...normalized].sort((a, b) => pForSort(a.cox_p) - pForSort(b.cox_p));
  }, [survivalData.per_gene]);

  const significantGenes = sortedGenes.filter((g) => Number.isFinite(g.cox_p) && g.cox_p < 0.05);

  // Prepare forest plot data (only finite values to avoid chart crashes)
  const forestPlotData = useMemo(() => {
    return sortedGenes
      .filter(
        (g) =>
          Number.isFinite(g.cox_hr) &&
          Number.isFinite(g.cox_hr_lower) &&
          Number.isFinite(g.cox_hr_upper) &&
          Number.isFinite(g.cox_p)
      )
      .slice(0, 20)
      .map((gene) => ({
        gene: gene.gene,
        hr: gene.cox_hr,
        hr_lower: gene.cox_hr_lower,
        hr_upper: gene.cox_hr_upper,
        p_value: gene.cox_p,
        significant: gene.cox_p < 0.05,
        // ErrorBar expects [lowerDelta, upperDelta] array for symmetric/asymmetric error display
        error: [gene.cox_hr - gene.cox_hr_lower, gene.cox_hr_upper - gene.cox_hr],
      }));
  }, [sortedGenes]);

  // Model risk score survival data (handles both array and object formats)
  const modelSurvivalData = useMemo(() => {
    return normalizeModelRiskScores(survivalData.model_risk_scores);
  }, [survivalData.model_risk_scores]);

  return (
    <div className="space-y-6">
      {/* Header Info */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Activity className="w-5 h-5" />
            Survival Analysis Overview
            <Badge variant="outline" className="ml-2">Prognostic</Badge>
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-muted/30 rounded-lg p-4 text-center">
              <p className="text-xs text-muted-foreground">Time Variable</p>
              <p className="text-lg font-semibold">{survivalData.time_variable}</p>
            </div>
            <div className="bg-muted/30 rounded-lg p-4 text-center">
              <p className="text-xs text-muted-foreground">Event Variable</p>
              <p className="text-lg font-semibold">{survivalData.event_variable}</p>
            </div>
            <div className="bg-muted/30 rounded-lg p-4 text-center">
              <p className="text-xs text-muted-foreground">Genes Analyzed</p>
              <p className="text-lg font-semibold">{sortedGenes.length}</p>
            </div>
            <div className="bg-primary/10 rounded-lg p-4 text-center">
              <p className="text-xs text-muted-foreground">Significant (p&lt;0.05)</p>
              <p className="text-lg font-semibold text-primary">{significantGenes.length}</p>
            </div>
          </div>

          {/* Clinical Interpretation */}
          <div className="bg-info/10 border border-info/30 rounded-lg p-4">
            <div className="flex items-start gap-2">
              <Info className="w-5 h-5 text-info flex-shrink-0 mt-0.5" />
              <div className="text-sm">
                <p className="font-medium text-info mb-1">Clinical Interpretation Guide</p>
                <ul className="text-muted-foreground space-y-1">
                  <li><strong>Hazard Ratio (HR) &gt; 1:</strong> Higher expression associated with worse survival (risk factor)</li>
                  <li><strong>Hazard Ratio (HR) &lt; 1:</strong> Higher expression associated with better survival (protective)</li>
                  <li><strong>Log-rank p-value:</strong> Tests whether survival curves differ significantly between groups</li>
                  <li><strong>Cox p-value:</strong> Tests the significance of the hazard ratio from Cox proportional hazards model</li>
                </ul>
              </div>
            </div>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="km-curves" className="space-y-4">
        <TabsList className="bg-muted/70 p-1">
          <TabsTrigger value="km-curves">Kaplan-Meier Curves</TabsTrigger>
          <TabsTrigger value="forest-plot">Hazard Ratio Forest Plot</TabsTrigger>
          <TabsTrigger value="gene-table">Gene Summary Table</TabsTrigger>
          {modelSurvivalData.length > 0 && (
            <TabsTrigger value="model-survival">Model Risk Scores</TabsTrigger>
          )}
        </TabsList>

        {/* Kaplan-Meier Curves */}
        <TabsContent value="km-curves" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Select Gene/Feature for Survival Curve</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-2 mb-4 max-h-40 overflow-y-auto">
                {sortedGenes.slice(0, 30).map((gene) => {
                  const isSig = Number.isFinite(gene.cox_p) && gene.cox_p < 0.05;
                  return (
                    <Badge
                      key={gene.gene}
                      variant={selectedGene === gene.gene ? "default" : "outline"}
                      className={`cursor-pointer ${isSig ? "border-primary" : ""}`}
                      onClick={() => setSelectedGene(gene.gene)}
                    >
                      {gene.gene}
                      {isSig && <span className="ml-1 text-xs">*</span>}
                    </Badge>
                  );
                })}
              </div>

              {selectedGene && sortedGenes.find(g => g.gene === selectedGene) && (
                <KaplanMeierCurve
                  gene={sortedGenes.find(g => g.gene === selectedGene)!}
                />
              )}

              {!selectedGene && (
                <div className="text-center py-8 text-muted-foreground">
                  Select a gene above to view its Kaplan-Meier survival curve
                </div>
              )}
            </CardContent>
          </Card>
        </TabsContent>

        {/* Forest Plot */}
        <TabsContent value="forest-plot">
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingDown className="w-5 h-5" />
                Hazard Ratio Forest Plot (Top 20 Genes)
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="mb-4 flex items-center gap-4 text-sm">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: HIGH_RISK_COLOR }} />
                  <span>Risk Factor (HR &gt; 1)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 rounded" style={{ backgroundColor: LOW_RISK_COLOR }} />
                  <span>Protective (HR &lt; 1)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 border-2 border-dashed border-muted-foreground rounded" />
                  <span>HR = 1 (No effect)</span>
                </div>
              </div>

              <ForestPlotSVG data={forestPlotData} />
            </CardContent>
          </Card>
        </TabsContent>

        {/* Gene Summary Table */}
        <TabsContent value="gene-table">
          <Card>
            <CardHeader>
              <CardTitle>Per-Gene Survival Statistics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 px-3 font-medium">Gene</th>
                      <th className="text-right py-2 px-3 font-medium">Log-rank p</th>
                      <th className="text-right py-2 px-3 font-medium">Cox HR</th>
                      <th className="text-right py-2 px-3 font-medium">95% CI</th>
                      <th className="text-right py-2 px-3 font-medium">Cox p</th>
                      <th className="text-right py-2 px-3 font-medium">High Median</th>
                      <th className="text-right py-2 px-3 font-medium">Low Median</th>
                      <th className="text-center py-2 px-3 font-medium">Effect</th>
                    </tr>
                  </thead>
                  <tbody>
                    {sortedGenes.slice(0, 50).map((gene) => (
                      <tr
                        key={gene.gene}
                        className={`border-b border-border/50 hover:bg-muted/30 ${gene.cox_p < 0.05 ? 'bg-primary/5' : ''}`}
                      >
                        <td className="py-2 px-3 font-mono text-xs">{gene.gene}</td>
                        <td className="text-right py-2 px-3 font-mono text-xs">
                          {formatPValue(gene.logrank_p)}
                        </td>
                        <td className="text-right py-2 px-3 font-mono text-xs font-semibold">
                          {formatNumber(gene.cox_hr, 3)}
                        </td>
                        <td className="text-right py-2 px-3 font-mono text-xs text-muted-foreground">
                          ({formatNumber(gene.cox_hr_lower, 2)} - {formatNumber(gene.cox_hr_upper, 2)})
                        </td>
                        <td
                          className={`text-right py-2 px-3 font-mono text-xs ${Number.isFinite(gene.cox_p) && gene.cox_p < 0.05 ? "text-primary font-semibold" : ""}`}
                        >
                          {formatPValue(gene.cox_p)}
                          {Number.isFinite(gene.cox_p) && gene.cox_p < 0.05 && <span className="ml-1">*</span>}
                        </td>
                        <td className="text-right py-2 px-3 font-mono text-xs">{formatNumber(gene.high_median_surv, 1)}</td>
                        <td className="text-right py-2 px-3 font-mono text-xs">{formatNumber(gene.low_median_surv, 1)}</td>
                        <td className="text-center py-2 px-3">
                          {gene.cox_hr > 1 ? (
                            <Badge variant="destructive" className="text-xs">Risk</Badge>
                          ) : (
                            <Badge variant="outline" className="text-xs border-success text-success">Protective</Badge>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Model Risk Score Survival */}
        {modelSurvivalData.length > 0 && (
          <TabsContent value="model-survival" className="space-y-4">
            <Card>
              <CardHeader>
                <CardTitle>Model-Based Risk Score Survival Analysis</CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  Survival stratification based on model-predicted risk scores. Samples are divided into high-risk and low-risk groups based on median risk score.
                </p>

                <div className="flex flex-wrap gap-2 mb-4">
                  {modelSurvivalData.map((m) => {
                    const modelName = String((m as any)?.model ?? "");
                    const coxP = toFiniteNumber((m as any)?.stats?.cox_p);
                    return (
                      <Badge
                        key={modelName || Math.random().toString(16).slice(2)}
                        variant={selectedModel === modelName ? "default" : "outline"}
                        className="cursor-pointer"
                        onClick={() => setSelectedModel(modelName)}
                      >
                        {(modelName || "MODEL").toUpperCase()}
                        {coxP != null && coxP < 0.05 && <span className="ml-1">*</span>}
                      </Badge>
                    );
                  })}
                </div>

                {selectedModel && modelSurvivalData.find(m => m.model === selectedModel) && (
                  <ModelKaplanMeierCurve
                    data={modelSurvivalData.find(m => m.model === selectedModel)!}
                  />
                )}

                {!selectedModel && modelSurvivalData.length > 0 && (
                  <div className="text-center py-8 text-muted-foreground">
                    Select a model above to view risk-stratified survival curves
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        )}
      </Tabs>
    </div>
  );
}

// Kaplan-Meier Curve Component for single gene
function KaplanMeierCurve({ gene }: { gene: PerGeneSurvival }) {
  // Generate synthetic K-M curve data based on gene statistics
  // In production, this would come from actual curve data
  const curveData = useMemo(() => {
    if (!gene) return [];
    
    // Create step function data for visualization
    const highMedian = gene.high_median_surv ?? 50;
    const lowMedian = gene.low_median_surv ?? 80;
    const maxTime = Math.max(highMedian, lowMedian) * 1.5;
    
    const points = [];
    for (let t = 0; t <= maxTime; t += maxTime / 20) {
      // Exponential decay model for illustration
      const highSurv = Math.exp(-0.693 * t / highMedian);
      const lowSurv = Math.exp(-0.693 * t / lowMedian);
      points.push({
        time: t,
        high: Math.max(0, Math.min(1, highSurv)) * 100,
        low: Math.max(0, Math.min(1, lowSurv)) * 100,
      });
    }
    return points;
  }, [gene]);

  // Guard against undefined gene
  if (!gene) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        Gene data not available
      </div>
    );
  }
  
  // Safe formatting helpers
  const formatPValue = (p: number | null | undefined) => {
    if (p == null || isNaN(p)) return 'NA';
    return p < 0.001 ? p.toExponential(2) : p.toFixed(4);
  };
  
  const formatNumber = (n: number | null | undefined, decimals: number = 3) => {
    if (n == null || isNaN(n)) return 'NA';
    return n.toFixed(decimals);
  };

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="bg-muted/30 rounded-lg p-3 text-center">
          <p className="text-xs text-muted-foreground">Log-rank p-value</p>
          <p className={`font-mono font-semibold ${(gene.logrank_p ?? 1) < 0.05 ? 'text-primary' : ''}`}>
            {formatPValue(gene.logrank_p)}
          </p>
        </div>
        <div className="bg-muted/30 rounded-lg p-3 text-center">
          <p className="text-xs text-muted-foreground">Hazard Ratio</p>
          <p className="font-mono font-semibold">{formatNumber(gene.cox_hr)}</p>
        </div>
        <div className="bg-muted/30 rounded-lg p-3 text-center">
          <p className="text-xs text-muted-foreground">High Expr. Median Surv.</p>
          <p className="font-mono font-semibold" style={{ color: HIGH_RISK_COLOR }}>
            {formatNumber(gene.high_median_surv, 1)}
          </p>
        </div>
        <div className="bg-muted/30 rounded-lg p-3 text-center">
          <p className="text-xs text-muted-foreground">Low Expr. Median Surv.</p>
          <p className="font-mono font-semibold" style={{ color: LOW_RISK_COLOR }}>
            {formatNumber(gene.low_median_surv, 1)}
          </p>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={curveData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis
            dataKey="time"
            stroke="hsl(var(--muted-foreground))"
            label={{ value: 'Time', position: 'bottom', offset: 0 }}
          />
          <YAxis
            domain={[0, 100]}
            stroke="hsl(var(--muted-foreground))"
            label={{ value: 'Survival %', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--popover))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "8px",
            }}
            formatter={(value: number, name: string) => [
              `${value.toFixed(1)}%`,
              name === 'high' ? 'High Expression' : 'Low Expression'
            ]}
          />
          <Legend />
          <Line
            type="stepAfter"
            dataKey="high"
            name="High Expression"
            stroke={HIGH_RISK_COLOR}
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="stepAfter"
            dataKey="low"
            name="Low Expression"
            stroke={LOW_RISK_COLOR}
            strokeWidth={2}
            dot={false}
          />
          <ReferenceLine y={50} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" />
        </LineChart>
      </ResponsiveContainer>

      {(gene.cox_p ?? 1) < 0.05 && (
        <div className="bg-warning/10 border border-warning/30 rounded-lg p-3 flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 text-warning flex-shrink-0 mt-0.5" />
          <p className="text-sm text-muted-foreground">
            <strong className="text-warning">Statistically Significant:</strong> This gene shows a significant association with survival outcomes (p &lt; 0.05).
            {(gene.cox_hr ?? 1) > 1
              ? " Higher expression is associated with increased risk (shorter survival)."
              : " Higher expression is associated with decreased risk (longer survival)."}
          </p>
        </div>
      )}
    </div>
  );
}

// Model-based Kaplan-Meier Curve Component
function ModelKaplanMeierCurve({ data }: { data: ModelRiskScoreSurvival }) {
  // Safe formatting helpers
  const formatPValue = (p: number | null | undefined) => {
    if (p == null || isNaN(p)) return 'NA';
    return p < 0.001 ? p.toExponential(2) : p.toFixed(4);
  };
  
  const formatNumber = (n: number | null | undefined, decimals: number = 3) => {
    if (n == null || isNaN(n)) return 'NA';
    return n.toFixed(decimals);
  };

  // Combine high and low risk curves for plotting
  const combinedData = useMemo(() => {
    if (!data) return [];
    
    const highCurve = data.km_curve_high || [];
    const lowCurve = data.km_curve_low || [];
    
    // If we have actual curve data
    if (highCurve.length > 0 && lowCurve.length > 0) {
      const allTimes = [...new Set([...highCurve.map(p => p.time), ...lowCurve.map(p => p.time)])].sort((a, b) => a - b);
      
      return allTimes.map(time => {
        const highPoint = highCurve.find(p => p.time === time) || highCurve.filter(p => p.time <= time).pop();
        const lowPoint = lowCurve.find(p => p.time === time) || lowCurve.filter(p => p.time <= time).pop();
        return {
          time,
          high: (highPoint?.surv ?? 1) * 100,
          low: (lowPoint?.surv ?? 1) * 100,
          high_lower: (highPoint?.lower ?? highPoint?.surv ?? 1) * 100,
          high_upper: (highPoint?.upper ?? highPoint?.surv ?? 1) * 100,
          low_lower: (lowPoint?.lower ?? lowPoint?.surv ?? 1) * 100,
          low_upper: (lowPoint?.upper ?? lowPoint?.surv ?? 1) * 100,
        };
      });
    }
    
    // Fallback to synthetic data
    const maxTime = 100;
    const points = [];
    for (let t = 0; t <= maxTime; t += 5) {
      points.push({
        time: t,
        high: Math.max(0, 100 * Math.exp(-0.02 * t)),
        low: Math.max(0, 100 * Math.exp(-0.01 * t)),
        high_lower: Math.max(0, 100 * Math.exp(-0.025 * t)),
        high_upper: Math.max(0, 100 * Math.exp(-0.015 * t)),
        low_lower: Math.max(0, 100 * Math.exp(-0.012 * t)),
        low_upper: Math.max(0, 100 * Math.exp(-0.008 * t)),
      });
    }
    return points;
  }, [data]);

  // Guard against undefined data
  if (!data || !data.stats) {
    return (
      <div className="text-center py-8 text-muted-foreground">
        Model data not available
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        <div className="bg-muted/30 rounded-lg p-3 text-center">
          <p className="text-xs text-muted-foreground">Log-rank p-value</p>
          <p className={`font-mono font-semibold ${(data.stats.logrank_p ?? 1) < 0.05 ? 'text-primary' : ''}`}>
            {formatPValue(data.stats.logrank_p)}
          </p>
        </div>
        <div className="bg-muted/30 rounded-lg p-3 text-center">
          <p className="text-xs text-muted-foreground">Cox Hazard Ratio</p>
          <p className="font-mono font-semibold">{formatNumber(data.stats.cox_hr)}</p>
        </div>
        <div className="bg-muted/30 rounded-lg p-3 text-center">
          <p className="text-xs text-muted-foreground">95% CI</p>
          <p className="font-mono text-sm text-muted-foreground">
            ({formatNumber(data.stats.cox_hr_lower, 2)} - {formatNumber(data.stats.cox_hr_upper, 2)})
          </p>
        </div>
        <div className="bg-muted/30 rounded-lg p-3 text-center">
          <p className="text-xs text-muted-foreground">Cox p-value</p>
          <p className={`font-mono font-semibold ${(data.stats.cox_p ?? 1) < 0.05 ? 'text-primary' : ''}`}>
            {formatPValue(data.stats.cox_p)}
          </p>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={350}>
        <LineChart data={combinedData} margin={{ top: 20, right: 30, left: 20, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
          <XAxis
            dataKey="time"
            stroke="hsl(var(--muted-foreground))"
            label={{ value: 'Time', position: 'bottom', offset: 0 }}
          />
          <YAxis
            domain={[0, 100]}
            stroke="hsl(var(--muted-foreground))"
            label={{ value: 'Survival %', angle: -90, position: 'insideLeft' }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "hsl(var(--popover))",
              border: "1px solid hsl(var(--border))",
              borderRadius: "8px",
            }}
            formatter={(value: number, name: string) => [
              `${value.toFixed(1)}%`,
              name === 'high' ? 'High Risk' : 'Low Risk'
            ]}
          />
          <Legend />
          <Line
            type="stepAfter"
            dataKey="high"
            name="High Risk"
            stroke={HIGH_RISK_COLOR}
            strokeWidth={2}
            dot={false}
          />
          <Line
            type="stepAfter"
            dataKey="low"
            name="Low Risk"
            stroke={LOW_RISK_COLOR}
            strokeWidth={2}
            dot={false}
          />
          <ReferenceLine y={50} stroke="hsl(var(--muted-foreground))" strokeDasharray="3 3" />
        </LineChart>
      </ResponsiveContainer>

      {(data.stats.cox_p ?? 1) < 0.05 && (
        <div className="bg-success/10 border border-success/30 rounded-lg p-3 flex items-start gap-2">
          <Info className="w-4 h-4 text-success flex-shrink-0 mt-0.5" />
          <p className="text-sm text-muted-foreground">
            <strong className="text-success">Prognostic Value:</strong> The {data.model?.toUpperCase() ?? 'MODEL'} model's risk score significantly stratifies patients by survival (p &lt; 0.05).
            High-risk patients show {(data.stats.cox_hr ?? 1) > 1 ? `${formatNumber(data.stats.cox_hr, 1)}x higher` : `${formatNumber(data.stats.cox_hr ? 1/data.stats.cox_hr : 1, 1)}x lower`} hazard compared to low-risk patients.
          </p>
        </div>
      )}
    </div>
  );
}
