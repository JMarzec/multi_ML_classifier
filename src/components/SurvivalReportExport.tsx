import { useState, useMemo } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Label } from "@/components/ui/label";
import { Checkbox } from "@/components/ui/checkbox";
import { Download, FileText, Loader2, Heart } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import type { MLResults, PerGeneSurvival, ModelRiskScoreSurvival } from "@/types/ml-results";
import { buildSingleRunROCSVG, buildSingleRunKMSVG, buildFeatureImportanceSVG, buildConfusionMatrixGridSVG } from "@/utils/chartToSvg";

interface SurvivalReportExportProps {
  data: MLResults;
}

interface ReportSections {
  overview: boolean;
  visualizations: boolean;
  kmCurves: boolean;
  forestPlot: boolean;
  geneTable: boolean;
  modelRiskScores: boolean;
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

function formatNumber(n: unknown, decimals: number = 3): string {
  const v = toFiniteNumber(n);
  if (v == null) return "NA";
  return v.toFixed(decimals);
}

// Normalize per_gene from array or object format
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

function normalizeModelRiskScores(raw: unknown): ModelRiskScoreSurvival[] {
  if (Array.isArray(raw)) {
    return raw.filter((m) => typeof (m as any)?.model === "string");
  }
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    return Object.values(raw).filter((m) => typeof (m as any)?.model === "string") as ModelRiskScoreSurvival[];
  }
  return [];
}

export function SurvivalReportExport({ data }: SurvivalReportExportProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const { toast } = useToast();
  const [sections, setSections] = useState<ReportSections>({
    overview: true,
    visualizations: true,
    kmCurves: true,
    forestPlot: true,
    geneTable: true,
    modelRiskScores: true,
  });

  const survivalData = data.survival_analysis;
  const sortedGenes = useMemo(() => {
    if (!survivalData) return [];
    const normalized = normalizePerGene(survivalData.per_gene);
    const pForSort = (p: number) => (Number.isFinite(p) ? p : Number.POSITIVE_INFINITY);
    return [...normalized].sort((a, b) => pForSort(a.cox_p) - pForSort(b.cox_p));
  }, [survivalData]);

  const modelSurvivalData = useMemo(() => {
    if (!survivalData) return [];
    return normalizeModelRiskScores(survivalData.model_risk_scores);
  }, [survivalData]);

  const significantGenes = sortedGenes.filter((g) => Number.isFinite(g.cox_p) && g.cox_p < 0.05);

  if (!survivalData) {
    return null;
  }

  const generateHTMLReport = () => {
    const now = new Date();

    let html = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Survival Analysis Report - ${now.toLocaleDateString()}</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
      line-height: 1.6; 
      color: #1a1a2e;
      background: #f8f9fa;
      padding: 2rem;
    }
    .container { max-width: 900px; margin: 0 auto; background: white; padding: 3rem; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.1); }
    h1 { color: #dc2626; margin-bottom: 0.5rem; font-size: 2rem; display: flex; align-items: center; gap: 0.5rem; }
    h2 { color: #1a1a2e; margin: 2rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #fecaca; }
    h3 { color: #374151; margin: 1.5rem 0 0.75rem; }
    .subtitle { color: #6b7280; margin-bottom: 2rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0; }
    .card { background: #fef2f2; padding: 1rem; border-radius: 8px; border: 1px solid #fecaca; }
    .card-title { font-size: 0.875rem; color: #991b1b; margin-bottom: 0.25rem; }
    .card-value { font-size: 1.5rem; font-weight: 700; color: #dc2626; }
    table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
    th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #e5e7eb; }
    th { background: #fef2f2; font-weight: 600; color: #991b1b; }
    tr:hover { background: #fef2f2; }
    .significant { background: #fef2f2; font-weight: 600; }
    .risk { color: #dc2626; }
    .protective { color: #16a34a; }
    .mono { font-family: 'SF Mono', Monaco, Consolas, monospace; }
    .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .badge-risk { background: #fecaca; color: #991b1b; }
    .badge-protective { background: #d1fae5; color: #065f46; }
    .info-box { background: #fef3c7; border: 1px solid #fcd34d; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
    .forest-row { display: flex; align-items: center; gap: 1rem; padding: 0.5rem 0; border-bottom: 1px solid #e5e7eb; }
    .forest-gene { width: 150px; font-family: monospace; font-size: 0.875rem; }
    .forest-bar-container { flex: 1; height: 20px; background: #f3f4f6; position: relative; border-radius: 4px; }
    .forest-bar { height: 100%; border-radius: 4px; }
    .forest-ref { position: absolute; left: 50%; top: 0; bottom: 0; width: 2px; background: #6b7280; }
    .forest-stats { width: 200px; font-family: monospace; font-size: 0.75rem; text-align: right; }
    @media print {
      body { background: white; padding: 0; }
      .container { box-shadow: none; padding: 0; }
      h2 { page-break-before: always; }
      h2:first-of-type { page-break-before: auto; }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>❤️ Survival Analysis Report</h1>
    <p class="subtitle">Generated on ${now.toLocaleDateString()} at ${now.toLocaleTimeString()}</p>
`;

    // Overview Section
    if (sections.overview) {
      html += `
    <section>
      <h2>Analysis Overview</h2>
      <div class="grid">
        <div class="card">
          <div class="card-title">Time Variable</div>
          <div class="card-value" style="font-size: 1.25rem;">${survivalData.time_variable}</div>
        </div>
        <div class="card">
          <div class="card-title">Event Variable</div>
          <div class="card-value" style="font-size: 1.25rem;">${survivalData.event_variable}</div>
        </div>
        <div class="card">
          <div class="card-title">Genes Analyzed</div>
          <div class="card-value">${sortedGenes.length}</div>
        </div>
        <div class="card">
          <div class="card-title">Significant (p<0.05)</div>
          <div class="card-value">${significantGenes.length}</div>
        </div>
      </div>
      
      <div class="info-box">
        <strong>Clinical Interpretation:</strong>
        <ul style="margin-top: 0.5rem; padding-left: 1.5rem;">
          <li><strong>Hazard Ratio (HR) > 1:</strong> Higher expression = worse survival (risk factor)</li>
          <li><strong>Hazard Ratio (HR) < 1:</strong> Higher expression = better survival (protective)</li>
          <li><strong>p < 0.05:</strong> Statistically significant association with survival</li>
        </ul>
      </div>
    </section>
`;
    }

    // Visualizations Section (SVG Charts)
    if (sections.visualizations) {
      const rocSvg = buildSingleRunROCSVG(data);
      const kmSvg = modelSurvivalData.length > 0 ? buildSingleRunKMSVG(data) : null;
      const hasFeatures = data.feature_importance && data.feature_importance.length > 0;
      const featureSvg = hasFeatures ? buildFeatureImportanceSVG(data.feature_importance, 15) : null;
      const confusionSvg = buildConfusionMatrixGridSVG(data);

      html += `
    <section>
      <h2>Performance Visualizations</h2>
      <h3 style="margin-top: 1rem;">ROC Curves - All Models</h3>
      <div style="text-align: center; margin: 1rem 0;">
        ${rocSvg}
      </div>

      <h3 style="margin-top: 2rem;">Confusion Matrices</h3>
      <div style="text-align: center; margin: 1rem 0;">
        ${confusionSvg}
      </div>
`;

      if (featureSvg) {
        html += `
      <h3 style="margin-top: 2rem;">Feature Importance</h3>
      <div style="text-align: center; margin: 1rem 0;">
        ${featureSvg}
      </div>
`;
      }

      if (kmSvg) {
        html += `
      <h3 style="margin-top: 2rem;">Survival Curves - Risk Stratification</h3>
      <div style="text-align: center; margin: 1rem 0;">
        ${kmSvg}
      </div>
`;
      }

      html += `
    </section>
`;
    }

    // Forest Plot Section
    if (sections.forestPlot && sortedGenes.length > 0) {
      const forestData = sortedGenes
        .filter((g) => Number.isFinite(g.cox_hr) && Number.isFinite(g.cox_p))
        .slice(0, 20);

      const maxHR = Math.max(...forestData.map((g) => g.cox_hr), 2);

      html += `
    <section>
      <h2>Hazard Ratio Forest Plot (Top 20 Genes)</h2>
      <p style="color: #6b7280; margin-bottom: 1rem;">Visual representation of hazard ratios with 95% confidence intervals. Reference line at HR=1.</p>
      
      <div style="margin-bottom: 1rem; display: flex; gap: 2rem; font-size: 0.875rem;">
        <span><span style="display: inline-block; width: 16px; height: 16px; background: #dc2626; border-radius: 4px; vertical-align: middle; margin-right: 4px;"></span> Risk Factor (HR > 1)</span>
        <span><span style="display: inline-block; width: 16px; height: 16px; background: #16a34a; border-radius: 4px; vertical-align: middle; margin-right: 4px;"></span> Protective (HR < 1)</span>
      </div>
      
      <div style="border: 1px solid #e5e7eb; border-radius: 8px; padding: 1rem;">
`;

      forestData.forEach((gene) => {
        const hrPercent = (gene.cox_hr / maxHR) * 50;
        const refPercent = (1 / maxHR) * 50;
        const isRisk = gene.cox_hr > 1;
        const color = isRisk ? "#dc2626" : "#16a34a";

        html += `
        <div class="forest-row">
          <div class="forest-gene">${gene.gene}</div>
          <div class="forest-bar-container">
            <div class="forest-ref" style="left: ${refPercent}%;"></div>
            <div class="forest-bar" style="width: ${hrPercent}%; background: ${color}; opacity: ${gene.cox_p < 0.05 ? 1 : 0.5};"></div>
          </div>
          <div class="forest-stats">
            HR: ${formatNumber(gene.cox_hr)} (${formatNumber(gene.cox_hr_lower, 2)}-${formatNumber(gene.cox_hr_upper, 2)}) p=${formatPValue(gene.cox_p)}${gene.cox_p < 0.05 ? " *" : ""}
          </div>
        </div>
`;
      });

      html += `
      </div>
    </section>
`;
    }

    // Gene Summary Table
    if (sections.geneTable && sortedGenes.length > 0) {
      html += `
    <section>
      <h2>Per-Gene Survival Statistics</h2>
      <table>
        <thead>
          <tr>
            <th>Gene</th>
            <th style="text-align: right;">Log-rank p</th>
            <th style="text-align: right;">Cox HR</th>
            <th style="text-align: right;">95% CI</th>
            <th style="text-align: right;">Cox p</th>
            <th style="text-align: center;">Effect</th>
          </tr>
        </thead>
        <tbody>
`;

      sortedGenes.slice(0, 50).forEach((gene) => {
        const isSignificant = Number.isFinite(gene.cox_p) && gene.cox_p < 0.05;
        const isRisk = Number.isFinite(gene.cox_hr) && gene.cox_hr > 1;

        html += `
          <tr class="${isSignificant ? "significant" : ""}">
            <td class="mono">${gene.gene}</td>
            <td class="mono" style="text-align: right;">${formatPValue(gene.logrank_p)}</td>
            <td class="mono" style="text-align: right; font-weight: 600;">${formatNumber(gene.cox_hr)}</td>
            <td class="mono" style="text-align: right; color: #6b7280;">(${formatNumber(gene.cox_hr_lower, 2)} - ${formatNumber(gene.cox_hr_upper, 2)})</td>
            <td class="mono ${isSignificant ? "risk" : ""}" style="text-align: right;">
              ${formatPValue(gene.cox_p)}${isSignificant ? " *" : ""}
            </td>
            <td style="text-align: center;">
              <span class="badge ${isRisk ? "badge-risk" : "badge-protective"}">${isRisk ? "Risk" : "Protective"}</span>
            </td>
          </tr>
`;
      });

      html += `
        </tbody>
      </table>
    </section>
`;
    }

    // Model Risk Scores Section
    if (sections.modelRiskScores && modelSurvivalData.length > 0) {
      html += `
    <section>
      <h2>Model-Based Risk Score Survival</h2>
      <p style="color: #6b7280; margin-bottom: 1rem;">Survival stratification based on model-predicted risk scores.</p>
      
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th style="text-align: right;">Log-rank p</th>
            <th style="text-align: right;">Cox HR</th>
            <th style="text-align: right;">95% CI</th>
            <th style="text-align: right;">Cox p</th>
            <th style="text-align: center;">Prognostic</th>
          </tr>
        </thead>
        <tbody>
`;

      modelSurvivalData.forEach((model) => {
        const stats = model.stats || {} as any;
        const coxP = toFiniteNumber(stats.cox_p);
        const isSignificant = coxP != null && coxP < 0.05;

        html += `
          <tr class="${isSignificant ? "significant" : ""}">
            <td style="font-weight: 600;">${(model.model || "").toUpperCase()}</td>
            <td class="mono" style="text-align: right;">${formatPValue(stats.logrank_p)}</td>
            <td class="mono" style="text-align: right; font-weight: 600;">${formatNumber(stats.cox_hr)}</td>
            <td class="mono" style="text-align: right; color: #6b7280;">(${formatNumber(stats.cox_hr_lower, 2)} - ${formatNumber(stats.cox_hr_upper, 2)})</td>
            <td class="mono ${isSignificant ? "risk" : ""}" style="text-align: right;">
              ${formatPValue(stats.cox_p)}${isSignificant ? " *" : ""}
            </td>
            <td style="text-align: center;">
              <span class="badge ${isSignificant ? "badge-risk" : "badge-protective"}">${isSignificant ? "Yes" : "No"}</span>
            </td>
          </tr>
`;
      });

      html += `
        </tbody>
      </table>
    </section>
`;
    }

    html += `
    <footer style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e5e7eb; text-align: center; color: #6b7280; font-size: 0.875rem;">
      <p>Generated by AccelBio Multi-Method ML Classifier Dashboard</p>
      <p>Analysis Date: ${new Date(data.metadata.generated_at).toLocaleDateString()}</p>
    </footer>
  </div>
</body>
</html>
`;

    return html;
  };

  const handleExport = async (format: "html" | "pdf") => {
    setIsGenerating(true);

    try {
      const htmlContent = generateHTMLReport();

      if (format === "html") {
        const blob = new Blob([htmlContent], { type: "text/html" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `survival-analysis-report-${new Date().toISOString().split("T")[0]}.html`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        toast({
          title: "Report Downloaded",
          description: "Your survival analysis report has been downloaded successfully.",
        });
      } else {
        const printWindow = window.open("", "_blank");
        if (printWindow) {
          printWindow.document.write(htmlContent);
          printWindow.document.close();
          printWindow.focus();
          setTimeout(() => {
            printWindow.print();
          }, 500);

          toast({
            title: "Print Dialog Opened",
            description: "Use the print dialog to save as PDF.",
          });
        }
      }

      setIsOpen(false);
    } catch (error) {
      console.error("Error generating survival report:", error);
      toast({
        title: "Export Failed",
        description: "There was an error generating your report. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <Heart className="w-4 h-4 mr-2" />
          Export Survival Report
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Export Survival Analysis Report</DialogTitle>
          <DialogDescription>
            Select sections to include in your survival analysis report
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          {Object.entries(sections).map(([key, value]) => (
            <div key={key} className="flex items-center space-x-3">
              <Checkbox
                id={key}
                checked={value}
                onCheckedChange={(checked) =>
                  setSections((prev) => ({ ...prev, [key]: !!checked }))
                }
              />
              <Label htmlFor={key} className="capitalize">
                {key === "kmCurves"
                  ? "Kaplan-Meier Curves Info"
                  : key === "visualizations"
                    ? "SVG Visualizations (ROC & KM)"
                    : key === "forestPlot"
                      ? "Hazard Ratio Forest Plot"
                      : key === "geneTable"
                        ? "Gene Summary Table"
                        : key === "modelRiskScores"
                          ? "Model Risk Scores"
                          : key.replace(/([A-Z])/g, " $1").trim()}
              </Label>
            </div>
          ))}
        </div>

        {isGenerating && (
          <div className="flex flex-col items-center justify-center py-6 space-y-3">
            <div className="relative">
              <div className="w-12 h-12 border-4 border-muted rounded-full animate-spin border-t-primary" />
            </div>
            <p className="text-sm text-muted-foreground animate-pulse">
              Generating report... This may take a moment.
            </p>
          </div>
        )}

        <div className={`flex gap-2 ${isGenerating ? 'opacity-50 pointer-events-none' : ''}`}>
          <Button
            onClick={() => handleExport("html")}
            disabled={isGenerating}
            className="flex-1"
          >
            {isGenerating ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <FileText className="w-4 h-4 mr-2" />
            )}
            Download HTML
          </Button>
          <Button
            onClick={() => handleExport("pdf")}
            disabled={isGenerating}
            variant="secondary"
            className="flex-1"
          >
            {isGenerating ? (
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
            ) : (
              <Download className="w-4 h-4 mr-2" />
            )}
            Print to PDF
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
