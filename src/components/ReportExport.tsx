import { useState } from "react";
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
import { Download, FileText, Loader2 } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import type { MLResults, PermutationMetric } from "@/types/ml-results";

interface ReportExportProps {
  data: MLResults;
}

interface ReportSections {
  summary: boolean;
  modelPerformance: boolean;
  featureImportance: boolean;
  permutationTesting: boolean;
  profileRanking: boolean;
  configuration: boolean;
}

export function ReportExport({ data }: ReportExportProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const { toast } = useToast();
  const [sections, setSections] = useState<ReportSections>({
    summary: true,
    modelPerformance: true,
    featureImportance: true,
    permutationTesting: true,
    profileRanking: true,
    configuration: true,
  });

  // Helper for safe number access
  const toFiniteNumber = (value: unknown): number | undefined => {
    if (typeof value === "number" && Number.isFinite(value)) return value;
    if (typeof value === "string") {
      const n = Number(value);
      if (Number.isFinite(n)) return n;
    }
    return undefined;
  };

  const isValidMetric = (metric: unknown): metric is PermutationMetric => {
    if (!metric || typeof metric !== 'object') return false;
    const m = metric as any;
    return toFiniteNumber(m.original) !== undefined;
  };

  const formatPercent = (val: unknown): string => {
    const n = toFiniteNumber(val);
    return n !== undefined ? (n * 100).toFixed(2) + "%" : "N/A";
  };

  const formatPValue = (val: unknown): string => {
    const n = toFiniteNumber(val);
    return n !== undefined ? n.toFixed(4) : "N/A";
  };

  const modelLabels: Record<string, string> = {
    rf: "Random Forest",
    svm: "SVM",
    xgboost: "XGBoost",
    knn: "KNN",
    mlp: "MLP",
    hard_vote: "Hard Voting",
    soft_vote: "Soft Voting",
  };

  const generateHTMLReport = () => {
    const now = new Date();
    
    let html = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ML Classification Report - ${now.toLocaleDateString()}</title>
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
    h1 { color: #0ea5e9; margin-bottom: 0.5rem; font-size: 2rem; }
    h2 { color: #1a1a2e; margin: 2rem 0 1rem; padding-bottom: 0.5rem; border-bottom: 2px solid #e5e7eb; }
    h3 { color: #374151; margin: 1.5rem 0 0.75rem; }
    .subtitle { color: #6b7280; margin-bottom: 2rem; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; margin: 1rem 0; }
    .card { background: #f8f9fa; padding: 1rem; border-radius: 8px; border: 1px solid #e5e7eb; }
    .card-title { font-size: 0.875rem; color: #6b7280; margin-bottom: 0.25rem; }
    .card-value { font-size: 1.5rem; font-weight: 700; color: #0ea5e9; }
    table { width: 100%; border-collapse: collapse; margin: 1rem 0; }
    th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #e5e7eb; }
    th { background: #f8f9fa; font-weight: 600; color: #374151; }
    tr:hover { background: #f8f9fa; }
    .highlight { color: #10b981; font-weight: 600; }
    .warning { color: #f59e0b; }
    .mono { font-family: 'SF Mono', Monaco, Consolas, monospace; }
    .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .badge-success { background: #d1fae5; color: #065f46; }
    .badge-info { background: #dbeafe; color: #1e40af; }
    .section { margin: 2rem 0; page-break-inside: avoid; }
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
    <h1>ML Classification Analysis Report</h1>
    <p class="subtitle">Generated on ${now.toLocaleDateString()} at ${now.toLocaleTimeString()}</p>
`;

    // Summary Section
    if (sections.summary) {
      const bestModel = Object.entries(data.model_performance)
        .filter(([, metrics]) => metrics?.auroc)
        .sort((a, b) => (b[1]!.auroc!.mean || 0) - (a[1]!.auroc!.mean || 0))[0];

      html += `
    <section class="section">
      <h2>Executive Summary</h2>
      <div class="grid">
        <div class="card">
          <div class="card-title">Best Model</div>
          <div class="card-value">${bestModel ? modelLabels[bestModel[0]] : "N/A"}</div>
        </div>
        <div class="card">
          <div class="card-title">Best AUROC</div>
          <div class="card-value">${bestModel ? ((bestModel[1]?.auroc?.mean || 0) * 100).toFixed(1) : "N/A"}%</div>
        </div>
        <div class="card">
          <div class="card-title">Features Selected</div>
          <div class="card-value">${data.selected_features?.length || 0}</div>
        </div>
        <div class="card">
          <div class="card-title">Permutation Tests</div>
          <div class="card-value">${data.metadata.config.n_permutations}</div>
        </div>
      </div>
    </section>
`;
    }

    // Model Performance Section
    if (sections.modelPerformance) {
      html += `
    <section class="section">
      <h2>Model Performance</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>AUROC</th>
            <th>Sensitivity</th>
            <th>Specificity</th>
            <th>F1 Score</th>
          </tr>
        </thead>
        <tbody>
`;
      Object.entries(data.model_performance)
        .filter(([, metrics]) => metrics?.auroc)
        .forEach(([model, metrics]) => {
          html += `
          <tr>
            <td><strong>${modelLabels[model] || model}</strong></td>
            <td class="mono">${metrics?.accuracy ? (metrics.accuracy.mean * 100).toFixed(1) + "%" : "N/A"}</td>
            <td class="mono highlight">${metrics?.auroc ? (metrics.auroc.mean * 100).toFixed(1) + "%" : "N/A"}</td>
            <td class="mono">${metrics?.sensitivity ? (metrics.sensitivity.mean * 100).toFixed(1) + "%" : "N/A"}</td>
            <td class="mono">${metrics?.specificity ? (metrics.specificity.mean * 100).toFixed(1) + "%" : "N/A"}</td>
            <td class="mono">${metrics?.f1_score ? (metrics.f1_score.mean * 100).toFixed(1) + "%" : "N/A"}</td>
          </tr>
`;
        });
      html += `
        </tbody>
      </table>
    </section>
`;
    }

    // Feature Importance Section
    if (sections.featureImportance && data.feature_importance?.length > 0) {
      html += `
    <section class="section">
      <h2>Top Feature Importance</h2>
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Feature</th>
            <th>Importance Score</th>
          </tr>
        </thead>
        <tbody>
`;
      data.feature_importance.slice(0, 20).forEach((feat, idx) => {
        html += `
          <tr>
            <td>${idx + 1}</td>
            <td class="mono">${feat.feature}</td>
            <td class="mono">${feat.importance.toFixed(4)}</td>
          </tr>
`;
      });
      html += `
        </tbody>
      </table>
    </section>
`;
    }

    // Permutation Testing Section
    if (sections.permutationTesting && data.permutation_testing) {
      const perm = data.permutation_testing;
      const hasOOB = isValidMetric(perm.rf_oob_error);
      const hasAUROC = isValidMetric(perm.rf_auroc);

      if (hasOOB || hasAUROC) {
        const oobPValue = hasOOB ? toFiniteNumber(perm.rf_oob_error.p_value) : undefined;
        const aurocPValue = hasAUROC ? toFiniteNumber(perm.rf_auroc.p_value) : undefined;

        html += `
    <section class="section">
      <h2>Permutation Testing Results</h2>
      <p style="margin-bottom: 1rem; color: #6b7280;">
        Permutation testing validates that model performance is significantly better than random chance.
      </p>
      <table>
        <thead>
          <tr>
            <th>Metric</th>
            <th>Original</th>
            <th>Permuted Mean</th>
            <th>Permuted SD</th>
            <th>p-value</th>
            <th>Significance</th>
          </tr>
        </thead>
        <tbody>
`;
        if (hasOOB) {
          const isSignificant = oobPValue !== undefined && oobPValue < 0.05;
          html += `
          <tr>
            <td>RF OOB Error</td>
            <td class="mono">${formatPercent(perm.rf_oob_error.original)}</td>
            <td class="mono">${formatPercent(perm.rf_oob_error.permuted_mean)}</td>
            <td class="mono">±${formatPercent(perm.rf_oob_error.permuted_sd)}</td>
            <td class="mono ${isSignificant ? 'highlight' : 'warning'}">${formatPValue(oobPValue)}</td>
            <td><span class="badge ${isSignificant ? 'badge-success' : 'badge-info'}">${isSignificant ? 'Significant' : 'Not Significant'}</span></td>
          </tr>
`;
        }
        if (hasAUROC) {
          const isSignificant = aurocPValue !== undefined && aurocPValue < 0.05;
          html += `
          <tr>
            <td>RF AUROC</td>
            <td class="mono">${formatPercent(perm.rf_auroc.original)}</td>
            <td class="mono">${formatPercent(perm.rf_auroc.permuted_mean)}</td>
            <td class="mono">±${formatPercent(perm.rf_auroc.permuted_sd)}</td>
            <td class="mono ${isSignificant ? 'highlight' : 'warning'}">${formatPValue(aurocPValue)}</td>
            <td><span class="badge ${isSignificant ? 'badge-success' : 'badge-info'}">${isSignificant ? 'Significant' : 'Not Significant'}</span></td>
          </tr>
`;
        }
        html += `
        </tbody>
      </table>
    </section>
`;
      }
    }

    // Profile Ranking Section
    if (sections.profileRanking && data.profile_ranking?.top_profiles) {
      html += `
    <section class="section">
      <h2>Top Profile Rankings</h2>
      <p style="margin-bottom: 1rem; color: #6b7280;">
        Top ${data.metadata.config.top_percent}% profiles ranked by prediction confidence.
      </p>
      <table>
        <thead>
          <tr>
            <th>Rank</th>
            <th>Sample</th>
            <th>Actual</th>
            <th>Predicted</th>
            <th>Confidence</th>
            <th>Correct</th>
          </tr>
        </thead>
        <tbody>
`;
      data.profile_ranking.top_profiles.slice(0, 20).forEach((profile) => {
        html += `
          <tr>
            <td>${profile.rank}</td>
            <td class="mono">${profile.sample_index}</td>
            <td>${profile.actual_class}</td>
            <td>${profile.predicted_class}</td>
            <td class="mono">${(profile.confidence * 100).toFixed(1)}%</td>
            <td><span class="badge ${profile.correct ? 'badge-success' : 'badge-info'}">${profile.correct ? '✓' : '✗'}</span></td>
          </tr>
`;
      });
      html += `
        </tbody>
      </table>
    </section>
`;
    }

    // Configuration Section
    if (sections.configuration) {
      html += `
    <section class="section">
      <h2>Analysis Configuration</h2>
      <div class="grid">
        <div class="card">
          <div class="card-title">Target Variable</div>
          <div style="font-weight: 600;">${data.metadata.config.target_variable}</div>
        </div>
        <div class="card">
          <div class="card-title">Cross-Validation</div>
          <div style="font-weight: 600;">${data.metadata.config.n_folds}-fold × ${data.metadata.config.n_repeats} repeats</div>
        </div>
        <div class="card">
          <div class="card-title">Feature Selection</div>
          <div style="font-weight: 600;">${data.metadata.config.feature_selection_method}</div>
        </div>
        <div class="card">
          <div class="card-title">Random Seed</div>
          <div style="font-weight: 600;">${data.metadata.config.seed}</div>
        </div>
      </div>
      <h3>Model Parameters</h3>
      <table>
        <tr><td><strong>RF Trees</strong></td><td class="mono">${data.metadata.config.rf_ntree}</td></tr>
        <tr><td><strong>XGBoost Rounds</strong></td><td class="mono">${data.metadata.config.xgb_nrounds}</td></tr>
        <tr><td><strong>Max Features</strong></td><td class="mono">${data.metadata.config.max_features}</td></tr>
        <tr><td><strong>Top Percent</strong></td><td class="mono">${data.metadata.config.top_percent}%</td></tr>
      </table>
    </section>
`;
    }

    html += `
    <footer style="margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #e5e7eb; text-align: center; color: #6b7280; font-size: 0.875rem;">
      <p>Generated by Multi-Method ML Diagnostic and Prognostic Classifier Dashboard</p>
      <p>R Version: ${data.metadata.r_version} | Analysis Date: ${new Date(data.metadata.generated_at).toLocaleDateString()}</p>
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
        a.download = `ml-classification-report-${new Date().toISOString().split("T")[0]}.html`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        toast({
          title: "Report Downloaded",
          description: "Your analysis report has been downloaded successfully.",
        });
      } else {
        // For PDF, open in new window for printing
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
      console.error("Error generating report:", error);
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
          <Download className="w-4 h-4 mr-2" />
          Export Report
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Export Analysis Report</DialogTitle>
          <DialogDescription>
            Select sections to include in your report
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
                {key.replace(/([A-Z])/g, " $1").trim()}
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
