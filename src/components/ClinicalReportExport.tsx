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
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Download, FileText, Loader2, User, Search } from "lucide-react";
import { useToast } from "@/hooks/use-toast";
import type { MLResults, ProfileRanking, PerGeneSurvival, ModelRiskScoreSurvival } from "@/types/ml-results";

interface ClinicalReportExportProps {
  data: MLResults;
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

function formatPercent(n: unknown): string {
  const v = toFiniteNumber(n);
  if (v == null) return "NA";
  return `${(v * 100).toFixed(1)}%`;
}

function normalizePerGene(raw: unknown): PerGeneSurvival[] {
  if (Array.isArray(raw)) {
    return raw.filter((g) => typeof (g as any)?.gene === "string") as PerGeneSurvival[];
  }
  if (raw && typeof raw === "object" && !Array.isArray(raw)) {
    return Object.values(raw).filter((g) => typeof (g as any)?.gene === "string") as PerGeneSurvival[];
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

export function ClinicalReportExport({ data }: ClinicalReportExportProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const { toast } = useToast();
  const [searchTerm, setSearchTerm] = useState("");
  const [selectedPatients, setSelectedPatients] = useState<string[]>([]);

  // Get all patient rankings
  const allPatients = useMemo(() => {
    const rankings = data.profile_ranking?.all_rankings || data.profile_ranking?.top_profiles || [];
    return rankings.map((p) => ({
      ...p,
      id: p.sample_id || `Sample_${p.sample_index}`,
    }));
  }, [data.profile_ranking]);

  // Filter patients by search
  const filteredPatients = useMemo(() => {
    if (!searchTerm) return allPatients;
    const term = searchTerm.toLowerCase();
    return allPatients.filter(
      (p) =>
        p.id.toLowerCase().includes(term) ||
        p.actual_class?.toLowerCase().includes(term) ||
        p.predicted_class?.toLowerCase().includes(term)
    );
  }, [allPatients, searchTerm]);

  // Survival data
  const survivalData = data.survival_analysis;
  const sortedGenes = useMemo(() => {
    if (!survivalData) return [];
    const normalized = normalizePerGene(survivalData.per_gene);
    return [...normalized]
      .filter((g) => Number.isFinite(g.cox_p))
      .sort((a, b) => a.cox_p - b.cox_p);
  }, [survivalData]);

  const modelSurvivalData = useMemo(() => {
    if (!survivalData) return [];
    return normalizeModelRiskScores(survivalData.model_risk_scores);
  }, [survivalData]);

  const togglePatient = (id: string) => {
    setSelectedPatients((prev) =>
      prev.includes(id) ? prev.filter((p) => p !== id) : [...prev, id]
    );
  };

  const selectAll = () => {
    setSelectedPatients(filteredPatients.map((p) => p.id));
  };

  const clearSelection = () => {
    setSelectedPatients([]);
  };

  const generateClinicalReport = (patients: ProfileRanking[]) => {
    const now = new Date();
    const modelLabels: Record<string, string> = {
      rf: "Random Forest",
      svm: "SVM",
      xgboost: "XGBoost",
      knn: "KNN",
      mlp: "MLP",
      hard_vote: "Hard Voting Ensemble",
      soft_vote: "Soft Voting Ensemble",
    };

    // Get best model
    const bestModel = Object.entries(data.model_performance)
      .filter(([, metrics]) => metrics?.auroc)
      .sort((a, b) => (b[1]!.auroc!.mean || 0) - (a[1]!.auroc!.mean || 0))[0];

    let html = `
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Clinical Report - ${patients.length} Patient(s) - ${now.toLocaleDateString()}</title>
  <style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { 
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; 
      line-height: 1.6; 
      color: #1a1a2e;
      background: #f8f9fa;
      padding: 2rem;
    }
    .container { max-width: 900px; margin: 0 auto; }
    .report-card { 
      background: white; 
      padding: 2rem; 
      border-radius: 12px; 
      box-shadow: 0 4px 24px rgba(0,0,0,0.1);
      margin-bottom: 2rem;
      page-break-inside: avoid;
    }
    .header { 
      display: flex; 
      justify-content: space-between; 
      align-items: flex-start;
      border-bottom: 2px solid #e5e7eb;
      padding-bottom: 1rem;
      margin-bottom: 1.5rem;
    }
    .patient-id { font-size: 1.5rem; font-weight: 700; color: #1a1a2e; }
    .timestamp { font-size: 0.875rem; color: #6b7280; }
    h2 { color: #374151; font-size: 1.125rem; margin: 1.5rem 0 0.75rem; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.5rem; }
    .grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin: 1rem 0; }
    .grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin: 1rem 0; }
    .metric-card { background: #f8f9fa; padding: 1rem; border-radius: 8px; text-align: center; }
    .metric-label { font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }
    .metric-value { font-size: 1.25rem; font-weight: 700; margin-top: 0.25rem; }
    .positive { color: #dc2626; }
    .negative { color: #16a34a; }
    .confidence-bar { height: 8px; background: #e5e7eb; border-radius: 4px; margin-top: 0.5rem; overflow: hidden; }
    .confidence-fill { height: 100%; border-radius: 4px; }
    .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.75rem; font-weight: 600; }
    .badge-correct { background: #d1fae5; color: #065f46; }
    .badge-incorrect { background: #fecaca; color: #991b1b; }
    .risk-section { background: #fef2f2; border: 1px solid #fecaca; border-radius: 8px; padding: 1rem; margin: 1rem 0; }
    .risk-title { font-weight: 600; color: #991b1b; margin-bottom: 0.5rem; }
    table { width: 100%; border-collapse: collapse; margin: 1rem 0; font-size: 0.875rem; }
    th, td { padding: 0.5rem; text-align: left; border-bottom: 1px solid #e5e7eb; }
    th { background: #f8f9fa; font-weight: 600; }
    .mono { font-family: 'SF Mono', Monaco, Consolas, monospace; }
    .footer { text-align: center; color: #6b7280; font-size: 0.75rem; padding: 1rem; border-top: 1px solid #e5e7eb; margin-top: 1rem; }
    .disclaimer { background: #fef3c7; border: 1px solid #fcd34d; border-radius: 8px; padding: 1rem; margin: 1rem 0; font-size: 0.875rem; }
    @media print {
      body { background: white; padding: 0; }
      .report-card { box-shadow: none; page-break-after: always; }
      .report-card:last-child { page-break-after: auto; }
    }
  </style>
</head>
<body>
  <div class="container">
`;

    // Generate report for each patient
    patients.forEach((patient) => {
      const patientId = (patient as any).id || patient.sample_id || `Sample_${patient.sample_index}`;
      const confidence = patient.confidence || patient.ensemble_probability || 0;
      const isCorrect = patient.correct;
      const riskScore0 = patient.risk_score_class_0;
      const riskScore1 = patient.risk_score_class_1;

      html += `
    <div class="report-card">
      <div class="header">
        <div>
          <div class="patient-id">🏥 Patient: ${patientId}</div>
          <div class="timestamp">Report generated: ${now.toLocaleDateString()} ${now.toLocaleTimeString()}</div>
        </div>
        <div>
          <span class="badge ${isCorrect ? "badge-correct" : "badge-incorrect"}">
            ${isCorrect ? "✓ Correctly Classified" : "✗ Misclassified"}
          </span>
        </div>
      </div>

      <div class="disclaimer">
        ⚠️ <strong>Disclaimer:</strong> This report is generated by a machine learning model for research purposes only. 
        Clinical decisions should be made by qualified healthcare professionals considering all available patient information.
      </div>

      <h2>📊 Classification Results</h2>
      <div class="grid">
        <div class="metric-card">
          <div class="metric-label">Actual Class</div>
          <div class="metric-value">${patient.actual_class}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Predicted Class</div>
          <div class="metric-value ${patient.predicted_class === patient.actual_class ? "negative" : "positive"}">${patient.predicted_class}</div>
        </div>
      </div>

      <div class="grid-3">
        <div class="metric-card">
          <div class="metric-label">Confidence Score</div>
          <div class="metric-value">${formatPercent(confidence)}</div>
          <div class="confidence-bar">
            <div class="confidence-fill" style="width: ${confidence * 100}%; background: ${confidence > 0.8 ? "#16a34a" : confidence > 0.6 ? "#eab308" : "#dc2626"};"></div>
          </div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Profile Rank</div>
          <div class="metric-value">#${patient.rank}</div>
        </div>
        <div class="metric-card">
          <div class="metric-label">Top Profile</div>
          <div class="metric-value">${patient.top_profile ? "Yes" : "No"}</div>
        </div>
      </div>
`;

      // Risk Scores if available
      if (riskScore0 !== undefined || riskScore1 !== undefined) {
        html += `
      <h2>⚠️ Risk Score Analysis</h2>
      <div class="risk-section">
        <div class="grid">
          <div class="metric-card" style="background: white;">
            <div class="metric-label">Class 0 Risk Score</div>
            <div class="metric-value">${riskScore0 !== undefined ? formatNumber(riskScore0, 1) : "NA"}</div>
          </div>
          <div class="metric-card" style="background: white;">
            <div class="metric-label">Class 1 Risk Score</div>
            <div class="metric-value">${riskScore1 !== undefined ? formatNumber(riskScore1, 1) : "NA"}</div>
          </div>
        </div>
        <p style="font-size: 0.875rem; color: #6b7280; margin-top: 0.5rem;">
          Risk scores range from 0-100. Higher scores indicate greater model confidence for that class.
        </p>
      </div>
`;
      }

      // Model Performance Summary
      html += `
      <h2>🤖 Model Performance Summary</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th style="text-align: right;">AUROC</th>
            <th style="text-align: right;">Accuracy</th>
            <th style="text-align: right;">Sensitivity</th>
            <th style="text-align: right;">Specificity</th>
          </tr>
        </thead>
        <tbody>
`;

      Object.entries(data.model_performance)
        .filter(([, metrics]) => metrics?.auroc)
        .forEach(([model, metrics]) => {
          const isBest = bestModel && model === bestModel[0];
          html += `
          <tr ${isBest ? 'style="background: #f0fdf4; font-weight: 600;"' : ""}>
            <td>${modelLabels[model] || model}${isBest ? " ★" : ""}</td>
            <td class="mono" style="text-align: right;">${metrics?.auroc ? formatPercent(metrics.auroc.mean) : "NA"}</td>
            <td class="mono" style="text-align: right;">${metrics?.accuracy ? formatPercent(metrics.accuracy.mean) : "NA"}</td>
            <td class="mono" style="text-align: right;">${metrics?.sensitivity ? formatPercent(metrics.sensitivity.mean) : "NA"}</td>
            <td class="mono" style="text-align: right;">${metrics?.specificity ? formatPercent(metrics.specificity.mean) : "NA"}</td>
          </tr>
`;
        });

      html += `
        </tbody>
      </table>
`;

      // Survival Analysis if available
      if (survivalData && sortedGenes.length > 0) {
        const topPrognosticGenes = sortedGenes.slice(0, 10);

        html += `
      <h2>❤️ Prognostic Markers (Top 10 Genes)</h2>
      <table>
        <thead>
          <tr>
            <th>Gene</th>
            <th style="text-align: right;">Hazard Ratio</th>
            <th style="text-align: right;">95% CI</th>
            <th style="text-align: right;">p-value</th>
            <th style="text-align: center;">Effect</th>
          </tr>
        </thead>
        <tbody>
`;

        topPrognosticGenes.forEach((gene) => {
          const isRisk = Number.isFinite(gene.cox_hr) && gene.cox_hr > 1;
          html += `
          <tr>
            <td class="mono">${gene.gene}</td>
            <td class="mono" style="text-align: right;">${formatNumber(gene.cox_hr)}</td>
            <td class="mono" style="text-align: right; color: #6b7280;">(${formatNumber(gene.cox_hr_lower, 2)} - ${formatNumber(gene.cox_hr_upper, 2)})</td>
            <td class="mono" style="text-align: right;">${formatPValue(gene.cox_p)}</td>
            <td style="text-align: center;"><span class="badge" style="background: ${isRisk ? "#fecaca" : "#d1fae5"}; color: ${isRisk ? "#991b1b" : "#065f46"};">${isRisk ? "Risk" : "Protective"}</span></td>
          </tr>
`;
        });

        html += `
        </tbody>
      </table>
`;

        // Model Risk Score Survival
        if (modelSurvivalData.length > 0) {
          html += `
      <h2>📈 Model Risk Score Prognostic Value</h2>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th style="text-align: right;">Cox HR</th>
            <th style="text-align: right;">p-value</th>
            <th style="text-align: center;">Prognostic</th>
          </tr>
        </thead>
        <tbody>
`;

          modelSurvivalData.forEach((model) => {
            const stats = model.stats || {} as any;
            const coxP = toFiniteNumber(stats.cox_p);
            const isPrognostic = coxP != null && coxP < 0.05;
            html += `
          <tr>
            <td style="font-weight: 600;">${(model.model || "").toUpperCase()}</td>
            <td class="mono" style="text-align: right;">${formatNumber(stats.cox_hr)}</td>
            <td class="mono" style="text-align: right;">${formatPValue(stats.cox_p)}</td>
            <td style="text-align: center;"><span class="badge" style="background: ${isPrognostic ? "#d1fae5" : "#f3f4f6"}; color: ${isPrognostic ? "#065f46" : "#6b7280"};">${isPrognostic ? "Yes" : "No"}</span></td>
          </tr>
`;
          });

          html += `
        </tbody>
      </table>
`;
        }
      }

      html += `
      <div class="footer">
        <p>AccelBio Multi-Method ML Classifier | Analysis: ${data.metadata.config.target_variable} | R ${data.metadata.r_version}</p>
      </div>
    </div>
`;
    });

    html += `
  </div>
</body>
</html>
`;

    return html;
  };

  const handleExport = async (format: "html" | "pdf") => {
    if (selectedPatients.length === 0) return;

    setIsGenerating(true);

    try {
      const patients = allPatients.filter((p) => selectedPatients.includes(p.id));
      const htmlContent = generateClinicalReport(patients as any);

      if (format === "html") {
        const blob = new Blob([htmlContent], { type: "text/html" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `clinical-report-${selectedPatients.length}-patients-${new Date().toISOString().split("T")[0]}.html`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);

        toast({
          title: "Report Downloaded",
          description: `Clinical report for ${selectedPatients.length} patient(s) downloaded successfully.`,
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
      console.error("Error generating clinical report:", error);
      toast({
        title: "Export Failed",
        description: "There was an error generating your report. Please try again.",
        variant: "destructive",
      });
    } finally {
      setIsGenerating(false);
    }
  };

  if (allPatients.length === 0) {
    return null;
  }

  return (
    <Dialog open={isOpen} onOpenChange={setIsOpen}>
      <DialogTrigger asChild>
        <Button variant="outline" size="sm">
          <User className="w-4 h-4 mr-2" />
          Clinical Report
        </Button>
      </DialogTrigger>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>Generate Clinical Report</DialogTitle>
          <DialogDescription>
            Select patients to include in the clinical report with survival analysis and predictions
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4 py-4">
          <div className="flex items-center gap-2">
            <Search className="w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search patients..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="flex-1"
            />
          </div>

          <div className="flex gap-2 text-sm">
            <Button variant="ghost" size="sm" onClick={selectAll}>
              Select All ({filteredPatients.length})
            </Button>
            <Button variant="ghost" size="sm" onClick={clearSelection}>
              Clear
            </Button>
            <span className="ml-auto text-muted-foreground">
              {selectedPatients.length} selected
            </span>
          </div>

          <ScrollArea className="h-64 border rounded-md">
            <div className="p-2 space-y-1">
              {filteredPatients.map((patient) => (
                <div
                  key={patient.id}
                  className={`flex items-center gap-3 p-2 rounded cursor-pointer hover:bg-muted/50 ${
                    selectedPatients.includes(patient.id) ? "bg-primary/10" : ""
                  }`}
                  onClick={() => togglePatient(patient.id)}
                >
                  <input
                    type="checkbox"
                    checked={selectedPatients.includes(patient.id)}
                    onChange={() => togglePatient(patient.id)}
                    className="rounded"
                  />
                  <div className="flex-1">
                    <span className="font-mono text-sm">{patient.id}</span>
                    <span className="text-xs text-muted-foreground ml-2">
                      {patient.predicted_class} ({formatPercent(patient.confidence)})
                    </span>
                  </div>
                  {patient.correct ? (
                    <span className="text-xs text-success">✓</span>
                  ) : (
                    <span className="text-xs text-destructive">✗</span>
                  )}
                </div>
              ))}
            </div>
          </ScrollArea>
        </div>

        {isGenerating && (
          <div className="flex flex-col items-center justify-center py-6 space-y-3">
            <div className="relative">
              <div className="w-12 h-12 border-4 border-muted rounded-full animate-spin border-t-primary" />
            </div>
            <p className="text-sm text-muted-foreground animate-pulse">
              Generating report for {selectedPatients.length} patient(s)... This may take a moment.
            </p>
          </div>
        )}

        <div className={`flex gap-2 ${isGenerating ? 'opacity-50 pointer-events-none' : ''}`}>
          <Button
            onClick={() => handleExport("html")}
            disabled={isGenerating || selectedPatients.length === 0}
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
            disabled={isGenerating || selectedPatients.length === 0}
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
