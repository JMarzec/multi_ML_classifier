import { Button } from "@/components/ui/button";
import { FileText, Printer } from "lucide-react";
import type { ComparisonRun } from "./ComparisonDashboard";
import { buildComparisonReportHTML } from "@/utils/comparisonReport";

interface ComparisonReportExportProps {
  runs: ComparisonRun[];
}

export function ComparisonReportExport({ runs }: ComparisonReportExportProps) {
  const handleExportHTML = () => {
    const html = buildComparisonReportHTML(runs);
    const blob = new Blob([html], { type: "text/html" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `comparison-report-${runs.length}runs-${new Date().toISOString().split("T")[0]}.html`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handlePrint = () => {
    const html = buildComparisonReportHTML(runs);
    const printWindow = window.open("", "_blank");
    if (printWindow) {
      printWindow.document.write(html);
      printWindow.document.close();
      setTimeout(() => {
        printWindow.print();
      }, 500);
    }
  };

  return (
    <div className="flex gap-2">
      <Button variant="outline" size="sm" onClick={handleExportHTML} className="gap-2">
        <FileText className="w-4 h-4" />
        Export HTML
      </Button>
      <Button variant="outline" size="sm" onClick={handlePrint} className="gap-2">
        <Printer className="w-4 h-4" />
        Print / PDF
      </Button>
    </div>
  );
}
