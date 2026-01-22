import { useMemo, useState } from "react";
import { Download, Search } from "lucide-react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { toast } from "@/hooks/use-toast";
import type { MLResults } from "@/types/ml-results";

interface ModelSignaturesSectionProps {
  runs: { name: string; data: MLResults }[];
  runColors: { fill: string; text: string; bg: string; border: string }[];
  runLabels: string[];
}

interface FeatureSignature {
  feature: string;
  importances: (number | null)[];
  ranks: (number | null)[];
  presentIn: number;
}

export function ModelSignaturesSection({
  runs,
  runColors,
  runLabels,
}: ModelSignaturesSectionProps) {
  const [searchTerm, setSearchTerm] = useState("");

  // Build feature signature data
  const signatureData = useMemo(() => {
    const featureMap = new Map<string, FeatureSignature>();

    runs.forEach((run, runIdx) => {
      // Get feature importance list sorted by importance
      const importanceList = [...(run.data.feature_importance || [])].sort(
        (a, b) => b.importance - a.importance
      );

      importanceList.forEach((item, rank) => {
        if (!featureMap.has(item.feature)) {
          featureMap.set(item.feature, {
            feature: item.feature,
            importances: Array(runs.length).fill(null),
            ranks: Array(runs.length).fill(null),
            presentIn: 0,
          });
        }
        const entry = featureMap.get(item.feature)!;
        entry.importances[runIdx] = item.importance;
        entry.ranks[runIdx] = rank + 1;
        entry.presentIn++;
      });
    });

    return Array.from(featureMap.values()).sort((a, b) => {
      // Sort by presence count desc, then by avg importance
      if (b.presentIn !== a.presentIn) return b.presentIn - a.presentIn;
      const avgA = a.importances.filter((v): v is number => v !== null).reduce((s, v) => s + v, 0) / a.presentIn;
      const avgB = b.importances.filter((v): v is number => v !== null).reduce((s, v) => s + v, 0) / b.presentIn;
      return avgB - avgA;
    });
  }, [runs]);

  // Filter by search
  const filteredData = useMemo(() => {
    if (!searchTerm.trim()) return signatureData;
    const term = searchTerm.toLowerCase();
    return signatureData.filter(row => row.feature.toLowerCase().includes(term));
  }, [signatureData, searchTerm]);

  // Export to CSV
  const exportToCSV = () => {
    const headers = [
      "Feature",
      ...runLabels.slice(0, runs.length).map(l => `${l} Importance`),
      ...runLabels.slice(0, runs.length).map(l => `${l} Rank`),
      "Present In (Runs)",
    ];

    const rows = signatureData.map(row => [
      row.feature,
      ...row.importances.map(v => (v !== null ? v.toFixed(6) : "")),
      ...row.ranks.map(v => (v !== null ? v.toString() : "")),
      row.presentIn.toString(),
    ]);

    const csvContent = [
      headers.join(","),
      ...rows.map(r => r.join(",")),
    ].join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `model_signatures_comparison_${new Date().toISOString().split("T")[0]}.csv`;
    link.click();
    URL.revokeObjectURL(url);

    toast({
      title: "CSV Exported",
      description: `Exported ${signatureData.length} features across ${runs.length} runs.`,
    });
  };

  if (signatureData.length === 0) {
    return (
      <div className="bg-card rounded-xl p-6 border border-border">
        <h3 className="text-lg font-semibold mb-4">Model Signatures</h3>
        <p className="text-muted-foreground text-center py-8">
          No feature importance data available.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-card rounded-xl p-6 border border-border">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">Model Signatures (Feature Importance)</h3>
        <Button variant="outline" size="sm" onClick={exportToCSV}>
          <Download className="w-4 h-4 mr-2" />
          Export CSV
        </Button>
      </div>

      <p className="text-sm text-muted-foreground mb-4">
        This table shows all features with their importance scores and rankings across runs. 
        Higher importance indicates greater contribution to model predictions.
      </p>

      {/* Search */}
      <div className="relative mb-4 max-w-sm">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
        <Input
          placeholder="Search features..."
          value={searchTerm}
          onChange={e => setSearchTerm(e.target.value)}
          className="pl-9"
        />
      </div>

      {/* Summary badges */}
      <div className="flex flex-wrap gap-2 mb-4">
        <Badge variant="secondary">
          Total features: {signatureData.length}
        </Badge>
        <Badge variant="outline" className="text-accent border-accent">
          Common to all: {signatureData.filter(f => f.presentIn === runs.length).length}
        </Badge>
        {runs.map((_, idx) => (
          <Badge 
            key={idx} 
            variant="outline" 
            className={cn(runColors[idx].text, runColors[idx].border)}
          >
            {runLabels[idx]}: {signatureData.filter(f => f.importances[idx] !== null).length} features
          </Badge>
        ))}
      </div>

      {/* Table */}
      <div className="overflow-x-auto max-h-[500px] overflow-y-auto border rounded-lg">
        <Table>
          <TableHeader className="sticky top-0 bg-background z-10">
            <TableRow>
              <TableHead className="min-w-[150px]">Feature</TableHead>
              {runs.map((_, idx) => (
                <TableHead key={idx} className={cn("text-center min-w-[120px]", runColors[idx].text)}>
                  {runLabels[idx]}
                  <div className="text-[10px] font-normal text-muted-foreground">Imp. / Rank</div>
                </TableHead>
              ))}
              <TableHead className="text-center">Present</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredData.slice(0, 100).map(row => (
              <TableRow key={row.feature}>
                <TableCell className="font-mono text-sm">{row.feature}</TableCell>
                {runs.map((_, idx) => {
                  const imp = row.importances[idx];
                  const rank = row.ranks[idx];
                  
                  return (
                    <TableCell key={idx} className="text-center">
                      {imp !== null ? (
                        <div className="flex flex-col items-center gap-0.5">
                          <span className="font-mono text-sm">
                            {imp < 0.001 ? imp.toExponential(2) : imp.toFixed(4)}
                          </span>
                          <Badge 
                            variant="outline" 
                            className={cn("text-[10px] px-1 py-0", runColors[idx].bg, runColors[idx].border)}
                          >
                            #{rank}
                          </Badge>
                        </div>
                      ) : (
                        <span className="text-muted-foreground">—</span>
                      )}
                    </TableCell>
                  );
                })}
                <TableCell className="text-center">
                  <Badge 
                    variant={row.presentIn === runs.length ? "default" : "secondary"}
                    className={row.presentIn === runs.length ? "bg-accent text-accent-foreground" : ""}
                  >
                    {row.presentIn}/{runs.length}
                  </Badge>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      {filteredData.length > 100 && (
        <p className="text-xs text-muted-foreground mt-2 text-center">
          Showing first 100 of {filteredData.length} features. Use search to filter.
        </p>
      )}

      {/* Legend */}
      <div className="mt-4 p-3 bg-muted/30 rounded-lg text-xs text-muted-foreground">
        <strong>Note:</strong> Feature importance values come from the Random Forest model's variable importance scores. 
        Rank indicates the feature's position when sorted by importance (1 = most important). 
        Features common across all runs may represent stable biomarker candidates.
      </div>
    </div>
  );
}
