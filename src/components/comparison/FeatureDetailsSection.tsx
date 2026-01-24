import { useMemo, useState } from "react";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Search, FileText, Download, Dna } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "@/hooks/use-toast";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import type { MLResults } from "@/types/ml-results";

// Gene annotation database links configuration
const geneDatabases = [
  { id: "gc", label: "GC", name: "GeneCards", color: "text-blue-500 hover:text-blue-600", 
    url: (gene: string) => `https://www.genecards.org/cgi-bin/carddisp.pl?gene=${encodeURIComponent(gene)}` },
  { id: "ncbi", label: "N", name: "NCBI Gene", color: "text-green-600 hover:text-green-700", 
    url: (gene: string) => `https://www.ncbi.nlm.nih.gov/gene/?term=${encodeURIComponent(gene)}` },
  { id: "ensembl", label: "E", name: "Ensembl", color: "text-red-500 hover:text-red-600", 
    url: (gene: string) => `https://www.ensembl.org/Human/Search/Results?q=${encodeURIComponent(gene)}` },
  { id: "uniprot", label: "U", name: "UniProt", color: "text-amber-600 hover:text-amber-700", 
    url: (gene: string) => `https://www.uniprot.org/uniprotkb?query=${encodeURIComponent(gene)}` },
  { id: "hpa", label: "PA", name: "Human Protein Atlas", color: "text-indigo-500 hover:text-indigo-600", 
    url: (gene: string) => `https://www.proteinatlas.org/search/${encodeURIComponent(gene)}` },
  { id: "civic", label: "Cv", name: "CIViC", color: "text-teal-500 hover:text-teal-600", 
    url: (gene: string) => `https://civicdb.org/entities/genes?name=${encodeURIComponent(gene)}` },
  { id: "vicc", label: "CV", name: "Cancer Variants", color: "text-rose-500 hover:text-rose-600", 
    url: (gene: string) => `https://search.cancervariants.org/?searchTerm=${encodeURIComponent(gene)}` },
  { id: "gepia", label: "GP", name: "GEPIA", color: "text-cyan-600 hover:text-cyan-700", 
    url: (gene: string) => `http://gepia.cancer-pku.cn/detail.php?gene=${encodeURIComponent(gene)}` },
  { id: "dgidb", label: "DG", name: "DGIdb", color: "text-purple-500 hover:text-purple-600", 
    url: (gene: string) => `https://beta.dgidb.org/genes/${encodeURIComponent(gene)}` },
  { id: "gdsc", label: "RX", name: "Cancer Rx Gene", color: "text-orange-500 hover:text-orange-600", 
    url: (gene: string) => `https://www.cancerrxgene.org/translation/Gene/${encodeURIComponent(gene)}` },
  { id: "cansar", label: "CS", name: "canSAR", color: "text-pink-500 hover:text-pink-600", 
    url: (gene: string) => `https://cansar.ai/search?q=${encodeURIComponent(gene)}` },
];

const GeneLinks = ({ gene }: { gene: string }) => {
  return (
    <TooltipProvider>
      <div className="inline-flex items-center gap-0.5 flex-wrap">
        <span className="font-mono text-sm mr-1">{gene}</span>
        {geneDatabases.map((db) => (
          <Tooltip key={db.id}>
            <TooltipTrigger asChild>
              <a
                href={db.url(gene)}
                target="_blank"
                rel="noopener noreferrer"
                className={cn(
                  "inline-flex items-center justify-center w-5 h-5 rounded hover:bg-muted transition-colors",
                  db.color
                )}
                onClick={(e) => e.stopPropagation()}
              >
                <span className="text-[8px] font-bold">{db.label}</span>
              </a>
            </TooltipTrigger>
            <TooltipContent side="top" className="text-xs">
              {db.name}
            </TooltipContent>
          </Tooltip>
        ))}
      </div>
    </TooltipProvider>
  );
};

interface FeatureDetailsSectionProps {
  runs: { name: string; data: MLResults }[];
  runColors: { fill: string; text: string; bg: string; border: string }[];
  runLabels: string[];
}

interface FeatureRow {
  feature: string;
  ranks: (number | null)[];
  importances: (number | null)[];
  avgRank: number;
  presentIn: number;
}

export function FeatureDetailsSection({
  runs,
  runColors,
  runLabels,
}: FeatureDetailsSectionProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [sortBy, setSortBy] = useState<"avgRank" | "presentIn">("avgRank");
  const [activeTab, setActiveTab] = useState("comparison");

  // Per-run gene signatures (selected features with importance)
  const runSignatures = useMemo(() => {
    return runs.map((run) => {
      const importanceMap = new Map(
        (run.data.feature_importance || []).map((f) => [f.feature, f.importance])
      );
      
      // Get selected features with their importance scores
      const features = (run.data.selected_features || []).map((feature, idx) => ({
        feature,
        importance: importanceMap.get(feature) ?? null,
        rank: idx + 1,
      }));

      // Sort by importance if available
      features.sort((a, b) => {
        if (a.importance !== null && b.importance !== null) {
          return b.importance - a.importance;
        }
        return a.rank - b.rank;
      });

      // Re-assign ranks after sorting
      features.forEach((f, i) => { f.rank = i + 1; });

      return {
        name: run.name,
        featureCount: features.length,
        features,
      };
    });
  }, [runs]);

  const featureData = useMemo(() => {
    // Get all unique features across all runs
    const allFeatures = new Set<string>();
    runs.forEach((run) => {
      run.data.selected_features?.forEach((f) => allFeatures.add(f));
    });

    // Build importance rankings per run
    const importanceMaps = runs.map((run) => {
      const sorted = [...(run.data.feature_importance || [])]
        .sort((a, b) => b.importance - a.importance);
      
      const map = new Map<string, { rank: number; importance: number }>();
      sorted.forEach((f, idx) => {
        map.set(f.feature, { rank: idx + 1, importance: f.importance });
      });
      return map;
    });

    // Build feature rows
    const rows: FeatureRow[] = [...allFeatures].map((feature) => {
      const ranks = importanceMaps.map((m) => m.get(feature)?.rank ?? null);
      const importances = importanceMaps.map((m) => m.get(feature)?.importance ?? null);
      const validRanks = ranks.filter((r): r is number => r !== null);
      const avgRank = validRanks.length > 0
        ? validRanks.reduce((a, b) => a + b, 0) / validRanks.length
        : Infinity;
      const presentIn = validRanks.length;

      return { feature, ranks, importances, avgRank, presentIn };
    });

    return rows;
  }, [runs]);

  const filteredData = useMemo(() => {
    let data = featureData;
    
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      data = data.filter((r) => r.feature.toLowerCase().includes(term));
    }
    
    if (sortBy === "avgRank") {
      data = [...data].sort((a, b) => a.avgRank - b.avgRank);
    } else {
      data = [...data].sort((a, b) => b.presentIn - a.presentIn || a.avgRank - b.avgRank);
    }
    
    return data;
  }, [featureData, searchTerm, sortBy]);

  const topFeatures = filteredData.slice(0, 30);

  // Export per-run signature to CSV
  const exportSignatureCSV = (runIndex: number) => {
    const sig = runSignatures[runIndex];
    const headers = ["Rank", "Gene/Feature", "Importance Score"];
    const rows = sig.features.map((f) => [
      f.rank.toString(),
      f.feature,
      f.importance !== null ? f.importance.toFixed(6) : "N/A",
    ]);

    const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `model_signature_${runLabels[runIndex].replace(" ", "_")}_${new Date().toISOString().split("T")[0]}.csv`;
    link.click();
    URL.revokeObjectURL(url);

    toast({
      title: "Signature Exported",
      description: `Exported ${sig.featureCount} genes for ${runLabels[runIndex]}.`,
    });
  };

  // Export all signatures comparison
  const exportAllSignaturesCSV = () => {
    const headers = [
      "Gene/Feature",
      ...runLabels.slice(0, runs.length).flatMap((l) => [`${l} Rank`, `${l} Importance`]),
      "Avg Rank",
      "Present In",
    ];

    const rows = filteredData.map((row) => [
      row.feature,
      ...row.ranks.flatMap((r, i) => [
        r !== null ? r.toString() : "",
        row.importances[i] !== null ? row.importances[i]!.toFixed(6) : "",
      ]),
      row.avgRank === Infinity ? "" : row.avgRank.toFixed(2),
      `${row.presentIn}/${runs.length}`,
    ]);

    const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `model_details_comparison_${new Date().toISOString().split("T")[0]}.csv`;
    link.click();
    URL.revokeObjectURL(url);

    toast({
      title: "Comparison Exported",
      description: `Exported ${filteredData.length} features across ${runs.length} runs.`,
    });
  };

  return (
    <div className="bg-card rounded-xl p-6 border border-border space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Dna className="w-5 h-5" />
          Model Details (Gene Signatures)
        </h3>
        
        <Button variant="outline" size="sm" onClick={exportAllSignaturesCSV}>
          <Download className="w-4 h-4 mr-2" />
          Export All CSV
        </Button>
      </div>

      <p className="text-sm text-muted-foreground">
        Selected genes (model signatures) with their importance scores across runs.
        These features were selected by the ML pipeline for predictive modeling.
      </p>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-2 max-w-md">
          <TabsTrigger value="comparison">Cross-Run Comparison</TabsTrigger>
          <TabsTrigger value="per-run">Per-Run Signatures</TabsTrigger>
        </TabsList>

        {/* Cross-Run Comparison Tab */}
        <TabsContent value="comparison" className="space-y-4 mt-4">
          <div className="flex items-center gap-3 flex-wrap">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
              <Input
                placeholder="Search features..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-9 w-48"
              />
            </div>
            
            <div className="flex border rounded-lg overflow-hidden">
              <button
                className={cn(
                  "px-3 py-1.5 text-xs font-medium transition-colors",
                  sortBy === "avgRank"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted hover:bg-muted/80"
                )}
                onClick={() => setSortBy("avgRank")}
              >
                By Rank
              </button>
              <button
                className={cn(
                  "px-3 py-1.5 text-xs font-medium transition-colors",
                  sortBy === "presentIn"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted hover:bg-muted/80"
                )}
                onClick={() => setSortBy("presentIn")}
              >
                By Presence
              </button>
            </div>
          </div>

          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[200px]">Feature</TableHead>
                  {runs.map((_, idx) => (
                    <TableHead
                      key={idx}
                      className={cn("text-center", runColors[idx].text)}
                    >
                      {runLabels[idx]}
                    </TableHead>
                  ))}
                  <TableHead className="text-center">Avg Rank</TableHead>
                  <TableHead className="text-center">Present In</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {topFeatures.map((row) => (
                  <TableRow key={row.feature}>
                    <TableCell><GeneLinks gene={row.feature} /></TableCell>
                    {row.ranks.map((rank, idx) => (
                      <TableCell key={idx} className="text-center">
                        {rank !== null ? (
                          <div className="flex flex-col items-center">
                            <span
                              className={cn(
                                "inline-flex items-center justify-center w-8 h-6 rounded text-xs font-medium",
                                rank <= 5
                                  ? "bg-accent/20 text-accent"
                                  : rank <= 10
                                  ? "bg-primary/20 text-primary"
                                  : "bg-muted text-muted-foreground"
                              )}
                            >
                              #{rank}
                            </span>
                            <span className="text-[10px] text-muted-foreground mt-0.5">
                              {row.importances[idx] !== null
                                ? row.importances[idx]!.toFixed(3)
                                : ""}
                            </span>
                          </div>
                        ) : (
                          <span className="text-muted-foreground">—</span>
                        )}
                      </TableCell>
                    ))}
                    <TableCell className="text-center font-medium">
                      {row.avgRank === Infinity ? "—" : row.avgRank.toFixed(1)}
                    </TableCell>
                    <TableCell className="text-center">
                      <Badge
                        variant={row.presentIn === runs.length ? "default" : "outline"}
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

          {filteredData.length > 30 && (
            <p className="text-xs text-muted-foreground text-center">
              Showing top 30 of {filteredData.length} features. Use search to find specific features.
            </p>
          )}
        </TabsContent>

        {/* Per-Run Signatures Tab */}
        <TabsContent value="per-run" className="mt-4">
          <div className={cn(
            "grid gap-4",
            runs.length === 2 && "grid-cols-1 md:grid-cols-2",
            runs.length >= 3 && "grid-cols-1 md:grid-cols-2 lg:grid-cols-3",
            runs.length === 4 && "grid-cols-1 md:grid-cols-2"
          )}>
            {runSignatures.map((sig, idx) => (
              <div 
                key={sig.name}
                className={cn("rounded-xl border p-4", runColors[idx].bg, runColors[idx].border)}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div className={cn("w-3 h-3 rounded-full", runColors[idx].text.replace("text-", "bg-"))} />
                    <span className={cn("font-semibold", runColors[idx].text)}>{runLabels[idx]}</span>
                  </div>
                  <Button 
                    variant="ghost" 
                    size="sm" 
                    onClick={() => exportSignatureCSV(idx)}
                    className="h-7 px-2"
                  >
                    <Download className="w-3.5 h-3.5" />
                  </Button>
                </div>
                
                <p className="text-xs text-muted-foreground mb-2 truncate" title={sig.name}>
                  {sig.name}
                </p>
                
                <Badge variant="secondary" className="mb-3">
                  <FileText className="w-3 h-3 mr-1" />
                  {sig.featureCount} selected genes
                </Badge>

                <div className="max-h-[300px] overflow-y-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead className="w-12 text-xs">#</TableHead>
                        <TableHead className="text-xs">Gene</TableHead>
                        <TableHead className="text-right text-xs">Importance</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sig.features.slice(0, 20).map((f) => (
                        <TableRow key={f.feature}>
                          <TableCell className="text-xs text-muted-foreground">{f.rank}</TableCell>
                          <TableCell><GeneLinks gene={f.feature} /></TableCell>
                          <TableCell className="text-right font-mono text-xs">
                            {f.importance !== null 
                              ? (f.importance < 0.001 ? f.importance.toExponential(2) : f.importance.toFixed(4))
                              : "—"
                            }
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                  {sig.features.length > 20 && (
                    <p className="text-[10px] text-muted-foreground text-center mt-2">
                      +{sig.features.length - 20} more genes (export CSV to see all)
                    </p>
                  )}
                </div>
              </div>
            ))}
          </div>
        </TabsContent>
      </Tabs>

      <div className="bg-muted/30 rounded-lg p-4 text-sm mt-4">
        <p className="text-muted-foreground">
          <strong>Note:</strong> The model signature consists of selected genes used for prediction.
          Importance scores indicate each gene's contribution to the model.
          Genes appearing across multiple runs suggest robust biomarkers.
        </p>
      </div>
    </div>
  );
}
