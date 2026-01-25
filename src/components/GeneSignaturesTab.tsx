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
import { Search, Download, Dna, ExternalLink } from "lucide-react";
import { cn } from "@/lib/utils";
import { toast } from "@/hooks/use-toast";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover";
import type { MLResults } from "@/types/ml-results";

// Gene annotation database links configuration - grouped by category
const geneDbCategories = [
  {
    name: "General",
    color: "text-blue-600",
    databases: [
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
    ],
  },
  {
    name: "Cancer",
    color: "text-rose-600",
    databases: [
      { id: "civic", label: "Cv", name: "CIViC", color: "text-teal-500 hover:text-teal-600", 
        url: (gene: string) => `https://civicdb.org/entities/genes?name=${encodeURIComponent(gene)}` },
      { id: "vicc", label: "CV", name: "Cancer Variants", color: "text-rose-500 hover:text-rose-600", 
        url: (gene: string) => `https://search.cancervariants.org/?searchTerm=%23${encodeURIComponent(gene)}` },
      { id: "gepia", label: "GP", name: "GEPIA", color: "text-cyan-600 hover:text-cyan-700", 
        url: (gene: string) => `http://gepia.cancer-pku.cn/detail.php?gene=${encodeURIComponent(gene)}` },
      { id: "cansar", label: "CS", name: "canSAR", color: "text-pink-500 hover:text-pink-600", 
        url: (gene: string) => `https://cansar.ai/search?q=${encodeURIComponent(gene)}` },
    ],
  },
  {
    name: "Drug",
    color: "text-purple-600",
    databases: [
      { id: "dgidb", label: "DG", name: "DGIdb", color: "text-purple-500 hover:text-purple-600", 
        url: (gene: string) => `https://beta.dgidb.org/results?searchType=gene&searchTerms=${encodeURIComponent(gene)}` },
      { id: "gdsc", label: "RX", name: "Cancer Rx Gene", color: "text-orange-500 hover:text-orange-600", 
        url: (gene: string) => `https://www.cancerrxgene.org/search?query=${encodeURIComponent(gene)}` },
    ],
  },
];

const GeneLinks = ({ gene }: { gene: string }) => {
  return (
    <TooltipProvider>
      <div className="inline-flex items-center gap-1">
        <span className="font-mono text-sm">{gene}</span>
        <Popover>
          <PopoverTrigger asChild>
            <button
              className="inline-flex items-center justify-center w-5 h-5 rounded hover:bg-muted transition-colors text-muted-foreground hover:text-foreground"
              onClick={(e) => e.stopPropagation()}
            >
              <ExternalLink className="w-3.5 h-3.5" />
            </button>
          </PopoverTrigger>
          <PopoverContent 
            side="right" 
            align="start" 
            className="w-auto p-3 bg-popover border border-border shadow-lg"
          >
            <div className="space-y-3">
              <p className="text-xs font-medium text-muted-foreground">
                External databases for <span className="font-mono text-foreground">{gene}</span>
              </p>
              {geneDbCategories.map((category) => (
                <div key={category.name} className="space-y-1.5">
                  <p className={cn("text-[10px] font-semibold uppercase tracking-wider", category.color)}>
                    {category.name}
                  </p>
                  <div className="flex flex-wrap gap-1">
                    {category.databases.map((db) => (
                      <Tooltip key={db.id}>
                        <TooltipTrigger asChild>
                          <a
                            href={db.url(gene)}
                            target="_blank"
                            rel="noopener noreferrer"
                            className={cn(
                              "inline-flex items-center justify-center px-2 py-1 rounded text-[10px] font-bold hover:bg-muted transition-colors border border-border/50",
                              db.color
                            )}
                            onClick={(e) => e.stopPropagation()}
                          >
                            {db.label}
                          </a>
                        </TooltipTrigger>
                        <TooltipContent side="top" className="text-xs">
                          {db.name}
                        </TooltipContent>
                      </Tooltip>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </PopoverContent>
        </Popover>
      </div>
    </TooltipProvider>
  );
};

interface GeneSignaturesTabProps {
  data: MLResults;
}

interface FeatureRow {
  feature: string;
  rank: number;
  importance: number;
}

export function GeneSignaturesTab({ data }: GeneSignaturesTabProps) {
  const [searchTerm, setSearchTerm] = useState("");
  const [sortBy, setSortBy] = useState<"rank" | "importance">("rank");
  const [activeTab, setActiveTab] = useState("table");

  // Build feature data with ranks and importance
  const featureData = useMemo(() => {
    const importanceMap = new Map(
      (data.feature_importance || []).map((f) => [f.feature, f.importance])
    );
    
    // Get selected features with their importance scores
    const features: FeatureRow[] = (data.selected_features || []).map((feature, idx) => ({
      feature,
      importance: importanceMap.get(feature) ?? 0,
      rank: idx + 1,
    }));

    // Sort by importance descending and re-assign ranks
    features.sort((a, b) => b.importance - a.importance);
    features.forEach((f, i) => { f.rank = i + 1; });

    return features;
  }, [data]);

  // Filter and sort data
  const filteredData = useMemo(() => {
    let filtered = featureData;
    
    if (searchTerm) {
      const term = searchTerm.toLowerCase();
      filtered = filtered.filter((r) => r.feature.toLowerCase().includes(term));
    }
    
    if (sortBy === "rank") {
      filtered = [...filtered].sort((a, b) => a.rank - b.rank);
    } else {
      filtered = [...filtered].sort((a, b) => b.importance - a.importance);
    }
    
    return filtered;
  }, [featureData, searchTerm, sortBy]);

  // Export to CSV
  const exportToCSV = () => {
    const headers = ["Rank", "Gene/Feature", "Importance Score"];
    const rows = featureData.map((f) => [
      f.rank.toString(),
      f.feature,
      f.importance.toFixed(6),
    ]);

    const csvContent = [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `model_signature_${new Date().toISOString().split("T")[0]}.csv`;
    link.click();
    URL.revokeObjectURL(url);

    toast({
      title: "Signature Exported",
      description: `Exported ${featureData.length} genes to CSV.`,
    });
  };

  if (featureData.length === 0) {
    return (
      <div className="bg-card rounded-xl p-12 border border-border text-center">
        <Dna className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
        <h3 className="text-lg font-semibold mb-2">No Gene Signatures Available</h3>
        <p className="text-muted-foreground">
          No selected features or feature importance data found in this analysis.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-card rounded-xl p-6 border border-border space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <Dna className="w-5 h-5" />
          Model Details (Gene Signatures)
        </h3>
        
        <Button variant="outline" size="sm" onClick={exportToCSV}>
          <Download className="w-4 h-4 mr-2" />
          Export CSV
        </Button>
      </div>

      <p className="text-sm text-muted-foreground">
        Selected genes (model signature) with their importance scores.
        These features were selected by the ML pipeline for predictive modeling.
      </p>

      <div className="flex flex-wrap gap-2 mb-2">
        <Badge variant="secondary">
          Total features: {featureData.length}
        </Badge>
        <Badge variant="outline" className="text-primary border-primary">
          Top feature: {featureData[0]?.feature}
        </Badge>
      </div>

      <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
        <TabsList className="grid w-full grid-cols-2 max-w-md">
          <TabsTrigger value="table">Ranked Table</TabsTrigger>
          <TabsTrigger value="list">Gene List</TabsTrigger>
        </TabsList>

        {/* Ranked Table Tab */}
        <TabsContent value="table" className="space-y-4 mt-4">
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
                  sortBy === "rank"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted hover:bg-muted/80"
                )}
                onClick={() => setSortBy("rank")}
              >
                By Rank
              </button>
              <button
                className={cn(
                  "px-3 py-1.5 text-xs font-medium transition-colors",
                  sortBy === "importance"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted hover:bg-muted/80"
                )}
                onClick={() => setSortBy("importance")}
              >
                By Importance
              </button>
            </div>
          </div>

          <div className="overflow-x-auto max-h-[500px] overflow-y-auto border rounded-lg">
            <Table>
              <TableHeader className="sticky top-0 bg-background z-10">
                <TableRow>
                  <TableHead className="w-[80px]">Rank</TableHead>
                  <TableHead className="w-[200px]">Feature</TableHead>
                  <TableHead className="text-right">Importance Score</TableHead>
                  <TableHead className="text-right w-[120px]">Relative %</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {filteredData.slice(0, 50).map((row) => {
                  const maxImportance = featureData[0]?.importance || 1;
                  const relativePercent = (row.importance / maxImportance) * 100;
                  
                  return (
                    <TableRow key={row.feature}>
                      <TableCell>
                        <Badge 
                          variant={row.rank <= 5 ? "default" : row.rank <= 10 ? "secondary" : "outline"}
                          className={row.rank <= 5 ? "bg-accent text-accent-foreground" : ""}
                        >
                          #{row.rank}
                        </Badge>
                      </TableCell>
                      <TableCell>
                        <GeneLinks gene={row.feature} />
                      </TableCell>
                      <TableCell className="text-right font-mono">
                        {row.importance < 0.001 
                          ? row.importance.toExponential(3) 
                          : row.importance.toFixed(4)}
                      </TableCell>
                      <TableCell className="text-right">
                        <div className="flex items-center justify-end gap-2">
                          <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                            <div 
                              className="h-full bg-primary rounded-full" 
                              style={{ width: `${relativePercent}%` }}
                            />
                          </div>
                          <span className="text-xs text-muted-foreground w-10">
                            {relativePercent.toFixed(0)}%
                          </span>
                        </div>
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>

          {filteredData.length > 50 && (
            <p className="text-xs text-muted-foreground text-center">
              Showing top 50 of {filteredData.length} features. Use search to find specific features.
            </p>
          )}
        </TabsContent>

        {/* Gene List Tab */}
        <TabsContent value="list" className="mt-4">
          <div className="bg-muted/30 rounded-lg p-4 border border-border">
            <div className="flex items-center justify-between mb-3">
              <h4 className="font-medium">Selected Gene Signature</h4>
              <Badge variant="outline">{featureData.length} genes</Badge>
            </div>
            
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-2">
              {featureData.map((row) => (
                <div 
                  key={row.feature}
                  className="bg-background rounded-md p-2 border border-border/50 hover:border-primary/50 transition-colors"
                >
                  <div className="flex items-center justify-between gap-1">
                    <GeneLinks gene={row.feature} />
                    <Badge 
                      variant="outline" 
                      className={cn(
                        "text-[10px] px-1 py-0",
                        row.rank <= 5 && "bg-accent/20 text-accent border-accent",
                        row.rank > 5 && row.rank <= 10 && "bg-primary/20 text-primary border-primary"
                      )}
                    >
                      #{row.rank}
                    </Badge>
                  </div>
                  <p className="text-[10px] text-muted-foreground mt-1 font-mono">
                    {row.importance.toFixed(4)}
                  </p>
                </div>
              ))}
            </div>
          </div>
        </TabsContent>
      </Tabs>

      {/* Legend */}
      <div className="mt-4 p-3 bg-muted/30 rounded-lg text-xs text-muted-foreground">
        <strong>Note:</strong> Feature importance values come from the Random Forest model's variable importance scores. 
        Click the <ExternalLink className="w-3 h-3 inline mx-0.5" /> icon next to any gene to access external databases 
        including GeneCards, NCBI, Ensembl, UniProt, and cancer-specific resources.
      </div>
    </div>
  );
}
