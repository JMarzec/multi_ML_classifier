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
import { Search, TrendingUp, ArrowUpDown } from "lucide-react";
import { cn } from "@/lib/utils";
import type { MLResults, FeatureImportance } from "@/types/ml-results";

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

  return (
    <div className="bg-card rounded-xl p-6 border border-border space-y-4">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <h3 className="text-lg font-semibold flex items-center gap-2">
          <TrendingUp className="w-5 h-5" />
          Feature Importance Rankings
        </h3>
        
        <div className="flex items-center gap-3">
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
      </div>

      <p className="text-sm text-muted-foreground">
        Comparison of feature importance rankings across runs. Lower rank = higher importance.
        Features present in more runs are highlighted.
      </p>

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
                <TableCell className="font-mono text-sm">{row.feature}</TableCell>
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

      <div className="bg-muted/30 rounded-lg p-4 text-sm mt-4">
        <p className="text-muted-foreground">
          <strong>Note:</strong> Rankings are based on feature importance scores from each analysis run.
          Features ranked highly (low number) across multiple runs indicate robust biomarkers.
          The importance values shown below ranks are the raw scores from the ensemble model.
        </p>
      </div>
    </div>
  );
}
