import { useMemo, useState } from "react";
import { Search, Filter, ArrowUpDown, Dna } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import type { MLResults } from "@/types/ml-results";

interface FeatureStabilityPanelProps {
  runs: { name: string; data: MLResults }[];
  runColors: { fill: string; text: string; bg: string; border: string }[];
  runLabels: string[];
}

interface FeatureStability {
  feature: string;
  intersectionCount: number; // How many runs include this feature
  intersectionLabel: string; // e.g., "A+B+C" or "All"
  runIndices: number[];
  avgImportanceRank: number;
  importanceRanks: (number | null)[]; // Rank in each run (null if not present)
}

type SortKey = "intersection" | "avgRank" | "feature";

export function FeatureStabilityPanel({ runs, runColors, runLabels }: FeatureStabilityPanelProps) {
  const [search, setSearch] = useState("");
  const [filterGroup, setFilterGroup] = useState<string>("all");
  const [sortBy, setSortBy] = useState<SortKey>("intersection");
  const [sortAsc, setSortAsc] = useState(false);

  // Build stability data
  const { stabilityData, intersectionGroups } = useMemo(() => {
    // Build feature importance ranks per run
    const featureRanksPerRun: Map<string, (number | null)[]> = new Map();
    
    runs.forEach((run, runIdx) => {
      const importance = run.data.feature_importance || [];
      // Sort by importance descending to get ranks
      const sorted = [...importance].sort((a, b) => b.importance - a.importance);
      sorted.forEach((f, rank) => {
        if (!featureRanksPerRun.has(f.feature)) {
          featureRanksPerRun.set(f.feature, new Array(runs.length).fill(null));
        }
        featureRanksPerRun.get(f.feature)![runIdx] = rank + 1;
      });
      
      // Also include selected features
      (run.data.selected_features || []).forEach((feature) => {
        if (!featureRanksPerRun.has(feature)) {
          featureRanksPerRun.set(feature, new Array(runs.length).fill(null));
        }
      });
    });

    // Calculate stability metrics
    const data: FeatureStability[] = [];
    const groups = new Set<string>();
    groups.add("all");

    featureRanksPerRun.forEach((ranks, feature) => {
      const runIndices = ranks
        .map((r, idx) => (r !== null ? idx : -1))
        .filter((idx) => idx !== -1);
      
      const intersectionCount = runIndices.length;
      
      // Create intersection label
      let intersectionLabel: string;
      if (intersectionCount === runs.length) {
        intersectionLabel = "All Runs";
      } else if (intersectionCount === 1) {
        intersectionLabel = `Only ${runLabels[runIndices[0]]}`;
      } else {
        intersectionLabel = runIndices.map((i) => runLabels[i].replace("Run ", "")).join("+");
      }
      groups.add(intersectionLabel);
      
      // Calculate average importance rank (only for runs where present)
      const validRanks = ranks.filter((r) => r !== null) as number[];
      const avgImportanceRank = validRanks.length > 0
        ? validRanks.reduce((a, b) => a + b, 0) / validRanks.length
        : 999;
      
      data.push({
        feature,
        intersectionCount,
        intersectionLabel,
        runIndices,
        avgImportanceRank,
        importanceRanks: ranks,
      });
    });

    return {
      stabilityData: data,
      intersectionGroups: Array.from(groups).sort((a, b) => {
        if (a === "all") return -1;
        if (b === "all") return 1;
        if (a === "All Runs") return -1;
        if (b === "All Runs") return 1;
        return a.localeCompare(b);
      }),
    };
  }, [runs, runLabels]);

  // Filter and sort
  const filteredData = useMemo(() => {
    let result = stabilityData;
    
    // Search filter
    if (search) {
      const term = search.toLowerCase();
      result = result.filter((f) => f.feature.toLowerCase().includes(term));
    }
    
    // Group filter
    if (filterGroup !== "all") {
      result = result.filter((f) => f.intersectionLabel === filterGroup);
    }
    
    // Sort
    result = [...result].sort((a, b) => {
      let cmp = 0;
      switch (sortBy) {
        case "intersection":
          cmp = b.intersectionCount - a.intersectionCount;
          break;
        case "avgRank":
          cmp = a.avgImportanceRank - b.avgImportanceRank;
          break;
        case "feature":
          cmp = a.feature.localeCompare(b.feature);
          break;
      }
      return sortAsc ? -cmp : cmp;
    });
    
    return result;
  }, [stabilityData, search, filterGroup, sortBy, sortAsc]);

  const toggleSort = (key: SortKey) => {
    if (sortBy === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortBy(key);
      setSortAsc(false);
    }
  };

  return (
    <div className="bg-card rounded-xl p-6 border border-border">
      <div className="flex items-center gap-2 mb-4">
        <Dna className="w-5 h-5 text-primary" />
        <h3 className="text-lg font-semibold">Feature Stability Analysis</h3>
      </div>
      
      <p className="text-sm text-muted-foreground mb-4">
        Features ranked by intersection frequency across runs and average importance rank.
        Higher intersection count + lower average rank = more stable/important features.
      </p>
      
      {/* Controls */}
      <div className="flex flex-wrap gap-3 mb-4">
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search features..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="pl-9"
          />
        </div>
        
        <div className="flex items-center gap-2">
          <Filter className="w-4 h-4 text-muted-foreground" />
          <Select value={filterGroup} onValueChange={setFilterGroup}>
            <SelectTrigger className="w-[160px]">
              <SelectValue placeholder="Filter by group" />
            </SelectTrigger>
            <SelectContent>
              {intersectionGroups.map((group) => (
                <SelectItem key={group} value={group}>
                  {group === "all" ? "All Groups" : group}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
      
      {/* Table */}
      <div className="overflow-x-auto max-h-[500px] overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-card z-10">
            <tr className="border-b border-border">
              <th 
                className="text-left py-2 px-3 cursor-pointer hover:bg-muted/50"
                onClick={() => toggleSort("feature")}
              >
                <div className="flex items-center gap-1">
                  Feature
                  {sortBy === "feature" && <ArrowUpDown className="w-3 h-3" />}
                </div>
              </th>
              <th 
                className="text-center py-2 px-3 cursor-pointer hover:bg-muted/50"
                onClick={() => toggleSort("intersection")}
              >
                <div className="flex items-center justify-center gap-1">
                  Runs
                  {sortBy === "intersection" && <ArrowUpDown className="w-3 h-3" />}
                </div>
              </th>
              <th className="text-center py-2 px-3">Intersection</th>
              <th 
                className="text-center py-2 px-3 cursor-pointer hover:bg-muted/50"
                onClick={() => toggleSort("avgRank")}
              >
                <div className="flex items-center justify-center gap-1">
                  Avg Rank
                  {sortBy === "avgRank" && <ArrowUpDown className="w-3 h-3" />}
                </div>
              </th>
              {runs.map((_, idx) => (
                <th key={idx} className={cn("text-center py-2 px-3", runColors[idx].text)}>
                  {runLabels[idx].replace("Run ", "")}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {filteredData.slice(0, 100).map((row) => (
              <tr key={row.feature} className="border-b border-border/50 hover:bg-muted/30">
                <td className="py-2 px-3 font-mono text-xs">{row.feature}</td>
                <td className="py-2 px-3 text-center">
                  <Badge variant={row.intersectionCount === runs.length ? "default" : "secondary"}>
                    {row.intersectionCount}/{runs.length}
                  </Badge>
                </td>
                <td className="py-2 px-3 text-center">
                  <span className={cn(
                    "text-xs px-2 py-0.5 rounded",
                    row.intersectionCount === runs.length 
                      ? "bg-accent/20 text-accent"
                      : "bg-muted text-muted-foreground"
                  )}>
                    {row.intersectionLabel}
                  </span>
                </td>
                <td className="py-2 px-3 text-center font-mono">
                  {row.avgImportanceRank < 999 ? row.avgImportanceRank.toFixed(1) : "—"}
                </td>
                {row.importanceRanks.map((rank, idx) => (
                  <td key={idx} className="py-2 px-3 text-center">
                    {rank !== null ? (
                      <span className={cn(
                        "font-mono text-xs",
                        rank <= 10 && runColors[idx].text
                      )}>
                        #{rank}
                      </span>
                    ) : (
                      <span className="text-muted-foreground">—</span>
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      
      {filteredData.length > 100 && (
        <p className="text-xs text-muted-foreground mt-3">
          Showing first 100 of {filteredData.length} features. Use search to find specific genes.
        </p>
      )}
      
      {filteredData.length === 0 && (
        <p className="text-center text-muted-foreground py-8">
          No features match your search/filter criteria.
        </p>
      )}
    </div>
  );
}
