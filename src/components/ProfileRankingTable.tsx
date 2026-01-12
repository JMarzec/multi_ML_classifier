import { useState, useMemo } from "react";
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
import { cn } from "@/lib/utils";
import type { ProfileRanking } from "@/types/ml-results";
import { Search, ArrowUpDown, CheckCircle2, XCircle } from "lucide-react";

interface ProfileRankingTableProps {
  rankings: ProfileRanking[];
  topPercent: number;
}

export function ProfileRankingTable({ rankings, topPercent }: ProfileRankingTableProps) {
  const [search, setSearch] = useState("");
  const [sortField, setSortField] = useState<keyof ProfileRanking>("rank");
  const [sortAsc, setSortAsc] = useState(true);

  const filteredAndSorted = useMemo(() => {
    let data = [...rankings];

    // Filter
    if (search) {
      const searchLower = search.toLowerCase();
      data = data.filter(
        (r) =>
          r.sample_index.toString().includes(searchLower) ||
          r.actual_class.toLowerCase().includes(searchLower) ||
          r.predicted_class.toLowerCase().includes(searchLower)
      );
    }

    // Sort
    data.sort((a, b) => {
      const aVal = a[sortField];
      const bVal = b[sortField];
      const comparison = typeof aVal === "number" 
        ? aVal - (bVal as number)
        : String(aVal).localeCompare(String(bVal));
      return sortAsc ? comparison : -comparison;
    });

    return data;
  }, [rankings, search, sortField, sortAsc]);

  const handleSort = (field: keyof ProfileRanking) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(true);
    }
  };

  const stats = useMemo(() => {
    const top = rankings.filter((r) => r.top_profile);
    const topCorrect = top.filter((r) => r.correct).length;
    const totalCorrect = rankings.filter((r) => r.correct).length;
    
    return {
      topCount: top.length,
      topAccuracy: top.length > 0 ? (topCorrect / top.length) * 100 : 0,
      overallAccuracy: (totalCorrect / rankings.length) * 100,
      avgConfidenceTop: top.length > 0 
        ? (top.reduce((sum, r) => sum + r.confidence, 0) / top.length) * 100 
        : 0,
    };
  }, [rankings]);

  return (
    <div className="bg-card rounded-xl border border-border">
      <div className="p-6 border-b border-border">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold">Profile Rankings</h3>
            <p className="text-sm text-muted-foreground mt-1">
              Top {topPercent}% profiles ({stats.topCount} samples) based on prediction confidence
            </p>
          </div>
          
          <div className="relative max-w-xs">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
            <Input
              placeholder="Search samples..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="pl-9 bg-muted/50"
            />
          </div>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
          <div className="bg-muted/50 rounded-lg p-3">
            <p className="text-xs text-muted-foreground">Top Profile Accuracy</p>
            <p className="text-xl font-bold text-accent">{stats.topAccuracy.toFixed(1)}%</p>
          </div>
          <div className="bg-muted/50 rounded-lg p-3">
            <p className="text-xs text-muted-foreground">Overall Accuracy</p>
            <p className="text-xl font-bold text-primary">{stats.overallAccuracy.toFixed(1)}%</p>
          </div>
          <div className="bg-muted/50 rounded-lg p-3">
            <p className="text-xs text-muted-foreground">Avg Confidence (Top)</p>
            <p className="text-xl font-bold text-secondary">{stats.avgConfidenceTop.toFixed(1)}%</p>
          </div>
          <div className="bg-muted/50 rounded-lg p-3">
            <p className="text-xs text-muted-foreground">Total Samples</p>
            <p className="text-xl font-bold text-foreground">{rankings.length}</p>
          </div>
        </div>
      </div>

      <div className="max-h-[500px] overflow-auto">
        <Table>
          <TableHeader className="sticky top-0 bg-card z-10">
            <TableRow>
              <TableHead className="cursor-pointer hover:bg-muted/50" onClick={() => handleSort("rank")}>
                <div className="flex items-center gap-1">
                  Rank <ArrowUpDown className="w-3 h-3" />
                </div>
              </TableHead>
              <TableHead className="cursor-pointer hover:bg-muted/50" onClick={() => handleSort("sample_index")}>
                <div className="flex items-center gap-1">
                  Sample <ArrowUpDown className="w-3 h-3" />
                </div>
              </TableHead>
              <TableHead>Actual</TableHead>
              <TableHead>Predicted</TableHead>
              <TableHead className="cursor-pointer hover:bg-muted/50" onClick={() => handleSort("ensemble_probability")}>
                <div className="flex items-center gap-1">
                  Probability <ArrowUpDown className="w-3 h-3" />
                </div>
              </TableHead>
              <TableHead className="cursor-pointer hover:bg-muted/50" onClick={() => handleSort("confidence")}>
                <div className="flex items-center gap-1">
                  Confidence <ArrowUpDown className="w-3 h-3" />
                </div>
              </TableHead>
              <TableHead>Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredAndSorted.slice(0, 100).map((row) => (
              <TableRow 
                key={row.sample_index}
                className={cn(
                  row.top_profile && "bg-primary/5",
                  !row.correct && "bg-destructive/5"
                )}
              >
                <TableCell>
                  <div className="flex items-center gap-2">
                    <span className="font-mono">#{row.rank}</span>
                    {row.top_profile && (
                      <Badge variant="secondary" className="bg-primary/20 text-primary text-xs">
                        Top {topPercent}%
                      </Badge>
                    )}
                  </div>
                </TableCell>
                <TableCell className="font-mono">{row.sample_index}</TableCell>
                <TableCell>
                  <Badge variant={row.actual_class === "1" ? "default" : "outline"}>
                    {row.actual_class === "1" ? "Positive" : "Negative"}
                  </Badge>
                </TableCell>
                <TableCell>
                  <Badge variant={row.predicted_class === "1" ? "default" : "outline"}>
                    {row.predicted_class === "1" ? "Positive" : "Negative"}
                  </Badge>
                </TableCell>
                <TableCell className="font-mono">
                  {(row.ensemble_probability * 100).toFixed(1)}%
                </TableCell>
                <TableCell>
                  <div className="flex items-center gap-2">
                    <div className="w-16 h-2 bg-muted rounded-full overflow-hidden">
                      <div 
                        className="h-full bg-primary rounded-full"
                        style={{ width: `${row.confidence * 100}%` }}
                      />
                    </div>
                    <span className="text-xs text-muted-foreground">
                      {(row.confidence * 100).toFixed(0)}%
                    </span>
                  </div>
                </TableCell>
                <TableCell>
                  {row.correct ? (
                    <CheckCircle2 className="w-5 h-5 text-accent" />
                  ) : (
                    <XCircle className="w-5 h-5 text-destructive" />
                  )}
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
        {filteredAndSorted.length > 100 && (
          <div className="p-4 text-center text-sm text-muted-foreground border-t border-border">
            Showing first 100 of {filteredAndSorted.length} samples
          </div>
        )}
      </div>
    </div>
  );
}
