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
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { ProfileRanking } from "@/types/ml-results";
import { Search, ArrowUpDown, CheckCircle2, XCircle, ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight } from "lucide-react";

interface ProfileRankingTableProps {
  rankings: ProfileRanking[];
  topPercent: number;
}

const CLASS_NAMES: Record<string, string> = {
  "0": "Negative",
  "1": "Positive",
};

const PAGE_SIZE = 50;

export function ProfileRankingTable({ rankings, topPercent }: ProfileRankingTableProps) {
  const [search, setSearch] = useState("");
  const [sortField, setSortField] = useState<keyof ProfileRanking>("rank");
  const [sortAsc, setSortAsc] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);

  const filteredAndSorted = useMemo(() => {
    let data = [...rankings];

    // Filter by sample_id or sample_index
    if (search) {
      const searchLower = search.toLowerCase();
      data = data.filter(
        (r) =>
          (r.sample_id && r.sample_id.toLowerCase().includes(searchLower)) ||
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

  // Reset page when search changes
  useMemo(() => {
    setCurrentPage(1);
  }, [search]);

  const totalPages = Math.ceil(filteredAndSorted.length / PAGE_SIZE);
  const paginatedData = filteredAndSorted.slice(
    (currentPage - 1) * PAGE_SIZE,
    currentPage * PAGE_SIZE
  );

  const handleSort = (field: keyof ProfileRanking) => {
    if (sortField === field) {
      setSortAsc(!sortAsc);
    } else {
      setSortField(field);
      setSortAsc(true);
    }
    setCurrentPage(1);
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

  // Helper to display sample name
  const getSampleName = (r: ProfileRanking) => {
    return r.sample_id || `Sample_${r.sample_index}`;
  };

  return (
    <div className="bg-card rounded-xl border border-border">
      <div className="p-6 border-b border-border">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h3 className="text-lg font-semibold">Profile Rankings</h3>
            <p className="text-sm text-muted-foreground mt-1">
              All {rankings.length} samples ranked by prediction confidence 
              <span className="text-primary"> (Top {topPercent}%: {stats.topCount} samples)</span>
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
              <TableHead className="cursor-pointer hover:bg-muted/50" onClick={() => handleSort("sample_id")}>
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
              <TableHead>
                <div className="text-center">
                  Risk Scores
                  <div className="text-xs font-normal text-muted-foreground">Neg / Pos</div>
                </div>
              </TableHead>
              <TableHead>Status</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {paginatedData.map((row) => (
              <TableRow 
                key={row.sample_id || row.sample_index}
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
                <TableCell className="font-mono font-medium">{getSampleName(row)}</TableCell>
                <TableCell>
                  <Badge variant={row.actual_class === "1" ? "default" : "outline"}>
                    {CLASS_NAMES[row.actual_class] || row.actual_class}
                  </Badge>
                </TableCell>
                <TableCell>
                  <Badge variant={row.predicted_class === "1" ? "default" : "outline"}>
                    {CLASS_NAMES[row.predicted_class] || row.predicted_class}
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
                  <div className="flex items-center gap-1 text-xs font-mono">
                    <span 
                      className={cn(
                        "px-2 py-0.5 rounded",
                        row.risk_score_class_0 && row.risk_score_class_0 > 50 
                          ? "bg-primary/20 text-primary font-semibold" 
                          : "bg-muted text-muted-foreground"
                      )}
                      title="Risk Score for Negative Class"
                    >
                      {row.risk_score_class_0 !== undefined ? row.risk_score_class_0.toFixed(0) : '--'}
                    </span>
                    <span className="text-muted-foreground">/</span>
                    <span 
                      className={cn(
                        "px-2 py-0.5 rounded",
                        row.risk_score_class_1 && row.risk_score_class_1 > 50 
                          ? "bg-destructive/20 text-destructive font-semibold" 
                          : "bg-muted text-muted-foreground"
                      )}
                      title="Risk Score for Positive Class"
                    >
                      {row.risk_score_class_1 !== undefined ? row.risk_score_class_1.toFixed(0) : '--'}
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
      </div>

      {/* Pagination Controls */}
      {totalPages > 1 && (
        <div className="flex items-center justify-between p-4 border-t border-border">
          <p className="text-sm text-muted-foreground">
            Showing {(currentPage - 1) * PAGE_SIZE + 1}-{Math.min(currentPage * PAGE_SIZE, filteredAndSorted.length)} of {filteredAndSorted.length} samples
          </p>
          <div className="flex items-center gap-1">
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={() => setCurrentPage(1)}
              disabled={currentPage === 1}
            >
              <ChevronsLeft className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage === 1}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="px-3 text-sm">
              Page {currentPage} of {totalPages}
            </span>
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
              disabled={currentPage === totalPages}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
            <Button
              variant="outline"
              size="icon"
              className="h-8 w-8"
              onClick={() => setCurrentPage(totalPages)}
              disabled={currentPage === totalPages}
            >
              <ChevronsRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      )}
    </div>
  );
}
