import { CheckCircle2, XCircle, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { PermutationTesting } from "@/types/ml-results";

interface PermutationTestingPanelProps {
  permutation: PermutationTesting;
}

export function PermutationTestingPanel({ permutation }: PermutationTestingPanelProps) {
  const renderMetric = (
    label: string,
    original: number,
    permutedMean: number,
    permutedSd: number,
    pValue: number,
    higherIsBetter: boolean = true
  ) => {
    const isSignificant = pValue < 0.05;
    const isTrending = pValue < 0.1;
    const direction = higherIsBetter 
      ? original > permutedMean 
      : original < permutedMean;
    const isValid = isSignificant && direction;

    return (
      <div className="bg-muted/50 rounded-lg p-5 border border-border">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-medium text-foreground">{label}</h4>
          <div className={cn(
            "flex items-center gap-2 px-3 py-1 rounded-full text-sm font-medium",
            isValid && "bg-accent/20 text-accent",
            !isValid && isTrending && "bg-warning/20 text-warning",
            !isValid && !isTrending && "bg-destructive/20 text-destructive"
          )}>
            {isValid ? (
              <>
                <CheckCircle2 className="w-4 h-4" />
                Significant
              </>
            ) : isTrending ? (
              <>
                <AlertTriangle className="w-4 h-4" />
                Trending
              </>
            ) : (
              <>
                <XCircle className="w-4 h-4" />
                Not Significant
              </>
            )}
          </div>
        </div>

        <div className="grid grid-cols-3 gap-4">
          <div>
            <p className="text-xs text-muted-foreground mb-1">Original Model</p>
            <p className="text-xl font-bold text-primary">
              {(original * (label.includes("AUROC") ? 100 : 100)).toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground mb-1">Permuted (Mean ± SD)</p>
            <p className="text-xl font-bold text-muted-foreground">
              {(permutedMean * 100).toFixed(1)}% ± {(permutedSd * 100).toFixed(1)}%
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground mb-1">P-Value</p>
            <p className={cn(
              "text-xl font-bold",
              pValue < 0.05 && "text-accent",
              pValue >= 0.05 && pValue < 0.1 && "text-warning",
              pValue >= 0.1 && "text-destructive"
            )}>
              {pValue < 0.001 ? "< 0.001" : pValue.toFixed(3)}
            </p>
          </div>
        </div>

        <div className="mt-4 pt-4 border-t border-border/50">
          <p className="text-sm text-muted-foreground">
            {higherIsBetter
              ? `Original ${label} is ${((original - permutedMean) * 100).toFixed(1)} percentage points ${original > permutedMean ? "higher" : "lower"} than permuted average.`
              : `Original ${label} is ${((permutedMean - original) * 100).toFixed(1)} percentage points ${original < permutedMean ? "lower" : "higher"} than permuted average.`
            }
          </p>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-card rounded-xl p-6 border border-border">
      <div className="mb-6">
        <h3 className="text-lg font-semibold mb-2">Permutation Testing</h3>
        <p className="text-sm text-muted-foreground">
          Validates that model performance is not due to random chance by comparing original results 
          to models trained on randomly shuffled labels.
        </p>
      </div>

      <div className="space-y-4">
        {renderMetric(
          "Random Forest OOB Error",
          permutation.rf_oob_error.original,
          permutation.rf_oob_error.permuted_mean,
          permutation.rf_oob_error.permuted_sd,
          permutation.rf_oob_error.p_value,
          false // Lower OOB error is better
        )}

        {renderMetric(
          "Random Forest AUROC",
          permutation.rf_auroc.original,
          permutation.rf_auroc.permuted_mean,
          permutation.rf_auroc.permuted_sd,
          permutation.rf_auroc.p_value,
          true // Higher AUROC is better
        )}
      </div>

      <div className="mt-6 p-4 bg-info/10 rounded-lg border border-info/20">
        <p className="text-sm text-info">
          <strong>Interpretation:</strong> A p-value &lt; 0.05 indicates the model's performance 
          is statistically significantly better than random chance, confirming the robustness 
          of the findings (Li et al. 2022).
        </p>
      </div>
    </div>
  );
}
