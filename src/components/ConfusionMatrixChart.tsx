import { confusionMetrics, metricPercent } from "@/utils/analytics";
import { useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { ConfusionMatrixData } from "@/types/ml-results";

interface ConfusionMatrixChartProps {
  data: ConfusionMatrixData;
  modelName: string;
}

export function ConfusionMatrixChart({ data, modelName }: ConfusionMatrixChartProps) {
  const { tp, tn, fp, fn } = data;
  const total = tp + tn + fp + fn;

  const metrics = useMemo(() => {
    const raw = confusionMetrics({tp, tn, fp, fn});
    return Object.fromEntries(Object.entries(raw).map(([key, value]) => [key, metricPercent(value)]));
  }, [tp, tn, fp, fn, total]);

  return (
    <div className="bg-card rounded-xl p-5 border border-border">
      <h4 className="font-semibold text-lg mb-4 text-center">{modelName}</h4>
      
      {/* Matrix Grid */}
      <div className="relative">
        {/* Axis Labels */}
        <div className="flex items-center justify-center mb-2">
          <span className="text-xs text-muted-foreground font-medium">Predicted</span>
        </div>
        
        <div className="flex">
          {/* Y-axis label */}
          <div className="flex items-center justify-center w-8">
            <span className="text-xs text-muted-foreground font-medium -rotate-90 whitespace-nowrap">Actual</span>
          </div>
          
          <div className="flex-1">
            {/* Column headers */}
            <div className="grid grid-cols-2 gap-1 mb-1 ml-16">
              <div className="text-center text-xs text-muted-foreground">Negative</div>
              <div className="text-center text-xs text-muted-foreground">Positive</div>
            </div>
            
            <div className="flex">
              {/* Row labels */}
              <div className="flex flex-col justify-around w-16 pr-2">
                <div className="text-right text-xs text-muted-foreground">Positive</div>
                <div className="text-right text-xs text-muted-foreground">Negative</div>
              </div>
              
              {/* Matrix cells */}
              <div className="grid grid-cols-2 gap-1 flex-1">
                {/* Row 1: Actual Positive */}
                <div className="aspect-square bg-warning/20 rounded-lg flex flex-col items-center justify-center border border-warning/30">
                  <span className="text-2xl font-bold">{fn}</span>
                  <span className="text-xs text-muted-foreground">FN</span>
                </div>
                <div className="aspect-square bg-success/20 rounded-lg flex flex-col items-center justify-center border border-success/30">
                  <span className="text-2xl font-bold text-success">{tp}</span>
                  <span className="text-xs text-muted-foreground">TP</span>
                </div>
                
                {/* Row 2: Actual Negative */}
                <div className="aspect-square bg-success/20 rounded-lg flex flex-col items-center justify-center border border-success/30">
                  <span className="text-2xl font-bold text-success">{tn}</span>
                  <span className="text-xs text-muted-foreground">TN</span>
                </div>
                <div className="aspect-square bg-warning/20 rounded-lg flex flex-col items-center justify-center border border-warning/30">
                  <span className="text-2xl font-bold">{fp}</span>
                  <span className="text-xs text-muted-foreground">FP</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Derived Metrics */}
      <div className="grid grid-cols-3 gap-2 mt-4 pt-4 border-t border-border">
        <div className="text-center">
          <p className="text-xs text-muted-foreground">Accuracy</p>
          <p className="font-mono font-semibold text-primary">{metrics.accuracy}</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-muted-foreground">Precision</p>
          <p className="font-mono font-semibold">{metrics.precision}</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-muted-foreground">Sensitivity</p>
          <p className="font-mono font-semibold text-success">{metrics.recall}</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-muted-foreground">Specificity</p>
          <p className="font-mono font-semibold text-info">{metrics.specificity}</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-muted-foreground">F1 Score</p>
          <p className="font-mono font-semibold">{metrics.f1}</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-muted-foreground">NPV</p>
          <p className="font-mono font-semibold">{metrics.npv}</p>
        </div>
      </div>
    </div>
  );
}

// Explanation component for the confusion matrix concepts
export function ConfusionMatrixExplanation() {
  return (
    <Card className="mb-6">
      <CardHeader>
        <CardTitle className="text-lg">Understanding the Confusion Matrix</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4 text-sm">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="space-y-3">
            <h4 className="font-semibold text-foreground">Matrix Components</h4>
            <ul className="space-y-2 text-muted-foreground">
              <li>
                <span className="font-semibold text-success">TP (True Positive):</span> Correctly predicted positive cases. 
                The model said "positive" and the actual class was positive.
              </li>
              <li>
                <span className="font-semibold text-success">TN (True Negative):</span> Correctly predicted negative cases.
                The model said "negative" and the actual class was negative.
              </li>
              <li>
                <span className="font-semibold text-warning">FP (False Positive):</span> Type I error. The model incorrectly 
                predicted positive when the actual class was negative. Also called "false alarm".
              </li>
              <li>
                <span className="font-semibold text-warning">FN (False Negative):</span> Type II error. The model incorrectly 
                predicted negative when the actual class was positive. Also called "miss".
              </li>
            </ul>
          </div>

          <div className="space-y-3">
            <h4 className="font-semibold text-foreground">Derived Metrics</h4>
            <ul className="space-y-2 text-muted-foreground">
              <li>
                <span className="font-semibold text-primary">Accuracy:</span> (TP+TN)/(Total) — Overall correctness. 
                Can be misleading with imbalanced classes.
              </li>
              <li>
                <span className="font-semibold text-foreground">Precision:</span> TP/(TP+FP) — Of all positive predictions, 
                how many were correct? Important when FP is costly.
              </li>
              <li>
                <span className="font-semibold text-success">Sensitivity (Recall):</span> TP/(TP+FN) — Of all actual positives, 
                how many did we catch? Critical in medical diagnostics.
              </li>
              <li>
                <span className="font-semibold text-info">Specificity:</span> TN/(TN+FP) — Of all actual negatives, 
                how many did we correctly identify? Important to avoid false alarms.
              </li>
              <li>
                <span className="font-semibold text-foreground">F1 Score:</span> Harmonic mean of Precision and Recall. 
                Balances both metrics.
              </li>
              <li>
                <span className="font-semibold text-foreground">NPV:</span> TN/(TN+FN) — Negative Predictive Value. 
                Of all negative predictions, how many were correct?
              </li>
            </ul>
          </div>
        </div>

        <div className="p-3 bg-muted/30 rounded-lg mt-4">
          <p className="text-muted-foreground">
            <strong className="text-foreground">Clinical Relevance:</strong> In diagnostic applications, 
            <strong className="text-success"> high sensitivity</strong> is crucial to avoid missing disease cases (minimize FN), 
            while <strong className="text-info">high specificity</strong> is important to avoid unnecessary treatments or anxiety 
            from false positives. The optimal balance depends on the clinical context and the relative costs of each error type.
          </p>
        </div>
      </CardContent>
    </Card>
  );
}
