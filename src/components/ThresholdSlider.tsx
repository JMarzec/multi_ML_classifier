import { useMemo } from "react";
import { Slider } from "@/components/ui/slider";
import type { ROCPoint } from "@/types/ml-results";

interface ThresholdSliderProps {
  threshold: number;
  onThresholdChange: (value: number) => void;
  rocCurve: ROCPoint[];
  modelName: string;
  color: string;
}

export function ThresholdSlider({
  threshold,
  onThresholdChange,
  rocCurve,
  modelName,
  color,
}: ThresholdSliderProps) {
  const metrics = useMemo(() => {
    // Find the closest point on the ROC curve based on threshold
    // Threshold affects the trade-off between sensitivity and specificity
    const sortedCurve = [...rocCurve].sort((a, b) => a.fpr - b.fpr);
    
    // Use threshold to interpolate on the curve
    const idx = Math.min(
      Math.floor(threshold * (sortedCurve.length - 1)),
      sortedCurve.length - 1
    );
    
    const point = sortedCurve[idx] || { fpr: 0, tpr: 0 };
    const sensitivity = point.tpr;
    const specificity = 1 - point.fpr;
    const ppv = sensitivity > 0 ? sensitivity / (sensitivity + point.fpr) : 0;
    const npv = specificity > 0 ? specificity / (specificity + (1 - sensitivity)) : 0;
    
    return {
      sensitivity,
      specificity,
      ppv: Math.min(ppv, 1),
      npv: Math.min(npv, 1),
      fpr: point.fpr,
      tpr: point.tpr,
    };
  }, [threshold, rocCurve]);

  return (
    <div className="bg-card rounded-xl p-5 border border-border">
      <div className="flex items-center gap-3 mb-4">
        <div
          className="w-3 h-3 rounded-full"
          style={{ backgroundColor: color }}
        />
        <h4 className="font-semibold">{modelName}</h4>
      </div>

      <div className="space-y-4">
        <div>
          <div className="flex justify-between text-sm mb-2">
            <span className="text-muted-foreground">Decision Threshold</span>
            <span className="font-mono font-semibold">{(threshold * 100).toFixed(0)}%</span>
          </div>
          <Slider
            value={[threshold * 100]}
            onValueChange={([v]) => onThresholdChange(v / 100)}
            min={0}
            max={100}
            step={1}
            className="w-full"
          />
          <div className="flex justify-between text-xs text-muted-foreground mt-1">
            <span>High Sensitivity</span>
            <span>High Specificity</span>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-3 pt-2">
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <p className="text-xs text-muted-foreground mb-1">Sensitivity (TPR)</p>
            <p className="font-mono font-bold text-lg text-primary">
              {(metrics.sensitivity * 100).toFixed(1)}%
            </p>
          </div>
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <p className="text-xs text-muted-foreground mb-1">Specificity (TNR)</p>
            <p className="font-mono font-bold text-lg text-secondary">
              {(metrics.specificity * 100).toFixed(1)}%
            </p>
          </div>
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <p className="text-xs text-muted-foreground mb-1">PPV (Precision)</p>
            <p className="font-mono font-bold text-lg">
              {(metrics.ppv * 100).toFixed(1)}%
            </p>
          </div>
          <div className="bg-muted/30 rounded-lg p-3 text-center">
            <p className="text-xs text-muted-foreground mb-1">NPV</p>
            <p className="font-mono font-bold text-lg">
              {(metrics.npv * 100).toFixed(1)}%
            </p>
          </div>
        </div>

        <div className="text-xs text-muted-foreground bg-muted/20 rounded-lg p-3">
          <p>
            <strong>Trade-off:</strong> Lowering the threshold increases sensitivity
            (catches more positives) but decreases specificity (more false positives).
          </p>
        </div>
      </div>
    </div>
  );
}
