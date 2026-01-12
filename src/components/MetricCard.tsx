import { cn } from "@/lib/utils";
import type { MetricStats } from "@/types/ml-results";

interface MetricCardProps {
  title: string;
  stats: MetricStats | undefined;
  icon?: React.ReactNode;
  colorClass?: string;
  format?: (value: number) => string;
}

export function MetricCard({ 
  title, 
  stats, 
  icon,
  colorClass = "text-primary",
  format = (v) => (v * 100).toFixed(1) + "%"
}: MetricCardProps) {
  if (!stats) return null;

  return (
    <div className="bg-card rounded-xl p-5 border border-border hover:border-primary/30 transition-all duration-300 group">
      <div className="flex items-center gap-3 mb-3">
        {icon && <div className={cn("transition-colors", colorClass)}>{icon}</div>}
        <h3 className="text-sm font-medium text-muted-foreground uppercase tracking-wide">
          {title}
        </h3>
      </div>
      
      <div className={cn("text-3xl font-bold mb-2 transition-colors", colorClass)}>
        {format(stats.mean)}
      </div>
      
      <div className="flex items-center gap-2 text-xs text-muted-foreground">
        <span>±{format(stats.sd)}</span>
        <span className="text-border">|</span>
        <span>Range: {format(stats.min)} - {format(stats.max)}</span>
      </div>
      
      <div className="mt-3 pt-3 border-t border-border/50">
        <div className="flex justify-between text-xs">
          <span className="text-muted-foreground">Q1</span>
          <span className="text-muted-foreground">Median</span>
          <span className="text-muted-foreground">Q3</span>
        </div>
        <div className="flex justify-between text-sm font-medium mt-1">
          <span>{format(stats.q25)}</span>
          <span>{format(stats.median)}</span>
          <span>{format(stats.q75)}</span>
        </div>
      </div>
    </div>
  );
}
