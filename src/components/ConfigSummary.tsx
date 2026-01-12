import { Calendar, Settings, Database, Layers, Shuffle, GitBranch } from "lucide-react";
import type { MLResultsMetadata } from "@/types/ml-results";

interface ConfigSummaryProps {
  metadata: MLResultsMetadata;
  selectedFeatures: string[];
}

export function ConfigSummary({ metadata, selectedFeatures }: ConfigSummaryProps) {
  const { config, generated_at, r_version } = metadata;

  return (
    <div className="bg-card rounded-xl p-6 border border-border">
      <div className="flex items-center gap-3 mb-4">
        <Settings className="w-5 h-5 text-primary" />
        <h3 className="text-lg font-semibold">Analysis Configuration</h3>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="flex items-start gap-3">
          <Calendar className="w-4 h-4 text-muted-foreground mt-0.5" />
          <div>
            <p className="text-xs text-muted-foreground">Generated</p>
            <p className="text-sm font-medium">
              {new Date(generated_at).toLocaleString()}
            </p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <Database className="w-4 h-4 text-muted-foreground mt-0.5" />
          <div>
            <p className="text-xs text-muted-foreground">Cross-Validation</p>
            <p className="text-sm font-medium">
              {config.n_folds}-fold × {config.n_repeats} repeats
            </p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <GitBranch className="w-4 h-4 text-muted-foreground mt-0.5" />
          <div>
            <p className="text-xs text-muted-foreground">Feature Selection</p>
            <p className="text-sm font-medium capitalize">
              {config.feature_selection_method}
            </p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <Layers className="w-4 h-4 text-muted-foreground mt-0.5" />
          <div>
            <p className="text-xs text-muted-foreground">Selected Features</p>
            <p className="text-sm font-medium">{selectedFeatures.length}</p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <Shuffle className="w-4 h-4 text-muted-foreground mt-0.5" />
          <div>
            <p className="text-xs text-muted-foreground">Permutations</p>
            <p className="text-sm font-medium">{config.n_permutations}</p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <Settings className="w-4 h-4 text-muted-foreground mt-0.5" />
          <div>
            <p className="text-xs text-muted-foreground">Random Seed</p>
            <p className="text-sm font-medium">{config.seed}</p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <Database className="w-4 h-4 text-muted-foreground mt-0.5" />
          <div>
            <p className="text-xs text-muted-foreground">Top Profiles</p>
            <p className="text-sm font-medium">{config.top_percent}%</p>
          </div>
        </div>

        <div className="flex items-start gap-3">
          <Settings className="w-4 h-4 text-muted-foreground mt-0.5" />
          <div>
            <p className="text-xs text-muted-foreground">R Version</p>
            <p className="text-sm font-medium text-muted-foreground">{r_version}</p>
          </div>
        </div>
      </div>

      {selectedFeatures.length > 0 && (
        <div className="mt-4 pt-4 border-t border-border">
          <p className="text-xs text-muted-foreground mb-2">Selected Features</p>
          <div className="flex flex-wrap gap-2">
            {selectedFeatures.slice(0, 15).map((feature) => (
              <span
                key={feature}
                className="px-2 py-1 bg-muted rounded text-xs font-mono"
              >
                {feature}
              </span>
            ))}
            {selectedFeatures.length > 15 && (
              <span className="px-2 py-1 bg-primary/20 text-primary rounded text-xs">
                +{selectedFeatures.length - 15} more
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
