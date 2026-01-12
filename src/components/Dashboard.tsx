import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import { 
  BarChart3, 
  Brain, 
  Shuffle, 
  Users, 
  Settings, 
  Download,
  ArrowLeft 
} from "lucide-react";
import { MetricCard } from "./MetricCard";
import { ModelComparisonChart } from "./ModelComparisonChart";
import { FeatureImportanceChart } from "./FeatureImportanceChart";
import { PermutationTestingPanel } from "./PermutationTestingPanel";
import { ProfileRankingTable } from "./ProfileRankingTable";
import { ConfigSummary } from "./ConfigSummary";
import type { MLResults } from "@/types/ml-results";

interface DashboardProps {
  data: MLResults;
  onReset: () => void;
}

export function Dashboard({ data, onReset }: DashboardProps) {
  const [metricFilter, setMetricFilter] = useState<"accuracy" | "auroc" | "f1_score" | "balanced_accuracy">("auroc");
  
  // Find best model
  const bestModel = Object.entries(data.model_performance)
    .filter(([, metrics]) => metrics?.auroc)
    .sort((a, b) => (b[1]!.auroc!.mean || 0) - (a[1]!.auroc!.mean || 0))[0];

  const modelLabels: Record<string, string> = {
    rf: "Random Forest",
    svm: "SVM",
    xgboost: "XGBoost",
    knn: "KNN",
    mlp: "MLP",
    hard_vote: "Hard Voting",
    soft_vote: "Soft Voting",
  };

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-background/80 backdrop-blur-lg border-b border-border">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button variant="ghost" size="icon" onClick={onReset}>
                <ArrowLeft className="w-5 h-5" />
              </Button>
              <div>
                <h1 className="text-xl font-bold gradient-text">ML Classifier Results</h1>
                <p className="text-sm text-muted-foreground">
                  Generated {new Date(data.metadata.generated_at).toLocaleDateString()}
                </p>
              </div>
            </div>
            
            <Button variant="outline" size="sm">
              <Download className="w-4 h-4 mr-2" />
              Export Report
            </Button>
          </div>
        </div>
      </header>

      <main className="container mx-auto px-4 py-8 space-y-8">
        {/* Summary Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            title="Best Model"
            stats={{
              mean: bestModel?.[1]?.auroc?.mean || 0,
              sd: bestModel?.[1]?.auroc?.sd || 0,
              median: bestModel?.[1]?.auroc?.median || 0,
              q25: bestModel?.[1]?.auroc?.q25 || 0,
              q75: bestModel?.[1]?.auroc?.q75 || 0,
              min: bestModel?.[1]?.auroc?.min || 0,
              max: bestModel?.[1]?.auroc?.max || 0,
            }}
            icon={<Brain className="w-5 h-5" />}
            colorClass="text-accent"
          />
          
          {data.model_performance.soft_vote?.accuracy && (
            <MetricCard
              title="Ensemble Accuracy"
              stats={data.model_performance.soft_vote.accuracy}
              icon={<BarChart3 className="w-5 h-5" />}
              colorClass="text-primary"
            />
          )}
          
          {data.model_performance.rf?.sensitivity && (
            <MetricCard
              title="RF Sensitivity"
              stats={data.model_performance.rf.sensitivity}
              icon={<Users className="w-5 h-5" />}
              colorClass="text-secondary"
            />
          )}
          
          {data.model_performance.rf?.specificity && (
            <MetricCard
              title="RF Specificity"
              stats={data.model_performance.rf.specificity}
              icon={<Shuffle className="w-5 h-5" />}
              colorClass="text-info"
            />
          )}
        </div>

        {/* Best Model Banner */}
        {bestModel && (
          <div className="bg-gradient-to-r from-primary/10 via-secondary/10 to-accent/10 rounded-xl p-6 border border-primary/20">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-muted-foreground mb-1">Best Performing Model</p>
                <h2 className="text-2xl font-bold text-foreground">
                  {modelLabels[bestModel[0]] || bestModel[0]}
                </h2>
              </div>
              <div className="text-right">
                <p className="text-sm text-muted-foreground mb-1">AUROC Score</p>
                <p className="text-3xl font-bold text-primary">
                  {((bestModel[1]?.auroc?.mean || 0) * 100).toFixed(1)}%
                </p>
              </div>
            </div>
          </div>
        )}

        {/* Configuration Summary */}
        <ConfigSummary 
          metadata={data.metadata} 
          selectedFeatures={data.selected_features || []} 
        />

        {/* Main Tabs */}
        <Tabs defaultValue="performance" className="space-y-6">
          <TabsList className="bg-muted/50 p-1">
            <TabsTrigger value="performance" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <BarChart3 className="w-4 h-4 mr-2" />
              Model Performance
            </TabsTrigger>
            <TabsTrigger value="features" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <Brain className="w-4 h-4 mr-2" />
              Feature Importance
            </TabsTrigger>
            <TabsTrigger value="permutation" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <Shuffle className="w-4 h-4 mr-2" />
              Permutation Testing
            </TabsTrigger>
            <TabsTrigger value="rankings" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <Users className="w-4 h-4 mr-2" />
              Profile Rankings
            </TabsTrigger>
            <TabsTrigger value="config" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground">
              <Settings className="w-4 h-4 mr-2" />
              Configuration
            </TabsTrigger>
          </TabsList>

          <TabsContent value="performance" className="space-y-6">
            <div className="flex gap-2 mb-4">
              {(["auroc", "accuracy", "f1_score", "balanced_accuracy"] as const).map((m) => (
                <Button
                  key={m}
                  variant={metricFilter === m ? "default" : "outline"}
                  size="sm"
                  onClick={() => setMetricFilter(m)}
                >
                  {m.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase())}
                </Button>
              ))}
            </div>
            <ModelComparisonChart 
              performance={data.model_performance} 
              metric={metricFilter}
            />
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {Object.entries(data.model_performance)
                .filter(([, metrics]) => metrics?.auroc)
                .map(([model, metrics]) => (
                  <div key={model} className="bg-card rounded-xl p-5 border border-border">
                    <h4 className="font-semibold text-lg mb-4">{modelLabels[model] || model}</h4>
                    <div className="space-y-3">
                      {metrics?.accuracy && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Accuracy</span>
                          <span className="font-mono">{(metrics.accuracy.mean * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {metrics?.auroc && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">AUROC</span>
                          <span className="font-mono text-primary">{(metrics.auroc.mean * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {metrics?.sensitivity && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Sensitivity</span>
                          <span className="font-mono">{(metrics.sensitivity.mean * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {metrics?.specificity && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">Specificity</span>
                          <span className="font-mono">{(metrics.specificity.mean * 100).toFixed(1)}%</span>
                        </div>
                      )}
                      {metrics?.f1_score && (
                        <div className="flex justify-between">
                          <span className="text-muted-foreground">F1 Score</span>
                          <span className="font-mono">{(metrics.f1_score.mean * 100).toFixed(1)}%</span>
                        </div>
                      )}
                    </div>
                  </div>
                ))}
            </div>
          </TabsContent>

          <TabsContent value="features">
            {data.feature_importance && data.feature_importance.length > 0 ? (
              <FeatureImportanceChart features={data.feature_importance} />
            ) : (
              <div className="bg-card rounded-xl p-12 border border-border text-center">
                <Brain className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">No feature importance data available</p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="permutation">
            {data.permutation_testing ? (
              <PermutationTestingPanel permutation={data.permutation_testing} />
            ) : (
              <div className="bg-card rounded-xl p-12 border border-border text-center">
                <Shuffle className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">No permutation testing data available</p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="rankings">
            {data.profile_ranking?.all_rankings ? (
              <ProfileRankingTable 
                rankings={data.profile_ranking.all_rankings}
                topPercent={data.metadata.config.top_percent}
              />
            ) : (
              <div className="bg-card rounded-xl p-12 border border-border text-center">
                <Users className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">No profile ranking data available</p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="config">
            <div className="bg-card rounded-xl p-6 border border-border">
              <h3 className="text-lg font-semibold mb-4">Full Configuration</h3>
              <pre className="bg-muted/50 rounded-lg p-4 overflow-auto text-sm font-mono">
                {JSON.stringify(data.metadata.config, null, 2)}
              </pre>
            </div>
          </TabsContent>
        </Tabs>
      </main>
    </div>
  );
}
