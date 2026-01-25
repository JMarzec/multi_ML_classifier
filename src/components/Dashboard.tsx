import { useState, useMemo } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Button } from "@/components/ui/button";
import {
  BarChart3,
  Brain,
  Shuffle,
  Users,
  Settings,
  ArrowLeft,
  Grid3X3,
  TrendingUp,
  Database,
  Beaker,
  LineChart,
  Code,
  Layers,
  Target,
  Grip,
  GitBranch,
  Info,
  BoxSelect,
  Activity,
  Dna,
  Filter,
  Heart,
} from "lucide-react";
import accelBioLogo from "@/assets/accelbio-logo.png";
import { MetricCard } from "./MetricCard";
import { ModelComparisonChart } from "./ModelComparisonChart";
import { FeatureImportanceChart } from "./FeatureImportanceChart";
import { FeatureImportanceStabilityTab } from "./FeatureImportanceStabilityTab";
import { PermutationTestingPanel } from "./PermutationTestingPanel";
import { ProfileRankingTable } from "./ProfileRankingTable";
import { ConfigSummary } from "./ConfigSummary";
import { ConfusionMatrixChart, ConfusionMatrixExplanation } from "./ConfusionMatrixChart";
import { ROCCurveChart } from "./ROCCurveChart";
import { ReportExport } from "./ReportExport";
import { SurvivalReportExport } from "./SurvivalReportExport";
import { ClinicalReportExport } from "./ClinicalReportExport";
import { ThemeToggle } from "./ThemeToggle";
import { DataPreprocessingTab } from "./DataPreprocessingTab";
import { SamplePredictionTab } from "./SamplePredictionTab";
import { LearningCurveTab } from "./LearningCurveTab";
import { ModelExportTab } from "./ModelExportTab";
import { BatchResultsTab } from "./BatchResultsTab";
import { CalibrationCurveTab } from "./CalibrationCurveTab";
import { ClusteringVisualizationTab } from "./ClusteringVisualizationTab";
import { CVFoldVisualizationTab } from "./CVFoldVisualizationTab";
import { FeatureExpressionBoxplotTab } from "./FeatureExpressionBoxplotTab";
import { PermutationDistributionTab } from "./PermutationDistributionTab";
import { MLMethodInfoPanel } from "./MLMethodInfoPanel";
import { FeatureSelectionVisualization } from "./FeatureSelectionVisualization";
import { RiskScoreDistributionTab } from "./RiskScoreDistributionTab";
import { SurvivalAnalysisTab } from "./SurvivalAnalysisTab";
import { GeneSignaturesTab } from "./GeneSignaturesTab";
import type { MLResults } from "@/types/ml-results";

interface DashboardProps {
  data: MLResults;
  onReset: () => void;
}

export function Dashboard({ data, onReset }: DashboardProps) {
  const [metricFilter, setMetricFilter] = useState<"accuracy" | "auroc" | "f1_score" | "balanced_accuracy">("auroc");

  // Compute effective ensemble accuracy using soft-vote fallback logic
  // If hard_vote accuracy is 0 or NaN but soft_vote is valid, prefer soft_vote
  const effectiveEnsembleStats = useMemo(() => {
    const softVote = data.model_performance.soft_vote;
    const hardVote = data.model_performance.hard_vote;

    // Check if soft_vote accuracy is valid
    const softAcc = softVote?.accuracy?.mean;
    const hardAcc = hardVote?.accuracy?.mean;

    // Prefer soft_vote if hard_vote is missing/zero
    if (softAcc && softAcc > 0) {
      return { stats: softVote.accuracy!, label: "Ensemble Accuracy (Soft)" };
    }
    if (hardAcc && hardAcc > 0) {
      return { stats: hardVote!.accuracy!, label: "Ensemble Accuracy (Hard)" };
    }
    // Fallback to soft_vote even if 0 (display placeholder)
    return softVote?.accuracy ? { stats: softVote.accuracy, label: "Ensemble Accuracy" } : null;
  }, [data.model_performance]);

  const bestModel = Object.entries(data.model_performance)
    .filter(([, metrics]) => metrics?.auroc && metrics.auroc.mean > 0)
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
      <header className="sticky top-0 z-50 bg-background/70 backdrop-blur-lg border-b border-border">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button variant="ghost" size="icon" onClick={onReset} aria-label="Back">
                <ArrowLeft className="w-5 h-5" />
              </Button>

              <div className="flex items-center gap-3">
                <img
                  src={accelBioLogo}
                  alt="Co-Lab AccelBio logo"
                  className="h-9 w-auto"
                />
                <div>
                  <h1 className="text-xl font-bold gradient-text">ML Classifier Results</h1>
                  <p className="text-sm text-muted-foreground">
                    Generated {new Date(data.metadata.generated_at).toLocaleDateString()}
                  </p>
                </div>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <ThemeToggle />
              <ReportExport data={data} />
              <SurvivalReportExport data={data} />
              <ClinicalReportExport data={data} />
            </div>
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
          
          {effectiveEnsembleStats && (
            <MetricCard
              title={effectiveEnsembleStats.label}
              stats={effectiveEnsembleStats.stats}
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

        <ConfigSummary 
          metadata={data.metadata} 
          selectedFeatures={data.selected_features || []} 
        />

        {/* Main Tabs - Enhanced visibility */}
        <Tabs defaultValue="data" className="space-y-6">
          <TabsList className="bg-muted/70 p-1.5 flex-wrap h-auto gap-1 border border-border rounded-xl shadow-sm">
            <TabsTrigger value="data" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Database className="w-4 h-4 mr-2" />
              Data
            </TabsTrigger>
            <TabsTrigger value="performance" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <BarChart3 className="w-4 h-4 mr-2" />
              Performance
            </TabsTrigger>
            <TabsTrigger value="confusion" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Grid3X3 className="w-4 h-4 mr-2" />
              Confusion
            </TabsTrigger>
            <TabsTrigger value="roc" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <TrendingUp className="w-4 h-4 mr-2" />
              ROC
            </TabsTrigger>
            <TabsTrigger value="features" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Brain className="w-4 h-4 mr-2" />
              Features
            </TabsTrigger>
            <TabsTrigger value="feature-selection" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Filter className="w-4 h-4 mr-2" />
              Selection
            </TabsTrigger>
            <TabsTrigger value="signatures" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Dna className="w-4 h-4 mr-2" />
              Signatures
            </TabsTrigger>
            <TabsTrigger value="boxplots" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <BoxSelect className="w-4 h-4 mr-2" />
              Expression
            </TabsTrigger>
            <TabsTrigger value="permutation" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Shuffle className="w-4 h-4 mr-2" />
              Permutation
            </TabsTrigger>
            <TabsTrigger value="perm-dist" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Activity className="w-4 h-4 mr-2" />
              Distributions
            </TabsTrigger>
            <TabsTrigger value="rankings" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Users className="w-4 h-4 mr-2" />
              Rankings
            </TabsTrigger>
            <TabsTrigger value="risk-scores" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Activity className="w-4 h-4 mr-2" />
              Risk Scores
            </TabsTrigger>
            <TabsTrigger value="prediction" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Beaker className="w-4 h-4 mr-2" />
              Predict
            </TabsTrigger>
            <TabsTrigger value="learning" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <LineChart className="w-4 h-4 mr-2" />
              Learning
            </TabsTrigger>
            <TabsTrigger value="export" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Code className="w-4 h-4 mr-2" />
              Export
            </TabsTrigger>
            <TabsTrigger value="batch" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Layers className="w-4 h-4 mr-2" />
              Batch
            </TabsTrigger>
            <TabsTrigger value="calibration" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Target className="w-4 h-4 mr-2" />
              Calibration
            </TabsTrigger>
            <TabsTrigger value="clustering" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Grip className="w-4 h-4 mr-2" />
              Clustering
            </TabsTrigger>
            <TabsTrigger value="cv-folds" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <GitBranch className="w-4 h-4 mr-2" />
              CV Folds
            </TabsTrigger>
            <TabsTrigger value="survival" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Heart className="w-4 h-4 mr-2" />
              Survival
            </TabsTrigger>
            <TabsTrigger value="ml-info" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Info className="w-4 h-4 mr-2" />
              ML Guide
            </TabsTrigger>
            <TabsTrigger value="config" className="data-[state=active]:bg-primary data-[state=active]:text-primary-foreground data-[state=active]:shadow-md font-medium">
              <Settings className="w-4 h-4 mr-2" />
              Config
            </TabsTrigger>
          </TabsList>

          <TabsContent value="data">
            <DataPreprocessingTab data={data} />
          </TabsContent>

          <TabsContent value="performance" className="space-y-6">
            <div className="flex gap-2 mb-4 flex-wrap">
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

          <TabsContent value="confusion" className="space-y-6">
            <ConfusionMatrixExplanation />
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {Object.entries(data.model_performance)
                .filter(([, metrics]) => metrics?.confusion_matrix)
                .map(([model, metrics]) => (
                  <ConfusionMatrixChart
                    key={model}
                    data={metrics!.confusion_matrix!}
                    modelName={modelLabels[model] || model}
                  />
                ))}
            </div>
            {!Object.values(data.model_performance).some(m => m?.confusion_matrix) && (
              <div className="bg-card rounded-xl p-12 border border-border text-center">
                <Grid3X3 className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">No confusion matrix data available. Try the demo mode or ensure your R script exports confusion matrices.</p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="roc">
            <ROCCurveChart performance={data.model_performance} />
          </TabsContent>

          <TabsContent value="features" className="space-y-6">
            {data.feature_importance && data.feature_importance.length > 0 ? (
              <FeatureImportanceChart features={data.feature_importance} />
            ) : (
              <div className="bg-card rounded-xl p-12 border border-border text-center">
                <Brain className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">No feature importance data available</p>
              </div>
            )}

            <FeatureImportanceStabilityTab data={data} />
          </TabsContent>

          <TabsContent value="feature-selection">
            <FeatureSelectionVisualization data={data} />
          </TabsContent>

          <TabsContent value="signatures">
            <GeneSignaturesTab data={data} />
          </TabsContent>

          <TabsContent value="boxplots">
            <FeatureExpressionBoxplotTab data={data} />
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

          <TabsContent value="risk-scores">
            {data.profile_ranking?.all_rankings ? (
              <RiskScoreDistributionTab 
                rankings={data.profile_ranking.all_rankings}
              />
            ) : (
              <div className="bg-card rounded-xl p-12 border border-border text-center">
                <Activity className="w-12 h-12 text-muted-foreground mx-auto mb-4" />
                <p className="text-muted-foreground">No risk score data available. Run the R script with profile ranking enabled.</p>
              </div>
            )}
          </TabsContent>

          <TabsContent value="perm-dist">
            <PermutationDistributionTab data={data} />
          </TabsContent>

          <TabsContent value="prediction">
            <SamplePredictionTab data={data} />
          </TabsContent>

          <TabsContent value="learning">
            <LearningCurveTab data={data} />
          </TabsContent>

          <TabsContent value="export">
            <ModelExportTab data={data} />
          </TabsContent>

          <TabsContent value="batch">
            <BatchResultsTab />
          </TabsContent>

          <TabsContent value="calibration">
            <CalibrationCurveTab data={data} />
          </TabsContent>

          <TabsContent value="clustering">
            <ClusteringVisualizationTab data={data} />
          </TabsContent>

          <TabsContent value="cv-folds">
            <CVFoldVisualizationTab data={data} />
          </TabsContent>

          <TabsContent value="survival">
            <SurvivalAnalysisTab data={data} />
          </TabsContent>

          <TabsContent value="ml-info">
            <MLMethodInfoPanel />
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
