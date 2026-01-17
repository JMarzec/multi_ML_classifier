export interface MetricStats {
  mean: number;
  sd: number;
  median: number;
  q25: number;
  q75: number;
  min: number;
  max: number;
}

export interface ConfusionMatrixData {
  tp: number;
  tn: number;
  fp: number;
  fn: number;
}

export interface ROCPoint {
  fpr: number;
  tpr: number;
  threshold?: number;
}

export interface ModelMetrics {
  accuracy?: MetricStats;
  sensitivity?: MetricStats;
  specificity?: MetricStats;
  precision?: MetricStats;
  f1_score?: MetricStats;
  balanced_accuracy?: MetricStats;
  auroc?: MetricStats;
  kappa?: MetricStats;
  confusion_matrix?: ConfusionMatrixData;
  roc_curve?: ROCPoint[];
}

export interface ModelPerformance {
  rf?: ModelMetrics;
  svm?: ModelMetrics;
  xgboost?: ModelMetrics;
  knn?: ModelMetrics;
  mlp?: ModelMetrics;
  hard_vote?: ModelMetrics;
  soft_vote?: ModelMetrics;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface FeatureImportanceStability {
  feature: string;
  mean_rank: number;
  sd_rank: number;
  top_n_frequency: number;
}

export interface CalibrationCurvePoint {
  bin: string;
  mean_pred: number;
  frac_pos: number;
  n: number;
  bin_center: number;
  mean_pred_pct: number;
  frac_pos_pct: number;
}

export type CalibrationCurves = Record<string, CalibrationCurvePoint[]>;

export interface ClusteringPoint {
  x: number;
  y: number;
  sample_id: string;
  actual_class: string;
}

export interface ClusteringExport {
  pca?: {
    points: ClusteringPoint[];
    variance_explained?: {
      pc1: number;
      pc2: number;
    };
  } | null;
  tsne?: {
    points: ClusteringPoint[];
  } | null;
  umap?: {
    points: ClusteringPoint[];
  } | null;
}

export interface PermutationMetric {
  permuted_mean: number;
  permuted_sd: number;
  original: number;
  p_value: number;
}

export interface PermutationDistribution {
  auroc: number[];
  accuracy: number[];
}

export interface PermutationDistributions {
  rf?: PermutationDistribution;
  svm?: PermutationDistribution;
  xgboost?: PermutationDistribution;
  knn?: PermutationDistribution;
  mlp?: PermutationDistribution;
  soft_vote?: PermutationDistribution;
}

export interface ActualDistributions {
  rf?: PermutationDistribution;
  svm?: PermutationDistribution;
  xgboost?: PermutationDistribution;
  knn?: PermutationDistribution;
  mlp?: PermutationDistribution;
  soft_vote?: PermutationDistribution;
}

export interface PermutationTesting {
  rf_oob_error: PermutationMetric;
  rf_auroc: PermutationMetric;
}

export interface ProfileRanking {
  sample_index: number;
  sample_id?: string;
  actual_class: string;
  ensemble_probability: number;
  predicted_class: string;
  confidence: number;
  correct: boolean;
  rank: number;
  top_profile: boolean;
  risk_score_class_0?: number;
  risk_score_class_1?: number;
}

export interface FeatureBoxplotClassStats {
  class: string;
  min: number;
  q1: number;
  median: number;
  q3: number;
  max: number;
  mean: number;
  n: number;
}

// ============================================================================
// SURVIVAL ANALYSIS TYPES
// ============================================================================

export interface SurvivalStats {
  logrank_p: number;
  cox_hr: number;
  cox_hr_lower: number;
  cox_hr_upper: number;
  cox_p: number;
}

export interface KaplanMeierPoint {
  time: number;
  surv: number;
  lower: number;
  upper: number;
  n_risk: number;
  n_event: number;
  n_censor: number;
  strata?: string;
}

export interface PerGeneSurvival {
  gene: string;
  logrank_p: number;
  cox_hr: number;
  cox_hr_lower: number;
  cox_hr_upper: number;
  cox_p: number;
  high_median_surv: number | null;
  low_median_surv: number | null;
}

export interface ModelRiskScoreSurvival {
  model: string;
  stats: SurvivalStats;
  km_curve_high: KaplanMeierPoint[];
  km_curve_low: KaplanMeierPoint[];
}

export interface SurvivalAnalysis {
  time_variable: string;
  event_variable: string;
  per_gene: PerGeneSurvival[];
  model_risk_scores?: ModelRiskScoreSurvival[];
}

export interface PreprocessingStats {
  original_samples: number;
  original_features: number;
  missing_values: number;
  missing_pct: number;
  class_distribution: Record<string, number>;
  constant_features_removed: number;
  // Train/Test split info (for CV mode)
  cv_folds?: number;
  cv_repeats?: number;
  train_samples_per_fold?: number;
  test_samples_per_fold?: number;
  train_class_distribution?: Record<string, number>;
  test_class_distribution?: Record<string, number>;
  // Full training mode flag
  full_training_mode?: boolean;
}

export interface MLResultsConfig {
  target_variable: string;
  seed: number;
  n_folds: number;
  n_repeats: number;
  top_percent: number;
  feature_selection_method: string;
  max_features: number;
  n_permutations: number;
  rf_ntree: number;
  xgb_nrounds: number;
  expression_matrix_file?: string;
  annotation_file?: string;
  svm_kernel?: string;
  knn_k?: number;
  analysis_mode?: string;
}

export interface MLResultsMetadata {
  generated_at: string;
  config: MLResultsConfig;
  r_version: string;
}

export interface MLResults {
  metadata: MLResultsMetadata;
  preprocessing?: PreprocessingStats | null;
  model_performance: ModelPerformance;
  feature_importance: FeatureImportance[];
  feature_importance_stability?: FeatureImportanceStability[] | null;
  feature_boxplot_stats?: Record<string, FeatureBoxplotClassStats[]> | null;
  calibration_curves?: CalibrationCurves | null;
  clustering?: ClusteringExport | null;
  permutation_testing: PermutationTesting | null;
  permutation_distributions?: PermutationDistributions | null;
  actual_distributions?: ActualDistributions | null;
  profile_ranking: {
    top_profiles: ProfileRanking[];
    all_rankings: ProfileRanking[];
  } | null;
  selected_features: string[];
  survival_analysis?: SurvivalAnalysis | null;
}
