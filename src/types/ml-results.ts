export interface MetricStats {
  mean: number;
  sd: number;
  median: number;
  q25: number;
  q75: number;
  min: number;
  max: number;
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

export interface PermutationMetric {
  permuted_mean: number;
  permuted_sd: number;
  original: number;
  p_value: number;
}

export interface PermutationTesting {
  rf_oob_error: PermutationMetric;
  rf_auroc: PermutationMetric;
}

export interface ProfileRanking {
  sample_index: number;
  actual_class: string;
  ensemble_probability: number;
  predicted_class: string;
  confidence: number;
  correct: boolean;
  rank: number;
  top_profile: boolean;
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
}

export interface MLResultsMetadata {
  generated_at: string;
  config: MLResultsConfig;
  r_version: string;
}

export interface MLResults {
  metadata: MLResultsMetadata;
  model_performance: ModelPerformance;
  feature_importance: FeatureImportance[];
  permutation_testing: PermutationTesting | null;
  profile_ranking: {
    top_profiles: ProfileRanking[];
    all_rankings: ProfileRanking[];
  } | null;
  selected_features: string[];
}
