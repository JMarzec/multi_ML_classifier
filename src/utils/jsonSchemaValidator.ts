import type { MLResults } from "@/types/ml-results";

export interface ValidationError {
  field: string;
  message: string;
  severity: "error" | "warning";
}

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
  warnings: ValidationError[];
}

/**
 * Validate the ML Results JSON schema with friendly error messages
 */
export function validateMLResultsSchema(data: unknown): ValidationResult {
  const errors: ValidationError[] = [];
  const warnings: ValidationError[] = [];

  if (!data || typeof data !== "object") {
    errors.push({
      field: "root",
      message: "Invalid JSON: Expected an object at the root level",
      severity: "error",
    });
    return { isValid: false, errors, warnings };
  }

  const obj = data as Record<string, unknown>;

  // Required fields
  if (!obj.metadata) {
    errors.push({
      field: "metadata",
      message: "Missing required field 'metadata'. This should contain analysis configuration.",
      severity: "error",
    });
  } else {
    validateMetadata(obj.metadata, errors, warnings);
  }

  if (!obj.model_performance) {
    errors.push({
      field: "model_performance",
      message: "Missing required field 'model_performance'. This should contain performance metrics for each ML model.",
      severity: "error",
    });
  } else {
    validateModelPerformance(obj.model_performance, errors, warnings);
  }

  if (!obj.feature_importance) {
    warnings.push({
      field: "feature_importance",
      message: "Missing 'feature_importance' array. Feature importance charts will be empty.",
      severity: "warning",
    });
  } else if (!Array.isArray(obj.feature_importance)) {
    errors.push({
      field: "feature_importance",
      message: "'feature_importance' should be an array of objects with 'feature' and 'importance' fields.",
      severity: "error",
    });
  }

  if (!obj.selected_features) {
    warnings.push({
      field: "selected_features",
      message: "Missing 'selected_features' array. Selected features list will be empty.",
      severity: "warning",
    });
  } else if (!Array.isArray(obj.selected_features)) {
    errors.push({
      field: "selected_features",
      message: "'selected_features' should be an array of feature names.",
      severity: "error",
    });
  }

  // Optional but important fields
  if (obj.survival_analysis) {
    validateSurvivalAnalysis(obj.survival_analysis, errors, warnings);
  }

  if (obj.profile_ranking) {
    validateProfileRanking(obj.profile_ranking, errors, warnings);
  }

  if (obj.preprocessing) {
    validatePreprocessing(obj.preprocessing, errors, warnings);
  }

  return {
    isValid: errors.length === 0,
    errors,
    warnings,
  };
}

function validateMetadata(metadata: unknown, errors: ValidationError[], warnings: ValidationError[]) {
  if (typeof metadata !== "object" || metadata === null) {
    errors.push({
      field: "metadata",
      message: "'metadata' should be an object containing configuration details.",
      severity: "error",
    });
    return;
  }

  const meta = metadata as Record<string, unknown>;

  if (!meta.config) {
    errors.push({
      field: "metadata.config",
      message: "Missing 'metadata.config'. This should contain analysis parameters like target_variable, n_folds, etc.",
      severity: "error",
    });
  } else {
    const config = meta.config as Record<string, unknown>;
    if (!config.target_variable) {
      warnings.push({
        field: "metadata.config.target_variable",
        message: "Missing 'target_variable'. Unable to display which variable was used for classification.",
        severity: "warning",
      });
    }
  }

  if (!meta.r_version) {
    warnings.push({
      field: "metadata.r_version",
      message: "Missing 'r_version'. R version information will not be displayed.",
      severity: "warning",
    });
  }

  if (!meta.generated_at) {
    warnings.push({
      field: "metadata.generated_at",
      message: "Missing 'generated_at'. Analysis timestamp will not be displayed.",
      severity: "warning",
    });
  }
}

function validateModelPerformance(perf: unknown, errors: ValidationError[], warnings: ValidationError[]) {
  if (typeof perf !== "object" || perf === null) {
    errors.push({
      field: "model_performance",
      message: "'model_performance' should be an object with model names as keys (rf, svm, xgboost, etc.).",
      severity: "error",
    });
    return;
  }

  const models = perf as Record<string, unknown>;
  const validModels = ["rf", "svm", "xgboost", "knn", "mlp", "hard_vote", "soft_vote"];
  const foundModels = Object.keys(models).filter((k) => validModels.includes(k));

  if (foundModels.length === 0) {
    errors.push({
      field: "model_performance",
      message: "No valid models found. Expected at least one of: rf, svm, xgboost, knn, mlp, hard_vote, soft_vote.",
      severity: "error",
    });
  }

  foundModels.forEach((modelName) => {
    const modelData = models[modelName] as Record<string, unknown>;
    if (!modelData.auroc && !modelData.accuracy) {
      warnings.push({
        field: `model_performance.${modelName}`,
        message: `Model '${modelName}' is missing both 'auroc' and 'accuracy' metrics.`,
        severity: "warning",
      });
    }
  });
}

function validateSurvivalAnalysis(survival: unknown, errors: ValidationError[], warnings: ValidationError[]) {
  if (typeof survival !== "object" || survival === null) {
    warnings.push({
      field: "survival_analysis",
      message: "'survival_analysis' should be an object. Survival analysis tab will be empty.",
      severity: "warning",
    });
    return;
  }

  const surv = survival as Record<string, unknown>;

  if (!surv.time_variable) {
    warnings.push({
      field: "survival_analysis.time_variable",
      message: "Missing 'time_variable'. Should specify the survival time column name.",
      severity: "warning",
    });
  }

  if (!surv.event_variable) {
    warnings.push({
      field: "survival_analysis.event_variable",
      message: "Missing 'event_variable'. Should specify the event/censoring column name.",
      severity: "warning",
    });
  }

  // Validate per_gene - can be array or object
  if (surv.per_gene) {
    const perGene = surv.per_gene;
    if (Array.isArray(perGene)) {
      if (perGene.length === 0) {
        warnings.push({
          field: "survival_analysis.per_gene",
          message: "'per_gene' array is empty. No gene-level survival statistics will be shown.",
          severity: "warning",
        });
      } else {
        // Check first entry for required fields
        const first = perGene[0] as Record<string, unknown>;
        if (!first.gene) {
          errors.push({
            field: "survival_analysis.per_gene[0].gene",
            message: "Per-gene survival entries must have a 'gene' field with the gene name.",
            severity: "error",
          });
        }
        validateSurvivalStats(first, "survival_analysis.per_gene[0]", warnings);
      }
    } else if (typeof perGene === "object" && perGene !== null) {
      // Object format keyed by gene name
      const geneKeys = Object.keys(perGene as object);
      if (geneKeys.length === 0) {
        warnings.push({
          field: "survival_analysis.per_gene",
          message: "'per_gene' object is empty. No gene-level survival statistics will be shown.",
          severity: "warning",
        });
      } else {
        const first = (perGene as Record<string, unknown>)[geneKeys[0]] as Record<string, unknown>;
        if (first && !first.gene) {
          warnings.push({
            field: `survival_analysis.per_gene.${geneKeys[0]}.gene`,
            message: "Per-gene survival entries should have a 'gene' field. Will try to use object key as gene name.",
            severity: "warning",
          });
        }
        if (first) {
          validateSurvivalStats(first, `survival_analysis.per_gene.${geneKeys[0]}`, warnings);
        }
      }
    } else {
      warnings.push({
        field: "survival_analysis.per_gene",
        message: "'per_gene' should be an array or object of gene survival statistics.",
        severity: "warning",
      });
    }
  } else {
    warnings.push({
      field: "survival_analysis.per_gene",
      message: "Missing 'per_gene'. Gene-level survival analysis will not be shown.",
      severity: "warning",
    });
  }

  // Validate model_risk_scores - can be array or object
  if (surv.model_risk_scores) {
    const modelRisk = surv.model_risk_scores;
    if (!Array.isArray(modelRisk) && (typeof modelRisk !== "object" || modelRisk === null)) {
      warnings.push({
        field: "survival_analysis.model_risk_scores",
        message: "'model_risk_scores' should be an array or object. Model risk survival tab will be hidden.",
        severity: "warning",
      });
    }
  }
}

function validateSurvivalStats(obj: Record<string, unknown>, path: string, warnings: ValidationError[]) {
  const requiredStats = ["logrank_p", "cox_hr", "cox_p"];
  requiredStats.forEach((stat) => {
    if (obj[stat] === undefined || obj[stat] === null) {
      warnings.push({
        field: `${path}.${stat}`,
        message: `Missing '${stat}'. This survival statistic will show as 'NA'.`,
        severity: "warning",
      });
    }
  });
}

function validateProfileRanking(ranking: unknown, errors: ValidationError[], warnings: ValidationError[]) {
  if (typeof ranking !== "object" || ranking === null) {
    warnings.push({
      field: "profile_ranking",
      message: "'profile_ranking' should be an object with 'top_profiles' and 'all_rankings' arrays.",
      severity: "warning",
    });
    return;
  }

  const rank = ranking as Record<string, unknown>;

  if (!rank.top_profiles && !rank.all_rankings) {
    warnings.push({
      field: "profile_ranking",
      message: "Missing both 'top_profiles' and 'all_rankings'. Profile ranking table will be empty.",
      severity: "warning",
    });
  }

  if (rank.all_rankings && Array.isArray(rank.all_rankings) && (rank.all_rankings as unknown[]).length > 0) {
    const first = (rank.all_rankings as unknown[])[0] as Record<string, unknown>;
    if (!first.sample_id && first.sample_index === undefined) {
      warnings.push({
        field: "profile_ranking.all_rankings[0]",
        message: "Profile entries should have 'sample_id' or 'sample_index' for identification.",
        severity: "warning",
      });
    }
  }
}

function validatePreprocessing(prep: unknown, errors: ValidationError[], warnings: ValidationError[]) {
  if (typeof prep !== "object" || prep === null) {
    warnings.push({
      field: "preprocessing",
      message: "'preprocessing' should be an object with data statistics.",
      severity: "warning",
    });
    return;
  }

  const p = prep as Record<string, unknown>;

  if (p.class_distribution) {
    if (Array.isArray(p.class_distribution)) {
      warnings.push({
        field: "preprocessing.class_distribution",
        message: "'class_distribution' is an array but should be an object with class names as keys and counts as values (e.g., {\"Class_A\": 50, \"Class_B\": 30}).",
        severity: "warning",
      });
    }
  }
}

/**
 * Format validation results as user-friendly messages
 */
export function formatValidationMessages(result: ValidationResult): string[] {
  const messages: string[] = [];

  if (result.errors.length > 0) {
    messages.push("❌ ERRORS (must fix):");
    result.errors.forEach((e) => {
      messages.push(`  • ${e.field}: ${e.message}`);
    });
  }

  if (result.warnings.length > 0) {
    if (messages.length > 0) messages.push("");
    messages.push("⚠️ WARNINGS (may affect display):");
    result.warnings.forEach((w) => {
      messages.push(`  • ${w.field}: ${w.message}`);
    });
  }

  return messages;
}
