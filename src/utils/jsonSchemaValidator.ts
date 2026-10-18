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

  if (!isRecord(data)) {
    errors.push({
      field: "root",
      message: "Invalid JSON: Expected an object at the root level",
      severity: "error",
    });
    return { isValid: false, errors, warnings };
  }

  const obj = data as Record<string, unknown>;

  validateStructures(obj, errors);

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
  if (!isRecord(metadata)) {
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
    if (!isRecord(meta.config)) {
      errors.push({field: "metadata.config", message: "Expected a configuration object", severity: "error"});
      return;
    }
    const config = meta.config;
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
  if (!isRecord(perf)) {
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
    const modelData = models[modelName];
    if (!isRecord(modelData)) {
      errors.push({field: `model_performance.${modelName}`, message: "Expected a model metrics object, not null", severity: "error"});
      return;
    }
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
        const first = perGene[0];
        if (!isRecord(first)) return;
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
    const first = (rank.all_rankings as unknown[])[0];
    if (!isRecord(first)) return;
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


export function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function validateStructures(obj: Record<string, unknown>, errors: ValidationError[]) {
  const fail = (field: string, message: string) => errors.push({ field, message, severity: "error" });
  const array = (value: unknown, field: string, item: (v: unknown) => boolean) => {
    if (value == null) return;
    if (!Array.isArray(value)) { fail(field, "Expected an array"); return; }
    value.forEach((v, i) => { if (!item(v)) fail(`${field}[${i}]`, "Invalid array entry"); });
  };
  array(obj.selected_features, "selected_features", v => typeof v === "string");
  array(obj.feature_importance, "feature_importance", v => isRecord(v) && typeof v.feature === "string" && typeof v.importance === "number" && Number.isFinite(v.importance));
  if (obj.profile_ranking != null) {
    if (!isRecord(obj.profile_ranking)) fail("profile_ranking", "Expected a rankings object");
    else for (const key of ["all_rankings", "top_profiles"]) {
      array(obj.profile_ranking[key], `profile_ranking.${key}`, v => isRecord(v) &&
        (typeof v.sample_id === "string" || typeof v.sample_index === "number") &&
        (typeof v.actual_class === "string" || typeof v.actual_class === "number") &&
        (typeof v.predicted_class === "string" || typeof v.predicted_class === "number"));
    }
  }
  if (isRecord(obj.model_performance)) for (const [name, model] of Object.entries(obj.model_performance)) {
    if (!isRecord(model)) { fail(`model_performance.${name}`, "Expected a model metrics object"); continue; }
    for (const metric of ["accuracy", "auroc", "sensitivity", "specificity", "precision", "f1_score", "balanced_accuracy", "kappa"]) {
      const value = model[metric];
      if (value != null && (!isRecord(value) || (value.mean != null && value.mean !== "NA" && (typeof value.mean !== "number" || !Number.isFinite(value.mean))))) fail(`model_performance.${name}.${metric}`, "Expected a metrics summary with a numeric mean or null");
    }
    array(model.roc_curve, `model_performance.${name}.roc_curve`, v => isRecord(v) && typeof v.fpr === "number" && typeof v.tpr === "number");
    const cm = model.confusion_matrix;
    if (cm != null && (!isRecord(cm) || ["tp", "tn", "fp", "fn"].some(k => typeof cm[k] !== "number"))) fail(`model_performance.${name}.confusion_matrix`, "Expected numeric TP, TN, FP and FN counts");
  }
  if (obj.survival_analysis != null) {
    if (!isRecord(obj.survival_analysis)) fail("survival_analysis", "Expected a survival analysis object");
    else for (const key of ["per_gene", "model_risk_scores"]) {
      const value = obj.survival_analysis[key];
      if (value != null && !Array.isArray(value) && !isRecord(value)) fail(`survival_analysis.${key}`, "Expected an array or keyed object");
      else if (value != null) {
        const entries = Array.isArray(value) ? value : Object.values(value);
        entries.forEach((v, i) => { if (!isRecord(v)) fail(`survival_analysis.${key}[${i}]`, "Expected a survival statistics object"); });
      }
    }
  }
  for (const key of ["clustering", "preprocessing", "calibration_curves", "feature_boxplot_stats"]) {
    if (obj[key] != null && !isRecord(obj[key])) fail(key, "Expected an object");
  }
  if (isRecord(obj.clustering)) for (const [method, embedding] of Object.entries(obj.clustering)) {
    if (embedding == null) continue;
    if (!isRecord(embedding)) { fail(`clustering.${method}`, "Expected an embedding object"); continue; }
    array(embedding.points, `clustering.${method}.points`, v => isRecord(v) && typeof v.x === "number" && typeof v.y === "number" && typeof v.sample_id === "string");
  }
}

/** The only entry point for results ingestion, including stateless tools. */
export function parseMLResults(text: string): MLResults {
  let raw: unknown;
  try { raw = JSON.parse(text); } catch { throw new Error("Invalid JSON syntax. Please check the file contents."); }
  const result = validateMLResultsSchema(raw);
  if (!result.isValid) throw new Error(result.errors.map(e => `${e.field}: ${e.message}`).join("\n"));
  // Structural validation above establishes the external JSON contract.
  const data = raw as MLResults;
  for (const model of Object.values(data.model_performance)) {
    if (!model) continue;
    for (const [key, value] of Object.entries(model)) {
      if (isRecord(value) && (value.mean === "NA" || value.mean === null)) {
        delete model[key];
      }
    }
  }
  data.feature_importance ??= [];
  data.selected_features ??= [];
  if (data.profile_ranking) {
    data.profile_ranking.all_rankings ??= [];
    data.profile_ranking.top_profiles ??= [];
    for (const r of [...data.profile_ranking.all_rankings, ...data.profile_ranking.top_profiles]) {
      r.actual_class = String(r.actual_class);
      r.predicted_class = String(r.predicted_class);
    }
  }
  return data;
}
