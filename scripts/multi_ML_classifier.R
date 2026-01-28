#!/usr/bin/env Rscript
# =============================================================================
# Multi-Method ML Diagnostic and Prognostic Classifier
# =============================================================================
# Implements various ML methods (RF, SVM, XGBoost, KNN, MLP) with voting
# classifiers, feature selection, and permutation testing for robust
# diagnostic and prognostic prediction.
#
# INPUT FILES:
#   1. Expression matrix (tab-delimited .txt or .tsv)
#      - Rows: FEATURES, Columns: SAMPLES
#      - Column names are sample IDs (no separate sample ID column needed)
#      - First column can optionally be feature names (will be used as row names)
#   2. Sample annotation file (tab-delimited .txt or .tsv)
#      - Must contain sample IDs and target variable column
#
# CONDA ENVIRONMENT:
#   conda env create -f ml_classifier_env.yml
#   conda activate ml_classifier
#
# ML METHOD NOTES:
#   - XGBoost and MLP need larger datasets (>100 samples) for optimal performance
#   - Random Forest, SVM and KNN are more robust on small datasets (<50 samples)
#   - Hard/soft voting ensembles may fail if any base learner collapses
#     (returns NA for all samples) - in such cases, soft voting is preferred
#
# Inspired by:
# - IntelliGenes (https://github.com/drzeeshanahmed/intelligenes)
# - Molecular Classification Analysis (https://github.com/CoLAB-AccelBio/molecular-classification-analysis)
# - Li et al. 2022 permutation strategy (https://pubmed.ncbi.nlm.nih.gov/35292087/)
# =============================================================================

suppressPackageStartupMessages({
  library(optparse)
  library(caret)
  library(randomForest)
  library(e1071)
  library(xgboost)
  library(class)
  library(nnet)
  library(pROC)
  library(jsonlite)
  library(dplyr)
  library(tidyr)
})

# Optional survival analysis libraries
survival_available <- requireNamespace("survival", quietly = TRUE)
survminer_available <- requireNamespace("survminer", quietly = TRUE)

if (survival_available) library(survival)

# =============================================================================
# PARAMTERS
# =============================================================================

MIN_VALID_FOLD_FRACTION <- 0.3   # at least 30% valid folds required
VAR_QUANTILE <- 0.25   # remove lowest 25% variance features
SCALE_DATA_DEFAULT <- FALSE   # data scaling before analysis
MODEL_SCALING <- list(
  rf      = FALSE,
  svm     = TRUE,
  knn     = TRUE,
  mlp     = TRUE,
  xgboost = FALSE,
  dr      = TRUE   # PCA / t-SNE / UMAP
)


option_list <- list(
  make_option(c("-e", "--expr"),
              type = "character",
              help = "Expression matrix file (rows=features, cols=samples)"),
  
  make_option(c("-a", "--annot"),
              type = "character",
              help = "Sample annotation file"),
  
  make_option(c("-t", "--target"),
              type = "character",
              help = "Target variable column name"),
  
  make_option(c("-m", "--mode"),
              type = "character",
              default = "full",
              help = "Analysis mode: full or fast [default: %default]"),
  
  make_option(c("--scale"),
              action = "store_true",
              default = FALSE,
              help = "Scale expression data (Z-score per feature)"),
  
  make_option(c("-o", "--outdir"),
              type = "character",
              default = "results",
              help = "Output directory [default: %default]"),
  
  make_option(c("-s", "--seed"),
              type = "integer",
              default = 42,
              help = "Random seed [default: %default]"),
  
  make_option(c("-f", "--n_folds"),
              type = "integer",
              default = 5,
              help = "Number of folds for CV [default: %default]"),
  
  make_option(c("-r", "--n_repeats"),
              type = "integer",
              default = 5,
              help = "Number of repeats of CV [default: %default]"),
  
  make_option(c("-p", "--n_permutations"),
              type = "integer",
              default = 100,
              help = "Number of permutations [default: %default]"),
  
  # Survival analysis options
  make_option(c("--time"),
              type = "character",
              default = NULL,
              help = "Time-to-event column name for survival analysis"),
  
  make_option(c("--event"),
              type = "character",
              default = NULL,
              help = "Event/censoring column name for survival analysis (1=event, 0=censored)")
)

opt <- parse_args(OptionParser(option_list = option_list))

# Optional: load t-SNE and UMAP libraries if available
tsne_available <- requireNamespace("Rtsne", quietly = TRUE)
umap_available <- requireNamespace("umap", quietly = TRUE)

if (tsne_available) library(Rtsne)
if (umap_available) library(umap)

# =============================================================================
# CONFIGURATION
# =============================================================================

config <- list(
  # Input files (tab-delimited: .txt or .tsv)
  # Expression matrix: rows = features, columns = samples (column names are sample IDs)
  expression_matrix_file = "expression_matrix.txt",  # or .tsv
  annotation_file = "sample_annotation.txt",
  
  # Column name for target variable in annotation file
  target_variable = "diagnosis",
  
  # Analysis mode: "full" (default) or "fast" (for testing, reduced accuracy)
  analysis_mode = "full",  # "full" or "fast"
  
  # Scale expression data (Z-score per feature)
  scale_data = SCALE_DATA_DEFAULT,
  model_scaling <- MODEL_SCALING,
  
  # Model settings
  seed = 42,
  n_folds = 5,
  n_repeats = 10,
  top_percent = 10,
  
  # Feature selection
  feature_selection_method = "stepwise",  # "forward", "backward", "stepwise", "none"
  max_features = 50,
  
  # Permutation testing
  n_permutations = 100,
  
  # RF specific
  rf_ntree = 500,
  rf_mtry = NULL,
  
  # SVM specific
  svm_kernel = "radial",
  svm_cost = 1,
  svm_gamma = NULL,
  
  # XGBoost specific
  xgb_nrounds = 100,
  xgb_max_depth = 6,
  xgb_eta = 0.3,
  
  # KNN specific
  knn_k = 5,
  
  # MLP specific
  mlp_size = 10,
  mlp_decay = 0.01,
  mlp_maxit = 200,
  
  # Batch processing (optional)
  # Set to NULL for single dataset, or list of dataset configs for batch mode
  batch_datasets = NULL,  # Example: list(list(expr="data1.txt", annot="annot1.txt", name="Dataset1"), ...)
  
  # Survival analysis (optional)
  time_variable = NULL,   # Column name for time-to-event
  event_variable = NULL,  # Column name for event status (1=event, 0=censored)
  
  # Output
  output_dir = "./results",
  output_json = NULL  # Will be auto-generated from annotation file if not specified
)

# Override config from command-line arguments if provided
if (!is.null(opt$expr)) config$expression_matrix_file <- opt$expr
if (!is.null(opt$annot)) config$annotation_file <- opt$annot
if (!is.null(opt$target)) config$target_variable <- opt$target
if (!is.null(opt$mode)) config$analysis_mode <- opt$mode
if (!is.null(opt$scale)) config$scale_data <- opt$scale
if (!is.null(opt$outdir)) config$output_dir <- opt$outdir
if (!is.null(opt$seed)) config$seed <- opt$seed
if (!is.null(opt$n_folds)) config$n_folds <- opt$n_folds
if (!is.null(opt$n_repeats)) config$n_repeats <- opt$n_repeats
if (!is.null(opt$n_permutations)) config$n_permutations <- opt$n_permutations
if (!is.null(opt$time)) config$time_variable <- opt$time
if (!is.null(opt$event)) config$event_variable <- opt$event

# Auto-generate output JSON filename from annotation file if not specified
if (is.null(config$output_json)) {
  annot_basename <- basename(config$annotation_file)
  annot_name <- sub("\\.[^.]+$", "", annot_basename)  # Remove extension
  config$output_json <- paste0("ml_results_", annot_name, ".json")
}


# =============================================================================
# FAST MODE CONFIGURATION
# =============================================================================
# When analysis_mode = "fast", these settings override defaults for quick testing
# This reduces accuracy but provides rapid feedback for testing pipelines

get_effective_config <- function(config) {
  if (config$analysis_mode == "fast") {
    log_message("FAST MODE ENABLED - Using reduced settings for quick testing", "WARN")
    config$n_folds <- min(config$n_folds, 3)
    config$n_repeats <- min(config$n_repeats, 1)
    config$n_permutations <- 10
    config$rf_ntree <- 50
    config$xgb_nrounds <- 20
    config$mlp_maxit <- 50
    config$max_features <- min(config$max_features, 10)
    config$feature_selection_method <- "none"  # Skip feature selection in fast mode
  }
  return(config)
}

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_message <- function(msg, level = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("[%s] %s: %s\n", timestamp, level, msg))
}

#' Display a text-based progress bar
#' @param current Current iteration
#' @param total Total iterations
#' @param prefix Text to show before the bar
#' @param width Width of the progress bar in characters
show_progress <- function(current, total, prefix = "Progress", width = 40) {
  pct <- current / total
  filled <- floor(pct * width)
  empty <- width - filled
  bar <- paste0(
    "\r", prefix, " [",
    paste(rep("=", filled), collapse = ""),
    ifelse(filled < width, ">", ""),
    paste(rep(" ", max(0, empty - 1)), collapse = ""),
    "] ",
    sprintf("%3d%%", round(pct * 100)),
    " (", current, "/", total, ")"
  )
  cat(bar)
  if (current == total) cat("\n")
  flush.console()
}

#' Read tab-delimited file (supports .txt and .tsv)
read_tabular <- function(file_path) {
  if (!file.exists(file_path)) {
    stop(sprintf("File not found: %s", file_path))
  }
  
  ext <- tolower(tools::file_ext(file_path))
  
  if (ext %in% c("txt", "tsv")) {
    data <- read.delim(file_path, stringsAsFactors = FALSE, sep = "\t", row.names = 1, check.names = FALSE)
  } else if (ext == "csv") {
    data <- read.csv(file_path, stringsAsFactors = FALSE, row.names = 1, check.names = FALSE)
  } else {
    data <- read.delim(file_path, stringsAsFactors = FALSE, sep = "\t", row.names = 1, check.names = FALSE)
  }
  
  return(data)
}

#' Safe calculation of metrics with handling for single-class predictions
#' @param actual Actual class labels
#' @param predicted Predicted class labels
#' @param probabilities Optional probability vector for AUROC calculation
#' @return List of metrics
calculate_metrics <- function(actual, predicted, probabilities = NULL) {
  # Ensure both vectors have the same factor levels
  all_levels <- union(levels(as.factor(actual)), levels(as.factor(predicted)))
  if (length(all_levels) < 2) {
    all_levels <- c("0", "1")
  }
  
  actual_factor <- factor(actual, levels = all_levels)
  predicted_factor <- factor(predicted, levels = all_levels)
  
  # Check if we have at least 2 levels with data
  actual_unique <- unique(as.character(actual))
  predicted_unique <- unique(as.character(predicted))
  
  # If predictions have only one class, return NA metrics
  if (length(predicted_unique) < 2 || length(actual_unique) < 2) {
    log_message("Warning: Single class in predictions or actual values, returning NA metrics", "WARN")
    metrics <- list(
      accuracy = NA_real_,
      sensitivity = NA_real_,
      specificity = NA_real_,
      precision = NA_real_,
      f1_score = NA_real_,
      balanced_accuracy = NA_real_,
      kappa = NA_real_,
      confusion_matrix = list(tp = 0, tn = 0, fp = 0, fn = 0)
    )
    
    if (!is.null(probabilities)) {
      metrics$auroc <- NA_real_
      metrics$roc_curve <- data.frame(fpr = c(0, 1), tpr = c(0, 1))
    }
    
    return(metrics)
  }
  
  cm <- tryCatch({
    confusionMatrix(
      data = predicted_factor,
      reference = actual_factor
    )
  }, error = function(e) {
    log_message(sprintf("confusionMatrix error: %s", e$message), "WARN")
    return(NULL)
  })
  
  if (is.null(cm)) {
    return(list(
      accuracy = NA_real_,
      sensitivity = NA_real_,
      specificity = NA_real_,
      precision = NA_real_,
      f1_score = NA_real_,
      balanced_accuracy = NA_real_,
      kappa = NA_real_,
      confusion_matrix = list(tp = 0, tn = 0, fp = 0, fn = 0),
      auroc = NA_real_,
      roc_curve = data.frame(fpr = c(0, 1), tpr = c(0, 1))
    ))
  }
  
  metrics <- list(
    accuracy = as.numeric(cm$overall["Accuracy"]),
    sensitivity = as.numeric(cm$byClass["Sensitivity"]),
    specificity = as.numeric(cm$byClass["Specificity"]),
    precision = as.numeric(cm$byClass["Pos Pred Value"]),
    f1_score = as.numeric(cm$byClass["F1"]),
    balanced_accuracy = as.numeric(cm$byClass["Balanced Accuracy"]),
    kappa = as.numeric(cm$overall["Kappa"]),
    confusion_matrix = list(
      tp = cm$table[2, 2],
      tn = cm$table[1, 1],
      fp = cm$table[2, 1],
      fn = cm$table[1, 2]
    )
  )
  
  if (!is.null(probabilities)) {
    roc_obj <- tryCatch({
      roc(actual, probabilities, quiet = TRUE)
    }, error = function(e) NULL)
    
    if (!is.null(roc_obj)) {
      metrics$auroc <- as.numeric(auc(roc_obj))
      metrics$roc_curve <- data.frame(
        fpr = 1 - roc_obj$specificities,
        tpr = roc_obj$sensitivities
      )
    } else {
      metrics$auroc <- NA_real_
      metrics$roc_curve <- data.frame(fpr = c(0, 1), tpr = c(0, 1))
    }
  }
  
  return(metrics)
}

normalize_importance <- function(importance) {
  min_val <- min(importance, na.rm = TRUE)
  max_val <- max(importance, na.rm = TRUE)
  if (max_val == min_val) return(rep(1, length(importance)))
  return((importance - min_val) / (max_val - min_val))
}

select_best_model <- function(summary_metrics) {
  
  valid_models <- c("rf", "svm", "xgboost", "knn", "mlp")
  
  scores <- sapply(valid_models, function(m) {
    if (is.null(summary_metrics[[m]])) return(NA_real_)
    
    if (!is.null(summary_metrics[[m]]$auroc)) {
      return(summary_metrics[[m]]$auroc$mean)
    }
    if (!is.null(summary_metrics[[m]]$balanced_accuracy)) {
      return(summary_metrics[[m]]$balanced_accuracy$mean)
    }
    NA_real_
  })
  
  scores <- scores[is.finite(scores)]
  if (length(scores) == 0) return(NULL)
  
  names(which.max(scores))
}


# =============================================================================
# DATA LOADING - Expression Matrix + Annotation
# =============================================================================

load_data <- function(config) {
  log_message("Loading expression matrix and sample annotations...")
  
  # Load expression matrix (rows = features, columns = samples)
  log_message(sprintf("Reading expression matrix: %s", config$expression_matrix_file))
  expr_matrix <- read_tabular(config$expression_matrix_file)
  
  # Store original expression values before transformation
  original_expr_matrix <- expr_matrix
  
  # Transpose: now rows = samples, columns = features
  expr_matrix <- as.data.frame(t(expr_matrix))
  sample_ids <- rownames(expr_matrix)
  
  log_message(sprintf("Expression matrix: %d samples x %d features", nrow(expr_matrix), ncol(expr_matrix)))
  
  # Load annotation file  
  log_message(sprintf("Reading annotation file: %s", config$annotation_file))
  annotation <- read.delim(config$annotation_file, stringsAsFactors = FALSE, sep = "\t")
  
  # Find sample ID column in annotation (first column or look for common pattern)
  annot_sample_col <- colnames(annotation)[1]
  annot_samples <- annotation[[annot_sample_col]]
  
  # Check for target variable
  if (!config$target_variable %in% colnames(annotation)) {
    stop(sprintf("Target variable '%s' not found in annotation file. Available: %s",
                 config$target_variable, paste(colnames(annotation), collapse = ", ")))
  }
  
  # Find common samples
  common_samples <- intersect(sample_ids, annot_samples)
  log_message(sprintf("Found %d common samples between expression matrix and annotation", length(common_samples)))
  
  if (length(common_samples) == 0) {
    stop("No common samples found. Ensure column names in expression matrix match sample IDs in annotation.")
  }
  
  # Filter and align data
  expr_matrix <- expr_matrix[rownames(expr_matrix) %in% common_samples, , drop = FALSE]
  annotation <- annotation[annotation[[annot_sample_col]] %in% common_samples, ]
  
  # Sort to ensure alignment
  expr_matrix <- expr_matrix[order(rownames(expr_matrix)), , drop = FALSE]
  annotation <- annotation[order(annotation[[annot_sample_col]]), ]
  
  sample_ids <- rownames(expr_matrix)
  X <- expr_matrix
  y <- as.factor(annotation[[config$target_variable]])
  
  # ==========================================================================
  # PRE-FLIGHT DATA QUALITY CHECK: Require at least 2 classes
  # ==========================================================================
  n_classes <- length(levels(y))
  if (n_classes < 2) {
    stop(paste0(
      "DATA QUALITY ERROR: Only ", n_classes, " class found after intersection ('",
      paste(levels(y), collapse = ", "), "'). ",
      "Binary classification requires at least 2 distinct classes in the target variable. ",
      "Please verify your annotation file contains samples from both classes that match ",
      "the sample IDs in the expression matrix."
    ))
  }
  
  # Store preprocessing stats before modifications
  # Calculate train/test split info based on config (will be updated later with actual config)
  n_folds <- if (!is.null(config$n_folds)) config$n_folds else 5
  n_repeats <- if (!is.null(config$n_repeats)) config$n_repeats else 3
  test_per_fold <- ceiling(length(sample_ids) / n_folds)
  train_per_fold <- length(sample_ids) - test_per_fold
  
  # Calculate approximate class distribution per fold (stratified)
  class_table <- table(y)
  train_class_dist <- as.list(round(class_table * (n_folds - 1) / n_folds))
  test_class_dist <- as.list(ceiling(class_table / n_folds))
  
  preprocessing_stats <- list(
    original_samples = length(sample_ids),
    original_features = ncol(X),
    missing_values = sum(is.na(X)),
    missing_pct = round(sum(is.na(X)) / (nrow(X) * ncol(X)) * 100, 2),
    class_distribution = as.list(class_table),
    constant_features_removed = 0,
    cv_folds = n_folds,
    cv_repeats = n_repeats,
    train_samples_per_fold = train_per_fold,
    test_samples_per_fold = test_per_fold,
    train_class_distribution = train_class_dist,
    test_class_distribution = test_class_dist,
    full_training_mode = FALSE
  )
  
  # Ensure X is numeric
  X <- as.data.frame(lapply(X, function(x) {
    if (is.character(x)) as.numeric(x) else x
  }))
  
  # Remove constant columns
  constant_cols <- sapply(X, function(x) length(unique(x[!is.na(x)])) <= 1)
  if (any(constant_cols)) {
    log_message(sprintf("Removing %d constant columns", sum(constant_cols)), "WARN")
    preprocessing_stats$constant_features_removed <- sum(constant_cols)
    X <- X[, !constant_cols]
  }
  
  # Handle missing values
  if (any(is.na(X))) {
    log_message("Imputing missing values with median", "WARN")
    X <- as.data.frame(lapply(X, function(x) {
      if (is.numeric(x)) x[is.na(x)] <- median(x, na.rm = TRUE)
      return(x)
    }))
  }
  
  # Store unscaled expression for boxplot export (subset to current features)
  X_raw <- X
  
  # --------------------------------------------------
  # Optional Z-score scaling (per feature)
  # --------------------------------------------------
  
  X_scaled <- X
  
  if (isTRUE(config$scale_data)) {
    
    log_message("Preparing scaled data matrix (Z-score per feature)", "INFO")
    
    feature_sd <- apply(X_scaled, 2, sd, na.rm = TRUE)
    nonzero_sd <- feature_sd > 0
    
    if (any(!nonzero_sd)) {
      log_message(sprintf(
        "Removed %d zero-variance features before scaling",
        sum(!nonzero_sd)
      ), "WARN")
    }
    
    X_scaled <- X_scaled[, nonzero_sd, drop = FALSE]
    
    X_scaled <- as.data.frame(
      t(apply(X_scaled, 1, function(x) scale(as.numeric(x))))
    )
    
    colnames(X_scaled) <- colnames(X_raw)[nonzero_sd]
    rownames(X_scaled) <- rownames(X_raw)
  }
  
  
  # ------------------------------------------------------------------
  # Variance-based feature filtering (CRITICAL for speed & stability)
  # ------------------------------------------------------------------
  
  feature_var <- apply(X_scaled, 2, var, na.rm = TRUE)
  
  # Remove bottom X_scaled% of variance
  var_cutoff <- quantile(feature_var, VAR_QUANTILE, na.rm = TRUE)
  
  keep_features <- feature_var > var_cutoff
  
  log_message(
    sprintf("Variance filtering: kept %d/%d features (%.1f%%)",
            sum(keep_features),
            length(keep_features),
            100 * sum(keep_features) / length(keep_features)),
    "INFO"
  )
  
  X_scaled <- X_scaled[, keep_features, drop = FALSE]
  
  # Apply caret near-zero variance filter
  nzv <- nearZeroVar(X_scaled, saveMetrics = TRUE)
  X_scaled <- X_scaled[, !nzv$nzv, drop = FALSE]
  
  log_message(
    sprintf("Removed %d near-zero variance features",
            sum(nzv$nzv)),
    "INFO"
  )
  
  log_message(sprintf("Final data: %d samples, %d features, %d classes",
                      nrow(X_scaled), ncol(X_scaled), length(levels(y))))
  log_message(sprintf("Class distribution: %s",
                      paste(names(table(y)), table(y), sep = "=", collapse = ", ")))
  
  return(list(
    X_raw    = X_raw,
    X_scaled = X_scaled,
    y = y,
    sample_ids = sample_ids,
    feature_names = colnames(X),
    preprocessing_stats = preprocessing_stats,
    annotation = annotation,
    annot_sample_col = annot_sample_col
  ))
}

# =============================================================================
# FEATURE SELECTION (same as before)
# =============================================================================

forward_selection <- function(X, y, max_features, seed = 42) {
  set.seed(seed)
  log_message("Performing forward selection...")
  
  selected <- c()
  remaining <- colnames(X)
  best_accuracy <- 0
  
  for (i in 1:min(max_features, length(remaining))) {
    results <- sapply(remaining, function(feat) {
      current_features <- c(selected, feat)
      data_subset <- X[, current_features, drop = FALSE]
      ctrl <- trainControl(method = "cv", number = 3)
      model <- train(x = data_subset, y = y, method = "rf",
                     trControl = ctrl, ntree = 100, tuneLength = 1)
      return(max(model$results$Accuracy))
    })
    
    best_feat <- names(which.max(results))
    best_acc <- max(results)
    
    if (best_acc > best_accuracy) {
      best_accuracy <- best_acc
      selected <- c(selected, best_feat)
      remaining <- setdiff(remaining, best_feat)
      log_message(sprintf("  Added feature %d: %s (Acc: %.4f)", i, best_feat, best_acc))
    } else {
      log_message("  No improvement, stopping early")
      break
    }
  }
  
  return(selected)
}

backward_elimination <- function(X, y, max_features = 50, seed = 42) {
  set.seed(seed)
  log_message("Performing backward elimination...")
  
  # Cap initial features if too many
  if (ncol(X) > max_features) {
    log_message(sprintf("Pre-filtering to top %d features by variance for backward elimination", max_features))
    feature_var <- apply(X, 2, var, na.rm = TRUE)
    top_idx <- order(feature_var, decreasing = TRUE)[1:max_features]
    X <- X[, top_idx, drop = FALSE]
  }
  
  selected <- colnames(X)
  ctrl <- trainControl(method = "cv", number = 3)
  model <- train(x = X, y = y, method = "rf",
                 trControl = ctrl, ntree = 100, tuneLength = 1)
  best_accuracy <- max(model$results$Accuracy)
  
  min_features <- 5
  while (length(selected) > min_features) {
    results <- sapply(selected, function(feat) {
      current_features <- setdiff(selected, feat)
      data_subset <- X[, current_features, drop = FALSE]
      model <- train(x = data_subset, y = y, method = "rf",
                     trControl = ctrl, ntree = 100, tuneLength = 1)
      return(max(model$results$Accuracy))
    })
    
    worst_feat <- names(which.max(results))
    acc_without <- max(results)
    
    if (acc_without >= best_accuracy - 0.01) {
      selected <- setdiff(selected, worst_feat)
      best_accuracy <- acc_without
      log_message(sprintf("  Removed: %s (Acc: %.4f, %d remaining)", 
                          worst_feat, acc_without, length(selected)))
    } else {
      break
    }
  }
  
  return(selected)
}

stepwise_selection <- function(X, y, max_features, seed = 42) {
  set.seed(seed)
  log_message("Performing stepwise selection...")
  
  selected <- c()
  remaining <- colnames(X)
  best_accuracy <- 0
  ctrl <- trainControl(method = "cv", number = 3)
  
  for (i in 1:min(max_features, length(remaining))) {
    results <- sapply(remaining, function(feat) {
      current_features <- c(selected, feat)
      data_subset <- X[, current_features, drop = FALSE]
      model <- train(x = data_subset, y = y, method = "rf",
                     trControl = ctrl, ntree = 100, tuneLength = 1)
      return(max(model$results$Accuracy))
    })
    
    best_feat <- names(which.max(results))
    best_acc <- max(results)
    
    if (best_acc > best_accuracy) {
      selected <- c(selected, best_feat)
      remaining <- setdiff(remaining, best_feat)
      best_accuracy <- best_acc
      log_message(sprintf("  Added: %s (Acc: %.4f)", best_feat, best_acc))
      
      if (length(selected) > 2) {
        for (feat in selected[-length(selected)]) {
          test_features <- setdiff(selected, feat)
          data_subset <- X[, test_features, drop = FALSE]
          model <- train(x = data_subset, y = y, method = "rf",
                         trControl = ctrl, ntree = 100, tuneLength = 1)
          acc <- max(model$results$Accuracy)
          
          if (acc >= best_accuracy) {
            selected <- test_features
            best_accuracy <- acc
            remaining <- c(remaining, feat)
            log_message(sprintf("  Removed: %s (Acc: %.4f)", feat, acc))
          }
        }
      }
    } else {
      break
    }
  }
  
  return(selected)
}

perform_feature_selection <- function(X, y, method, max_features, seed = 42) {
  # Enforce max_features limit (default 50)
  max_features <- min(max_features, 50)
  log_message(sprintf("Feature selection: method=%s, max_features=%d", method, max_features))
  
  selected <- switch(method,
         "forward" = forward_selection(X, y, max_features, seed),
         "backward" = backward_elimination(X, y, max_features, seed),
         "stepwise" = stepwise_selection(X, y, max_features, seed),
         "none" = {
           # For "none" method, select top features by variance (capped at max_features)
           if (ncol(X) <= max_features) {
             colnames(X)
           } else {
             log_message(sprintf("Capping features from %d to %d using variance ranking", ncol(X), max_features))
             feature_var <- apply(X, 2, var, na.rm = TRUE)
             top_idx <- order(feature_var, decreasing = TRUE)[1:max_features]
             colnames(X)[top_idx]
           }
         }
  )
  
  # Final enforcement of max_features limit
  if (length(selected) > max_features) {
    log_message(sprintf("Trimming selected features from %d to %d", length(selected), max_features), "WARN")
    selected <- selected[1:max_features]
  }
  
  return(selected)
}

# =============================================================================
# MODEL TRAINING
# =============================================================================

train_rf <- function(X_train, y_train, config) {
  mtry <- config$rf_mtry
  if (is.null(mtry)) mtry <- floor(sqrt(ncol(X_train)))
  
  model <- randomForest(x = X_train, y = y_train, ntree = config$rf_ntree,
                        mtry = mtry, importance = TRUE)
  importance <- importance(model, type = 2)[, 1]
  
  return(list(model = model, importance = importance,
              oob_error = model$err.rate[config$rf_ntree, "OOB"]))
}

train_svm <- function(X_train, y_train, config) {
  gamma <- config$svm_gamma
  if (is.null(gamma)) gamma <- 1 / ncol(X_train)
  
  model <- svm(x = as.matrix(X_train), y = y_train, kernel = config$svm_kernel,
               cost = config$svm_cost, gamma = gamma, probability = TRUE)
  return(list(model = model))
}

train_xgboost <- function(X_train, y_train, config) {
  X_matrix <- as.matrix(X_train)
  y_numeric <- as.numeric(y_train) - 1
  dtrain <- xgb.DMatrix(data = X_matrix, label = y_numeric)
  
  params <- list(objective = "binary:logistic", eval_metric = "auc",
                 max_depth = config$xgb_max_depth, eta = config$xgb_eta)
  
  model <- xgb.train(params = params, data = dtrain, 
                     nrounds = config$xgb_nrounds, verbose = 0)
  importance <- xgb.importance(model = model)
  
  return(list(model = model, importance = importance))
}

train_knn <- function(X_train, y_train, config) {
  return(list(X_train = X_train, y_train = y_train, k = config$knn_k))
}

train_mlp <- function(X_train, y_train, config) {
  model <- nnet(x = as.matrix(X_train), y = class.ind(y_train),
                size = config$mlp_size, decay = config$mlp_decay,
                maxit = config$mlp_maxit, softmax = TRUE, trace = FALSE)
  return(list(model = model))
}

# Prediction functions
#' Safe extraction of class probability
#' Handles cases where probability matrix may not have expected class columns
#' @param prob_matrix Probability matrix from model prediction
#' @param target_class Target class column name (default "1")
#' @return Vector of probabilities
safe_get_prob <- function(prob_matrix, target_class = "1", n = NULL, default = 0.5) {
  # If we have no probability output (e.g., model couldn't be trained), return a
  # neutral probability vector of length n when possible.
  if (is.null(prob_matrix)) {
    if (!is.null(n) && n > 0) return(rep(default, n))
    return(numeric(0))
  }
  
  # Some predictors return a bare vector already
  if (is.vector(prob_matrix)) {
    return(as.numeric(prob_matrix))
  }
  
  if (!is.null(colnames(prob_matrix)) && target_class %in% colnames(prob_matrix)) {
    return(as.numeric(prob_matrix[, target_class]))
  }
  
  if (ncol(prob_matrix) >= 2) {
    # Common convention: second column is the positive class
    return(as.numeric(prob_matrix[, 2]))
  }
  
  if (ncol(prob_matrix) == 1) {
    return(as.numeric(prob_matrix[, 1]))
  }
  
  if (!is.null(n) && n > 0) return(rep(default, n))
  return(numeric(0))
}

predict_rf <- function(model_obj, X_test) {
  pred <- predict(model_obj$model, X_test)
  prob_matrix <- predict(model_obj$model, X_test, type = "prob")
  prob <- safe_get_prob(prob_matrix, "1", n = nrow(X_test))
  return(list(predictions = pred, probabilities = prob))
}

predict_svm <- function(model_obj, X_test) {
  pred <- predict(model_obj$model, as.matrix(X_test))
  prob_attr <- predict(model_obj$model, as.matrix(X_test), probability = TRUE)
  prob_matrix <- attr(prob_attr, "probabilities")
  prob <- safe_get_prob(prob_matrix, "1", n = nrow(X_test))
  return(list(predictions = pred, probabilities = prob))
}

predict_xgboost <- function(model_obj, X_test) {
  prob <- predict(model_obj$model, as.matrix(X_test))
  pred <- factor(ifelse(prob > 0.5, "1", "0"), levels = c("0", "1"))
  return(list(predictions = pred, probabilities = prob))
}

predict_knn <- function(model_obj, X_test) {
  pred <- knn(train = model_obj$X_train, test = X_test,
              cl = model_obj$y_train, k = model_obj$k, prob = TRUE)
  prob <- attr(pred, "prob")
  prob <- ifelse(pred == "1", prob, 1 - prob)
  return(list(predictions = pred, probabilities = prob))
}

predict_mlp <- function(model_obj, X_test) {
  prob_matrix <- predict(model_obj$model, as.matrix(X_test))
  prob <- safe_get_prob(prob_matrix, n = nrow(X_test))
  pred <- factor(ifelse(prob > 0.5, "1", "0"), levels = c("0", "1"))
  return(list(predictions = pred, probabilities = prob))
}

# Ensemble voting
hard_voting <- function(predictions_list) {
  pred_matrix <- do.call(cbind, lapply(predictions_list, as.character))
  final_pred <- apply(pred_matrix, 1, function(row) names(which.max(table(row))))
  return(factor(final_pred, levels = c("0", "1")))
}

soft_voting <- function(probabilities_list, weights = NULL) {
  n_models <- length(probabilities_list)
  if (is.null(weights)) weights <- rep(1/n_models, n_models)
  avg_prob <- Reduce(`+`, Map(`*`, probabilities_list, weights))
  pred <- factor(ifelse(avg_prob > 0.5, "1", "0"), levels = c("0", "1"))
  return(list(predictions = pred, probabilities = avg_prob))
}

is_valid_prediction <- function(pred, y_test) {
  !is.null(pred) &&
    length(pred) == length(y_test) &&
    length(unique(as.character(pred))) > 1 &&
    length(unique(as.character(y_test))) > 1
}

safe_stats <- function(values) {
  
  if (is.list(values)) {
    values <- unlist(values, use.names = FALSE)
  }
  
  values <- as.numeric(values)
  values <- values[is.finite(values)]
  
  n <- length(values)
  
  if (n == 0) {
    return(list(
      mean = NA_real_, sd = NA_real_, median = NA_real_,
      q25 = NA_real_, q75 = NA_real_, min = NA_real_, max = NA_real_,
      n_folds = 0
    ))
  }
  
  if (n == 1) {
    return(list(
      mean = values, sd = 0, median = values,
      q25 = values, q75 = values,
      min = values, max = values,
      n_folds = 1
    ))
  }
  
  list(
    mean = mean(values),
    sd = sd(values),
    median = median(values),
    q25 = as.numeric(quantile(values, 0.25, names = FALSE)),
    q75 = as.numeric(quantile(values, 0.75, names = FALSE)),
    min = min(values),
    max = max(values),
    n_folds = n
  )
}


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

run_cv_all_methods <- function(X_raw, X_scaled, y, config, selected_features = NULL) {
  set.seed(config$seed)
  
  ensemble_valid <- FALSE

  valid_fold_counter <- list(
    rf = 0, svm = 0, xgboost = 0,
    knn = 0, mlp = 0,
    hard_vote = 0, soft_vote = 0
  )
  
  attempted_folds <- 0L
  
  if (!is.null(selected_features)) X_raw <- X_raw[, selected_features, drop = FALSE]
  
  n_samples <- nrow(X_raw)
  set.seed(config$seed)
  
  folds <- list()
  fold_id <- 1
  
  for (r in seq_len(config$n_repeats)) {
    strat_folds <- createFolds(y, k = config$n_folds, returnTrain = TRUE)
    for (f in seq_along(strat_folds)) {
      folds[[fold_id]] <- strat_folds[[f]]
      fold_id <- fold_id + 1
    }
  }
  
  
  all_results <- list(rf = list(), svm = list(), xgboost = list(), 
                      knn = list(), mlp = list(), 
                      hard_vote = list(), soft_vote = list())
  
  importance_scores <- list()
  
  # Collect per-fold predictions for calibration curves export
  cv_predictions <- list(rf = NULL, svm = NULL, xgboost = NULL, knn = NULL, mlp = NULL, soft_vote = NULL)
  
  # Collect per-fold importance for stability analysis
  fold_importance <- list()
  
  log_message(sprintf("Running %d-fold CV with %d repeats", config$n_folds, config$n_repeats))
  n_folds_total <- length(folds)
  
  for (i in seq_along(folds)) {
    
    attempted_folds <- attempted_folds + 1L
    
    train_idx <- folds[[i]]
    test_idx <- setdiff(1:n_samples, train_idx)
    y_train <- y[train_idx]
    y_test <- y[test_idx]
    
    # Show progress bar
    show_progress(i, n_folds_total, "CV Progress")
    
    # Skip fold if single class in training set
    if (length(unique(as.character(y_train))) < 2) {
      log_message(sprintf("Skipping fold %d: single class in training set", i), "WARN")
      next
    }
    
    
    get_X <- function(model_name) {
      use_scaled <- isTRUE(config$model_scaling[[model_name]])
      X <- if (use_scaled) X_scaled else X_raw
      
      if (!is.null(selected_features)) {
        X <- X[, intersect(selected_features, colnames(X)), drop = FALSE]
      }
      return(X)
    }
    
    X_rf  <- get_X("rf")
    X_svm <- get_X("svm")
    X_knn <- get_X("knn")
    X_mlp <- get_X("mlp")
    X_xgb <- get_X("xgboost")
    
    models <- list(
      rf      = tryCatch(train_rf(X_rf[train_idx, ], y_train, config), error = function(e) NULL),
      svm     = tryCatch(train_svm(X_svm[train_idx, ], y_train, config), error = function(e) NULL),
      knn     = tryCatch(train_knn(X_knn[train_idx, ], y_train, config), error = function(e) NULL),
      mlp     = tryCatch(train_mlp(X_mlp[train_idx, ], y_train, config), error = function(e) NULL),
      xgboost = tryCatch(train_xgboost(X_xgb[train_idx, ], y_train, config), error = function(e) NULL)
    )
    
    preds <- list()
    probs <- list()
    
    if (!is.null(models$rf)) {
      result <- predict_rf(models$rf, X_rf[test_idx, ])
      preds$rf <- result$predictions
      probs$rf <- result$probabilities
      
      # 🔒 Skip fold if predictions are single-class
      if (length(unique(as.character(preds$rf))) < 2 ||
          length(unique(as.character(y_test))) < 2) {
        
        log_message(
          sprintf("Skipping RF metrics for fold %d: single-class prediction or test set", i),
          "WARN"
        )
        
      } else {
        all_results$rf[[i]] <- calculate_metrics(y_test, preds$rf, probs$rf)
        valid_fold_counter$rf <- valid_fold_counter$rf + 1
      }
      
      if (!is.null(models$rf$importance)) {
        importance_scores[[length(importance_scores) + 1]] <- models$rf$importance
        fold_importance[[length(fold_importance) + 1]] <- models$rf$importance
      }
      # Collect for calibration
      cv_predictions$rf <- rbind(cv_predictions$rf, data.frame(actual = as.character(y_test), prob = probs$rf))
    }
    
    if (!is.null(models$svm)) {
      result <- predict_svm(models$svm, X_svm[test_idx, ])
      preds$svm <- result$predictions
      probs$svm <- result$probabilities
      
      # 🔒 Skip fold if predictions are single-class
      if (length(unique(as.character(preds$svm))) < 2 ||
          length(unique(as.character(y_test))) < 2) {
        
        log_message(
          sprintf("Skipping SVM metrics for fold %d: single-class prediction or test set", i),
          "WARN"
        )
        
      } else {
        all_results$svm[[i]] <- calculate_metrics(y_test, preds$svm, probs$svm)
        valid_fold_counter$svm <- valid_fold_counter$svm + 1
      }
      cv_predictions$svm <- rbind(cv_predictions$svm, data.frame(actual = as.character(y_test), prob = probs$svm))
    }
    
    if (!is.null(models$xgboost)) {
      result <- predict_xgboost(models$xgboost, X_xgb[test_idx, ])
      preds$xgboost <- result$predictions
      probs$xgboost <- result$probabilities
      
      # 🔒 Skip fold if predictions are single-class
      if (length(unique(as.character(preds$xgboost))) < 2 ||
          length(unique(as.character(y_test))) < 2) {
        
        log_message(
          sprintf("Skipping XGboost metrics for fold %d: single-class prediction or test set", i),
          "WARN"
        )
        
      } else {
        all_results$xgboost[[i]] <- calculate_metrics(y_test, preds$xgboost, probs$xgboost)
        valid_fold_counter$xgboost <- valid_fold_counter$xgboost + 1
      }
      
      cv_predictions$xgboost <- rbind(cv_predictions$xgboost, data.frame(actual = as.character(y_test), prob = probs$xgboost))
    }
    
    if (!is.null(models$knn)) {
      result <- predict_knn(models$knn, X_knn[test_idx, ])
      preds$knn <- result$predictions
      probs$knn <- result$probabilities
      
      # 🔒 Skip fold if predictions are single-class
      if (length(unique(as.character(preds$knn))) < 2 ||
          length(unique(as.character(y_test))) < 2) {
        
        log_message(
          sprintf("Skipping KNN metrics for fold %d: single-class prediction or test set", i),
          "WARN"
        )
        
      } else {
        all_results$knn[[i]] <- calculate_metrics(y_test, preds$knn, probs$knn)
        valid_fold_counter$knn <- valid_fold_counter$knn + 1
      }
      cv_predictions$knn <- rbind(cv_predictions$knn, data.frame(actual = as.character(y_test), prob = probs$knn))
    }
    
    if (!is.null(models$mlp)) {
      result <- predict_mlp(models$mlp, X_mlp[test_idx, ])
      preds$mlp <- result$predictions
      probs$mlp <- result$probabilities
      
      # 🔒 Skip fold if predictions are single-class
      if (length(unique(as.character(preds$mlp))) < 2 ||
          length(unique(as.character(y_test))) < 2) {
        
        log_message(
          sprintf("Skipping MLP metrics for fold %d: single-class prediction or test set", i),
          "WARN"
        )
        
      } else {
        all_results$mlp[[i]] <- calculate_metrics(y_test, preds$mlp, probs$mlp)
        valid_fold_counter$mlp <- valid_fold_counter$mlp + 1
      }
      cv_predictions$mlp <- rbind(cv_predictions$mlp, data.frame(actual = as.character(y_test), prob = probs$mlp))
    }
    
    # ---------------------------------------------------------
    # SAFE ENSEMBLE VOTING (exclude invalid / skipped models)
    # ---------------------------------------------------------
    
    valid_pred_names <- names(preds)[
      sapply(names(preds), function(m) {
        is_valid_prediction(preds[[m]], y_test)
      })
    ]
    
    valid_prob_names <- names(probs)[
      sapply(names(probs), function(m) {
        !is.null(probs[[m]]) &&
          length(probs[[m]]) == length(y_test) &&
          length(unique(probs[[m]])) > 1
      })
    ]
    
    # --- HARD VOTING ---
    if (length(valid_pred_names) >= 2) {
      
      hard_pred <- hard_voting(preds[valid_pred_names])
      
      if (is_valid_prediction(hard_pred, y_test)) {
        all_results$hard_vote[[i]] <- calculate_metrics(y_test, hard_pred)
        valid_fold_counter$hard_vote <- valid_fold_counter$hard_vote + 1
        ensemble_valid <- TRUE
      } else {
        log_message(sprintf(
          "Skipping HARD VOTE metrics for fold %d: invalid ensemble prediction", i),
          "WARN"
        )
      }
    } else {
      log_message(sprintf(
        "Skipping HARD VOTE for fold %d: <2 valid base models", i),
        "WARN"
      )
    }
    
    # --- SOFT VOTING ---
    if (length(valid_prob_names) >= 2) {
      
      # Equal weights only for valid models
      weights <- rep(1 / length(valid_prob_names), length(valid_prob_names))
      
      soft_result <- soft_voting(
        probabilities_list = probs[valid_prob_names],
        weights = weights
      )
      
      if (is_valid_prediction(soft_result$predictions, y_test)) {
        all_results$soft_vote[[i]] <- calculate_metrics(
          y_test,
          soft_result$predictions,
          soft_result$probabilities
        )
        valid_fold_counter$soft_vote <- valid_fold_counter$soft_vote + 1
        ensemble_valid <- TRUE 
      } else {
        log_message(sprintf(
          "Skipping SOFT VOTE metrics for fold %d: invalid ensemble prediction", i),
          "WARN"
        )
      }
      
    } else {
      log_message(sprintf(
        "Skipping SOFT VOTE for fold %d: <2 valid base models", i),
        "WARN"
      )
      
      ensemble_valid <- FALSE
    }
  }
  
  feature_importance <- data.frame(feature = colnames(X_raw))
  if (length(importance_scores) > 0) {
    avg_importance <- rowMeans(do.call(cbind, importance_scores), na.rm = TRUE)
    feature_importance$importance <- avg_importance
    feature_importance <- feature_importance[order(-feature_importance$importance), ]
  }
  
  return(list(
    results = all_results,
    feature_importance = feature_importance,
    cv_predictions = cv_predictions,
    fold_importance = fold_importance,
    valid_folds = valid_fold_counter,
    attempted_folds = attempted_folds,
    ensemble_valid = ensemble_valid
  ))
  
}

# =============================================================================
# PERMUTATION TESTING
# =============================================================================

run_permutation_test <- function(X, y, config, selected_features = NULL) {
  n_permutations <- config$n_permutations
  log_message(sprintf("Running permutation testing with %d permutations", n_permutations))
  
  if (!is.null(selected_features)) X <- X[, selected_features, drop = FALSE]
  
  # Store per-model permutation metrics for distribution plots
  permutation_results <- list(
    rf_oob_error = numeric(n_permutations),
    rf_auroc = numeric(n_permutations),
    # Per-model auroc and accuracy distributions
    per_model = list(
      rf = list(auroc = numeric(n_permutations), accuracy = numeric(n_permutations)),
      svm = list(auroc = numeric(n_permutations), accuracy = numeric(n_permutations)),
      xgboost = list(auroc = numeric(n_permutations), accuracy = numeric(n_permutations)),
      knn = list(auroc = numeric(n_permutations), accuracy = numeric(n_permutations)),
      mlp = list(auroc = numeric(n_permutations), accuracy = numeric(n_permutations)),
      soft_vote = list(auroc = numeric(n_permutations), accuracy = numeric(n_permutations))
    )
  )
  
  set.seed(config$seed)
  folds <- createFolds(y, k = config$n_folds)
  
  for (p in 1:n_permutations) {
    # Show progress bar
    show_progress(p, n_permutations, "Permutation Testing")
    
    y_permuted <- sample(y)
    
    # RF OOB error
    rf_model <- tryCatch({
      randomForest(x = X, y = y_permuted, ntree = min(config$rf_ntree, 200), importance = FALSE)
    }, error = function(e) NULL)
    
    if (!is.null(rf_model)) {
      permutation_results$rf_oob_error[p] <- rf_model$err.rate[nrow(rf_model$err.rate), "OOB"]
    }
    
    # Collect CV predictions per model for this permutation
    cv_probs_rf <- numeric(length(y))
    cv_probs_svm <- numeric(length(y))
    cv_probs_xgb <- numeric(length(y))
    cv_probs_knn <- numeric(length(y))
    cv_probs_mlp <- numeric(length(y))
    cv_probs_soft <- numeric(length(y))
    
    cv_preds_rf <- character(length(y))
    cv_preds_svm <- character(length(y))
    cv_preds_xgb <- character(length(y))
    cv_preds_knn <- character(length(y))
    cv_preds_mlp <- character(length(y))
    cv_preds_soft <- character(length(y))
    
    for (fold in folds) {
      train_idx <- setdiff(1:length(y), fold)
      test_idx <- fold
      
      probs_list <- list()
      
      # RF
      rf_fold <- tryCatch(train_rf(X_rf[train_idx, ], y_train, config), error = function(e) NULL)
      if (!is.null(rf_fold)) {
        res <- predict_rf(rf_fold, X_rf[test_idx, ])
        cv_probs_rf[test_idx] <- res$probabilities
        cv_preds_rf[test_idx] <- as.character(res$predictions)
        probs_list$rf <- res$probabilities
      }
      
      # SVM
      svm_fold <- tryCatch(train_svm(X_svm[train_idx, ], y_train, config), error = function(e) NULL)
      if (!is.null(svm_fold)) {
        res <- predict_svm(svm_fold, svm[train_idx, ])
        cv_probs_svm[test_idx] <- res$probabilities
        cv_preds_svm[test_idx] <- as.character(res$predictions)
        probs_list$svm <- res$probabilities
      }
      
      # XGBoost
      xgb_fold <- tryCatch(train_xgboost(X_xgb[train_idx, ], y_train, config), error = function(e) NULL)
      if (!is.null(xgb_fold)) {
        res <- predict_xgboost(xgb_fold, X_xgb[test_idx, ])
        cv_probs_xgb[test_idx] <- res$probabilities
        cv_preds_xgb[test_idx] <- as.character(res$predictions)
        probs_list$xgb <- res$probabilities
      }
      
      # KNN
      knn_fold <- tryCatch(train_knn(X_knn[train_idx, ], y_train, config), error = function(e) NULL)
      if (!is.null(knn_fold)) {
        res <- predict_knn(knn_fold, X_knn[test_idx, ])
        cv_probs_knn[test_idx] <- res$probabilities
        cv_preds_knn[test_idx] <- as.character(res$predictions)
        probs_list$knn <- res$probabilities
      }
      
      # MLP
      mlp_fold <- tryCatch(train_mlp(X_mlp[train_idx, ], y_train, config), error = function(e) NULL)
      if (!is.null(mlp_fold)) {
        res <- predict_mlp(mlp_fold, X_mlp[test_idx, ])
        cv_probs_mlp[test_idx] <- res$probabilities
        cv_preds_mlp[test_idx] <- as.character(res$predictions)
        probs_list$mlp <- res$probabilities
      }
      
      # Soft voting for this fold
      if (length(probs_list) > 1) {
        soft_res <- soft_voting(probs_list)
        cv_probs_soft[test_idx] <- soft_res$probabilities
        cv_preds_soft[test_idx] <- as.character(soft_res$predictions)
      }
    }
    
    # Calculate metrics for each model on permuted data
    calc_auroc <- function(probs, actual) {
      if (all(probs == 0)) return(NA_real_)
      roc_obj <- tryCatch(roc(actual, probs, quiet = TRUE), error = function(e) NULL)
      if (!is.null(roc_obj)) return(as.numeric(auc(roc_obj)))
      return(NA_real_)
    }
    
    calc_accuracy <- function(preds, actual) {
      if (all(preds == "")) return(NA_real_)
      return(mean(preds == as.character(actual), na.rm = TRUE))
    }
    
    permutation_results$rf_auroc[p] <- calc_auroc(cv_probs_rf, y_permuted)
    permutation_results$per_model$rf$auroc[p] <- calc_auroc(cv_probs_rf, y_permuted)
    permutation_results$per_model$rf$accuracy[p] <- calc_accuracy(cv_preds_rf, y_permuted)
    
    permutation_results$per_model$svm$auroc[p] <- calc_auroc(cv_probs_svm, y_permuted)
    permutation_results$per_model$svm$accuracy[p] <- calc_accuracy(cv_preds_svm, y_permuted)
    
    permutation_results$per_model$xgboost$auroc[p] <- calc_auroc(cv_probs_xgb, y_permuted)
    permutation_results$per_model$xgboost$accuracy[p] <- calc_accuracy(cv_preds_xgb, y_permuted)
    
    permutation_results$per_model$knn$auroc[p] <- calc_auroc(cv_probs_knn, y_permuted)
    permutation_results$per_model$knn$accuracy[p] <- calc_accuracy(cv_preds_knn, y_permuted)
    
    permutation_results$per_model$mlp$auroc[p] <- calc_auroc(cv_probs_mlp, y_permuted)
    permutation_results$per_model$mlp$accuracy[p] <- calc_accuracy(cv_preds_mlp, y_permuted)
    
    permutation_results$per_model$soft_vote$auroc[p] <- calc_auroc(cv_probs_soft, y_permuted)
    permutation_results$per_model$soft_vote$accuracy[p] <- calc_accuracy(cv_preds_soft, y_permuted)
  }
  
  return(permutation_results)
}

# =============================================================================
# PROFILE RANKING WITH CLASS-SPECIFIC RISK SCORES
# =============================================================================

rank_profiles <- function(X, y, models, config, sample_ids = NULL, annotation = NULL) {
  log_message("Ranking profiles by prediction confidence with class-specific risk scores...")
  
  all_probs <- list()
  all_probs_class0 <- list()  # Probability for class 0 (negative)
  
  # RF
  if (!is.null(models$rf)) {
    tryCatch({
      rf_probs <- predict(models$rf, X, type = "prob")
      all_probs[["rf"]] <- rf_probs[, 2]
      all_probs_class0[["rf"]] <- rf_probs[, 1]
    }, error = function(e) log_message(sprintf("RF prediction failed: %s", e$message), "WARN"))
  }
  
  # SVM
  if (!is.null(models$svm)) {
    tryCatch({
      svm_probs <- predict(models$svm, X, probability = TRUE)
      svm_attr <- attr(svm_probs, "probabilities")
      all_probs[["svm"]] <- svm_attr[, 2]
      all_probs_class0[["svm"]] <- svm_attr[, 1]
    }, error = function(e) log_message(sprintf("SVM prediction failed: %s", e$message), "WARN"))
  }
  
  # XGBoost
  if (!is.null(models$xgboost)) {
    tryCatch({
      xgb_probs <- predict(models$xgboost, as.matrix(X))
      all_probs[["xgboost"]] <- xgb_probs
      all_probs_class0[["xgboost"]] <- 1 - xgb_probs
    }, error = function(e) log_message(sprintf("XGBoost prediction failed: %s", e$message), "WARN"))
  }
  
  # KNN
  if (!is.null(models$knn_train_data) && !is.null(models$knn_train_labels)) {
    tryCatch({
      k <- ifelse(is.null(config$knn_k), 5, config$knn_k)
      knn_pred <- class::knn(models$knn_train_data, X, models$knn_train_labels, k = k, prob = TRUE)
      knn_prob <- attr(knn_pred, "prob")
      knn_prob <- ifelse(as.character(knn_pred) == levels(y)[2], knn_prob, 1 - knn_prob)
      all_probs[["knn"]] <- knn_prob
      all_probs_class0[["knn"]] <- 1 - knn_prob
    }, error = function(e) log_message(sprintf("KNN prediction failed: %s", e$message), "WARN"))
  }
  
  # MLP
  if (!is.null(models$mlp)) {
    tryCatch({
      mlp_probs <- predict(models$mlp, X, type = "raw")
      if (ncol(mlp_probs) >= 2) {
        all_probs[["mlp"]] <- mlp_probs[, 2]
        all_probs_class0[["mlp"]] <- mlp_probs[, 1]
      } else {
        all_probs[["mlp"]] <- as.vector(mlp_probs)
        all_probs_class0[["mlp"]] <- 1 - as.vector(mlp_probs)
      }
    }, error = function(e) log_message(sprintf("MLP prediction failed: %s", e$message), "WARN"))
  }
  
  # Average probabilities for ensemble
  if (length(all_probs) == 0) {
    log_message("No valid model probabilities available for ranking", "WARN")
    return(NULL)
  }
  
  # Calculate mean probability across models
  pos_class <- levels(y)[2]
  neg_class <- levels(y)[1]
  
  avg_prob <- if (length(all_probs) == 1) {
    as.numeric(all_probs[[1]])
  } else {
    as.numeric(rowMeans(do.call(cbind, all_probs), na.rm = TRUE))
  }
  
  avg_prob_class0 <- if (length(all_probs_class0) == 1) {
    as.numeric(all_probs_class0[[1]])
  } else {
    as.numeric(rowMeans(do.call(cbind, all_probs_class0), na.rm = TRUE))
  }
  
  confidence <- abs(avg_prob - 0.5) * 2
  
  # Create risk scores (scaled 0-100)
  risk_score_positive <- avg_prob * 100
  risk_score_negative <- avg_prob_class0 * 100
  
  ranking <- data.frame(
    sample_index = 1:nrow(X),
    actual_class = as.character(y),
    ensemble_probability = avg_prob,
    predicted_class = ifelse(avg_prob > 0.5, pos_class, neg_class),
    confidence = confidence,
    correct = as.character(y) == ifelse(avg_prob > 0.5, pos_class, neg_class),
    risk_score_class_0 = round(risk_score_negative, 2),
    risk_score_class_1 = round(risk_score_positive, 2)
  )
  
  # Add sample_ids if provided
  if (!is.null(sample_ids) && length(sample_ids) == nrow(X)) {
    ranking$sample_id <- sample_ids
  } else {
    ranking$sample_id <- paste0("Sample_", 1:nrow(X))
  }
  
  # Add survival data if annotation is provided and config has time/event variables
  if (!is.null(annotation) && !is.null(config$time_variable) && !is.null(config$event_variable)) {
    time_col <- config$time_variable
    event_col <- config$event_variable
    
    if (time_col %in% colnames(annotation) && event_col %in% colnames(annotation)) {
      annot_sample_col <- colnames(annotation)[1]
      
      # Create lookup for survival data by sample ID
      surv_lookup <- data.frame(
        sample_id = annotation[[annot_sample_col]],
        surv_time = suppressWarnings(as.numeric(as.character(annotation[[time_col]]))),
        surv_event = suppressWarnings(as.numeric(as.character(annotation[[event_col]]))),
        stringsAsFactors = FALSE
      )
      
      # Merge survival data into ranking by sample_id
      ranking$surv_time <- surv_lookup$surv_time[match(ranking$sample_id, surv_lookup$sample_id)]
      ranking$surv_event <- surv_lookup$surv_event[match(ranking$sample_id, surv_lookup$sample_id)]
      
      log_message(sprintf("Added survival data to profile ranking: %d samples with valid time, %d with valid event",
                          sum(!is.na(ranking$surv_time)), sum(!is.na(ranking$surv_event))))
    } else {
      log_message("Survival columns not found in annotation, skipping surv_time/surv_event in ranking", "WARN")
    }
  }
  
  ranking <- ranking[order(-ranking$confidence), ]
  ranking$rank <- 1:nrow(ranking)
  
  top_n <- ceiling(nrow(ranking) * (config$top_percent / 100))
  ranking$top_profile <- ranking$rank <= top_n
  
  return(ranking)
}

# =============================================================================
# RESULTS AGGREGATION
# =============================================================================

aggregate_results <- function(results_list, attempted_folds, valid_folds) {
  
  if (length(attempted_folds) != 1 || is.na(attempted_folds) || attempted_folds == 0) {
    log_message("No valid CV folds attempted — skipping aggregation", "WARN")
    return(list())
  }
  
  methods <- names(results_list)
  
  cv_summary <- lapply(methods, function(method) {
    
    folds <- results_list[[method]]
    if (length(folds) == 0) return(NULL)
    
    # Count valid folds
    valid_idx <- sapply(folds, function(f) {
      !is.null(f) &&
        !is.na(f$accuracy) &&
        is.finite(f$accuracy)
    })
    
    n_total <- attempted_folds
    n_valid <- valid_folds[[method]]
    
    if (n_valid < MIN_VALID_FOLD_FRACTION * n_total) {
      log_message(
        sprintf("Skipping %s: only %d/%d valid folds",
                method, n_valid, n_total),
        "WARN"
      )
      return(NULL)
    }
    
    
    folds <- folds[valid_idx]
    
    if (length(results_list[[method]]) == 0) return(NULL)
    
    metrics <- c("accuracy", "sensitivity", "specificity", "precision", 
                 "f1_score", "balanced_accuracy", "auroc", "kappa")
    
    stats <- lapply(metrics, function(m) {
      
      raw_values <- lapply(folds, function(r) {
        v <- r[[m]]
        if (is.null(v) || !is.numeric(v)) return(NA_real_)
        as.numeric(v)
      })
      
      values <- unlist(raw_values, use.names = FALSE)
      values <- values[is.finite(values)]
      
      safe_stats(values)
    })
    
    names(stats) <- metrics
    
    # Get average confusion matrix
    cms <- lapply(results_list[[method]], function(r) r$confusion_matrix)
    cms <- cms[!sapply(cms, is.null)]
    if (length(cms) > 0) {
      stats$confusion_matrix <- list(
        tp = round(mean(sapply(cms, function(x) x$tp))),
        tn = round(mean(sapply(cms, function(x) x$tn))),
        fp = round(mean(sapply(cms, function(x) x$fp))),
        fn = round(mean(sapply(cms, function(x) x$fn)))
      )
    }
    
    # Get average ROC curve
    rocs <- lapply(results_list[[method]], function(r) r$roc_curve)
    rocs <- rocs[!sapply(rocs, is.null)]
    if (length(rocs) > 0) {
      fpr_grid <- seq(0, 1, by = 0.02)
      avg_tpr <- sapply(fpr_grid, function(fpr) {
        tprs <- sapply(rocs, function(roc) {
          idx <- which.min(abs(roc$fpr - fpr))
          roc$tpr[idx]
        })
        mean(tprs, na.rm = TRUE)
      })
      stats$roc_curve <- lapply(seq_along(fpr_grid), function(i) {
        list(fpr = fpr_grid[i], tpr = avg_tpr[i])
      })
    }
    
    return(stats)
  })
  
  # After cv_summary is created
  names(cv_summary) <- methods
  
  # --- HARD VOTE FALLBACK FIX ---
  base_models <- setdiff(names(cv_summary), c("hard_vote", "soft_vote"))
  
  get_mean_accuracy <- function(m) {
    if (is.null(cv_summary[[m]]$accuracy$mean)) return(NA_real_)
    as.numeric(cv_summary[[m]]$accuracy$mean)
  }
  
  acc_means <- sapply(base_models, get_mean_accuracy)
  acc_means <- acc_means[is.finite(acc_means)]
  
  if (length(acc_means) > 0) {
    best_model <- names(which.max(acc_means))
  } else {
    best_model <- NULL
  }
  
  # Replace hard_vote if needed
  if (!is.null(cv_summary$hard_vote) &&
      !is.null(cv_summary$hard_vote$accuracy$mean) &&
      cv_summary$hard_vote$accuracy$mean == 0 &&
      !is.null(best_model)) {
    
    log_message(
      sprintf(
        "Hard vote failed (accuracy=0). Falling back to best model: %s (mean=%.3f)",
        best_model, acc_means[best_model]
      ),
      "WARN"
    )
    
    cv_summary$hard_vote <- cv_summary[[best_model]]
    cv_summary$hard_vote$fallback_from <- best_model
  }
  
  return(cv_summary)
}


# =============================================================================
# DERIVED EXPORTS (Calibration / Clustering / Stability / Expression Boxplots)
# =============================================================================

compute_calibration_curve <- function(actual, prob, n_bins = 10) {
  df <- data.frame(actual = as.character(actual), prob = as.numeric(prob))
  df <- df[!is.na(df$prob), , drop = FALSE]
  if (nrow(df) == 0) return(data.frame())
  
  # Map actual to 0/1 when possible; otherwise treat the last factor level as "positive".
  actual_factor <- factor(df$actual)
  pos_level <- tail(levels(actual_factor), 1)
  df$y01 <- ifelse(df$actual == pos_level, 1, 0)
  
  df$bin <- cut(df$prob, breaks = seq(0, 1, length.out = n_bins + 1), include.lowest = TRUE)
  
  agg <- df %>%
    group_by(bin) %>%
    summarise(
      mean_pred = mean(prob, na.rm = TRUE),
      frac_pos = mean(y01, na.rm = TRUE),
      n = dplyr::n(),
      .groups = "drop"
    )
  
  agg$bin_center <- (as.numeric(agg$mean_pred) * 100)
  agg$mean_pred_pct <- as.numeric(agg$mean_pred) * 100
  agg$frac_pos_pct <- as.numeric(agg$frac_pos) * 100
  
  return(as.data.frame(agg))
}

compute_calibration_curves_from_cv <- function(cv_predictions, n_bins = 10) {
  curves <- list()
  for (name in names(cv_predictions)) {
    df <- cv_predictions[[name]]
    if (is.null(df) || nrow(df) == 0) next
    curves[[name]] <- compute_calibration_curve(df$actual, df$prob, n_bins = n_bins)
  }
  return(curves)
}

compute_pca_embedding <- function(X, y, sample_ids = NULL) {
  
  # PCA requires at least 2 features
  if (ncol(X) < 2) {
    log_message("PCA skipped: <2 features available", "WARN")
    return(NULL)
  }
  
  if (nrow(X) < 2) return(NULL)
  
  pca <- tryCatch(
    prcomp(X, center = TRUE, scale. = FALSE),
    error = function(e) NULL
  )
  if (is.null(pca)) return(NULL)
  
  n_pc <- min(2, ncol(pca$x))
  
  coords <- as.data.frame(pca$x[, seq_len(n_pc), drop = FALSE])
  
  # Pad missing PC if only one exists
  if (n_pc == 1) {
    coords$PC2 <- 0
  }
  
  colnames(coords) <- c("x", "y")
  coords$sample_id <- if (!is.null(sample_ids)) sample_ids else rownames(X)
  coords$actual_class <- as.character(y)
  
  var_expl <- (pca$sdev^2) / sum(pca$sdev^2)
  
  variance_explained <- list(
    pc1 = as.numeric(var_expl[1]),
    pc2 = ifelse(length(var_expl) >= 2, as.numeric(var_expl[2]), 0)
  )
  
  return(list(points = coords, variance_explained = variance_explained))
}

compute_tsne_embedding <- function(X, y, sample_ids = NULL) {
  if (!tsne_available) return(NULL)
  if (nrow(X) < 4) return(NULL)
  
  perp <- min(30, floor((nrow(X) - 1) / 3))
  if (perp < 1) return(NULL)
  
  tsne_res <- tryCatch({
    Rtsne::Rtsne(as.matrix(X), dims = 2, perplexity = perp, verbose = FALSE, check_duplicates = FALSE)
  }, error = function(e) NULL)
  
  if (is.null(tsne_res)) return(NULL)
  
  coords <- as.data.frame(tsne_res$Y)
  colnames(coords) <- c("x", "y")
  coords$sample_id <- if (!is.null(sample_ids)) sample_ids else rownames(X)
  coords$actual_class <- as.character(y)
  
  return(list(points = coords))
}

compute_umap_embedding <- function(X, y, sample_ids = NULL) {
  if (!umap_available) return(NULL)
  if (nrow(X) < 4) return(NULL)
  
  n_neighbors <- min(15, nrow(X) - 1)
  if (n_neighbors < 2) return(NULL)
  
  umap_config <- umap::umap.defaults
  umap_config$n_neighbors <- n_neighbors
  
  umap_res <- tryCatch({
    umap::umap(as.matrix(X), config = umap_config)
  }, error = function(e) NULL)
  
  if (is.null(umap_res)) return(NULL)
  
  coords <- as.data.frame(umap_res$layout)
  colnames(coords) <- c("x", "y")
  coords$sample_id <- if (!is.null(sample_ids)) sample_ids else rownames(X)
  coords$actual_class <- as.character(y)
  
  return(list(points = coords))
}

compute_feature_importance_stability <- function(fold_importance, top_n = 50) {
  if (length(fold_importance) == 0) return(NULL)
  
  # fold_importance is a list of named numeric vectors (RF importance)
  feats <- unique(unlist(lapply(fold_importance, names)))
  if (length(feats) == 0) return(NULL)
  
  rank_mat <- sapply(fold_importance, function(imp) {
    v <- imp[feats]
    v[is.na(v)] <- -Inf
    rank(-v, ties.method = "average")
  })
  
  mean_rank <- rowMeans(rank_mat, na.rm = TRUE)
  sd_rank <- apply(rank_mat, 1, sd, na.rm = TRUE)
  
  freq_top <- rowMeans(rank_mat <= top_n, na.rm = TRUE)
  
  out <- data.frame(
    feature = feats,
    mean_rank = as.numeric(mean_rank),
    sd_rank = as.numeric(sd_rank),
    top_n_frequency = as.numeric(freq_top)
  )
  
  out <- out[order(out$mean_rank), ]
  rownames(out) <- NULL
  
  return(head(out, top_n))
}

#' Compute box plot statistics for top N features by class
compute_feature_boxplot_stats <- function(unscaled_expr, y, top_features, top_n = 20) {
  if (is.null(unscaled_expr) || nrow(unscaled_expr) == 0) return(NULL)
  if (is.null(top_features) || length(top_features) == 0) return(NULL)
  
  # Use only features that exist in unscaled expression
  available_features <- intersect(top_features, colnames(unscaled_expr))
  if (length(available_features) == 0) return(NULL)
  
  features_to_use <- head(available_features, top_n)
  
  result <- list()
  classes <- levels(y)
  
  for (feat in features_to_use) {
    values <- unscaled_expr[[feat]]
    
    class_stats <- lapply(classes, function(cls) {
      cls_values <- values[y == cls]
      cls_values <- cls_values[!is.na(cls_values)]
      if (length(cls_values) == 0) return(NULL)
      
      q <- quantile(cls_values, probs = c(0, 0.25, 0.5, 0.75, 1), na.rm = TRUE)
      
      list(
        class = cls,
        min = as.numeric(q[1]),
        q1 = as.numeric(q[2]),
        median = as.numeric(q[3]),
        q3 = as.numeric(q[4]),
        max = as.numeric(q[5]),
        mean = mean(cls_values, na.rm = TRUE),
        n = length(cls_values)
      )
    })
    class_stats <- Filter(Negate(is.null), class_stats)
    
    result[[feat]] <- class_stats
  }
  
  return(result)
}

#' Compute actual distributions from CV results for comparison with permuted distributions
#' @param cv_results The cv_results object from run_cv_all_methods
#' @return List with per-model AUROC and Accuracy distributions from actual CV folds
compute_actual_distributions <- function(cv_results) {
  models <- c("rf", "svm", "xgboost", "knn", "mlp", "soft_vote")
  actual_dist <- list()
  
  for (model in models) {
    if (is.null(cv_results$results[[model]])) next
    
    folds <- cv_results$results[[model]]
    auroc_vals <- sapply(folds, function(f) {
      if (is.null(f) || is.null(f$auroc) || !is.finite(f$auroc)) return(NA)
      f$auroc
    })
    accuracy_vals <- sapply(folds, function(f) {
      if (is.null(f) || is.null(f$accuracy) || !is.finite(f$accuracy)) return(NA)
      f$accuracy
    })
    
    # Remove NAs
    auroc_vals <- auroc_vals[!is.na(auroc_vals)]
    accuracy_vals <- accuracy_vals[!is.na(accuracy_vals)]
    
    if (length(auroc_vals) > 0 || length(accuracy_vals) > 0) {
      actual_dist[[model]] <- list(
        auroc = as.numeric(auroc_vals),
        accuracy = as.numeric(accuracy_vals)
      )
    }
  }
  
  if (length(actual_dist) == 0) return(NULL)
  return(actual_dist)
}

# =============================================================================
# SURVIVAL ANALYSIS FUNCTIONS
# =============================================================================

#' Perform per-gene survival analysis using Kaplan-Meier and Cox PH models
#' @param expr_data Expression matrix (rows=samples, cols=features)
#' @param annotation Data frame with survival data
#' @param time_col Name of time-to-event column
#' @param event_col Name of event status column
#' @param features Vector of feature names to analyze
#' @return List with per-gene survival statistics
perform_survival_analysis <- function(expr_data, annotation, time_col, event_col, features, sample_ids) {
  if (!survival_available) {
    log_message("Survival analysis skipped: 'survival' package not installed", "WARN")
    return(NULL)
  }
  
  log_message("Performing survival analysis...")
  
  # Validate columns exist
  if (!time_col %in% colnames(annotation)) {
    log_message(sprintf("Time variable '%s' not found in annotation. Available columns: %s", 
                        time_col, paste(colnames(annotation), collapse = ", ")), "WARN")
    return(NULL)
  }
  if (!event_col %in% colnames(annotation)) {
    log_message(sprintf("Event variable '%s' not found in annotation. Available columns: %s", 
                        event_col, paste(colnames(annotation), collapse = ", ")), "WARN")
    return(NULL)
  }
  
  log_message(sprintf("Time variable: '%s', Event variable: '%s'", time_col, event_col))
  log_message(sprintf("Annotation has %d rows, expression has %d samples", nrow(annotation), nrow(expr_data)))
  
  # Extract survival data with robust numeric conversion (suppress coercion warnings)
  surv_time <- suppressWarnings(as.numeric(as.character(annotation[[time_col]])))
  surv_event <- suppressWarnings(as.numeric(as.character(annotation[[event_col]])))
  
  # Log conversion results
  na_time_orig <- sum(is.na(annotation[[time_col]]))
  na_event_orig <- sum(is.na(annotation[[event_col]]))
  na_time_after <- sum(is.na(surv_time))
  na_event_after <- sum(is.na(surv_event))
  
  log_message(sprintf("Time values: %d total, %d NA in original, %d NA after conversion", 
                      length(surv_time), na_time_orig, na_time_after))
  log_message(sprintf("Event values: %d total, %d NA in original, %d NA after conversion", 
                      length(surv_event), na_event_orig, na_event_after))
  
  # Check for valid time values (must be positive)
  positive_time <- sum(!is.na(surv_time) & surv_time > 0)
  log_message(sprintf("Positive time values: %d", positive_time))
  
  # Remove samples with missing survival data
  valid_idx <- !is.na(surv_time) & !is.na(surv_event) & surv_time > 0
  log_message(sprintf("Valid samples for survival analysis: %d / %d", sum(valid_idx), length(valid_idx)))
  
  if (sum(valid_idx) < 10) {
    log_message("Insufficient samples with valid survival data (< 10)", "WARN")
    return(NULL)
  }
  
  per_gene_results <- list()
  features_to_analyze <- head(features, 50)  # Limit to top 50 features
  
  for (i in seq_along(features_to_analyze)) {
    feat <- features_to_analyze[i]
    if (i %% 10 == 1) show_progress(i, length(features_to_analyze), "Survival Analysis")
    
    if (!feat %in% colnames(expr_data)) next
    
    expr_values <- expr_data[[feat]][valid_idx]
    time_vals <- surv_time[valid_idx]
    event_vals <- surv_event[valid_idx]
    
    # Median split for Kaplan-Meier
    median_expr <- median(expr_values, na.rm = TRUE)
    high_group <- expr_values >= median_expr
    
    # Skip if one group is too small
    if (sum(high_group) < 3 || sum(!high_group) < 3) next
    
    tryCatch({
      # Create survival object
      surv_obj <- Surv(time_vals, event_vals)
      
      # Log-rank test
      surv_diff <- survdiff(surv_obj ~ high_group)
      logrank_p <- 1 - pchisq(surv_diff$chisq, df = 1)
      
      # Cox proportional hazards
      cox_fit <- coxph(surv_obj ~ expr_values)
      cox_summary <- summary(cox_fit)
      
      cox_hr <- as.numeric(exp(cox_fit$coefficients))
      cox_ci <- exp(confint(cox_fit))
      cox_p <- cox_summary$coefficients[, "Pr(>|z|)"]
      
      # Kaplan-Meier fits for median survival
      km_high <- survfit(surv_obj[high_group] ~ 1)
      km_low <- survfit(surv_obj[!high_group] ~ 1)
      
      # Extract median survival times
      high_median <- if (!is.na(summary(km_high)$table["median"])) summary(km_high)$table["median"] else NA
      low_median <- if (!is.na(summary(km_low)$table["median"])) summary(km_low)$table["median"] else NA
      
      per_gene_results[[feat]] <- list(
        gene = feat,
        logrank_p = as.numeric(logrank_p),
        cox_hr = as.numeric(cox_hr),
        cox_hr_lower = as.numeric(cox_ci[1]),
        cox_hr_upper = as.numeric(cox_ci[2]),
        cox_p = as.numeric(cox_p),
        high_median_surv = as.numeric(high_median),
        low_median_surv = as.numeric(low_median)
      )
    }, error = function(e) {
      log_message(sprintf("Survival analysis failed for %s: %s", feat, e$message), "WARN")
    })
  }
  
  show_progress(length(features_to_analyze), length(features_to_analyze), "Survival Analysis")
  
  log_message(sprintf("Survival analysis completed for %d features", length(per_gene_results)))
  
  return(list(
    time_variable = time_col,
    event_variable = event_col,
    per_gene = do.call(rbind, lapply(per_gene_results, function(x) {
      data.frame(x, stringsAsFactors = FALSE)
    }))
  ))
}

#' Perform survival analysis based on model risk scores
#' @param risk_scores Data frame with sample IDs and risk scores per model
#' @param annotation Annotation data frame
#' @param time_col Time-to-event column name
#' @param event_col Event status column name
#' @return List with model-specific survival results
perform_model_risk_survival <- function(rankings, annotation, time_col, event_col, sample_ids) {
  if (!survival_available) return(NULL)
  if (is.null(rankings) || nrow(rankings) == 0) return(NULL)
  
  log_message("Performing model risk score survival analysis...")
  
  # Use ensemble probability as risk score - MUST align by sample_id
  if (!"ensemble_probability" %in% colnames(rankings)) return(NULL)
  if (!"sample_id" %in% colnames(rankings)) {
    log_message("Rankings missing sample_id column, cannot align with survival data", "WARN")
    return(NULL)
  }
  
  # Match rankings to annotation by sample_id
  # First, find the sample ID column in annotation
  annot_sample_col <- NULL
  for (col in c("sample_id", "Sample_ID", "SampleID", "sample", "Sample")) {
    if (col %in% colnames(annotation)) {
      annot_sample_col <- col
      break
    }
  }
  if (is.null(annot_sample_col)) {
    # Use first column as sample ID
    annot_sample_col <- colnames(annotation)[1]
  }
  
  # Create merged data frame to ensure proper alignment
  ranking_subset <- rankings[, c("sample_id", "ensemble_probability")]
  colnames(ranking_subset) <- c("merge_id", "risk_score")
  
  annotation$merge_id <- annotation[[annot_sample_col]]
  merged <- merge(annotation, ranking_subset, by = "merge_id", all.x = FALSE, all.y = FALSE)
  
  log_message(sprintf("Matched %d samples between rankings and annotation for survival", nrow(merged)))
  
  if (nrow(merged) < 10) {
    log_message("Insufficient matched samples for survival analysis (< 10)", "WARN")
    return(NULL)
  }
  
  # Get survival data from merged (now properly aligned with risk scores)
  surv_time <- suppressWarnings(as.numeric(as.character(merged[[time_col]])))
  surv_event <- suppressWarnings(as.numeric(as.character(merged[[event_col]])))
  risk_scores <- merged$risk_score
  
  # Filter for valid survival data
  valid_idx <- !is.na(surv_time) & !is.na(surv_event) & !is.na(risk_scores) & surv_time > 0
  if (sum(valid_idx) < 10) return(NULL)
  
  tryCatch({
    time_vals <- surv_time[valid_idx]
    event_vals <- surv_event[valid_idx]
    risk_vals <- risk_scores[valid_idx]
    
    log_message(sprintf("Survival analysis: %d valid samples, risk score range: %.3f - %.3f", 
                        length(risk_vals), min(risk_vals), max(risk_vals)))
    
    # Median split
    high_risk <- risk_vals >= median(risk_vals, na.rm = TRUE)
    
    # Log group sizes and event distribution for debugging
    n_high <- sum(high_risk)
    n_low <- sum(!high_risk)
    events_high <- sum(event_vals[high_risk], na.rm = TRUE)
    events_low <- sum(event_vals[!high_risk], na.rm = TRUE)
    log_message(sprintf("Risk groups: High=%d (events=%d), Low=%d (events=%d), median cutoff=%.3f", 
                        n_high, events_high, n_low, events_low, median(risk_vals, na.rm = TRUE)))
    
    if (n_high < 3 || n_low < 3) return(NULL)
    
    surv_obj <- Surv(time_vals, event_vals)
    
    # Log-rank test
    surv_diff <- survdiff(surv_obj ~ high_risk)
    logrank_p <- 1 - pchisq(surv_diff$chisq, df = 1)
    log_message(sprintf("Log-rank chi-sq=%.3f, p=%.4e", surv_diff$chisq, logrank_p))
    
    # Cox model
    cox_fit <- coxph(surv_obj ~ risk_vals)
    cox_summary <- summary(cox_fit)
    cox_hr <- as.numeric(exp(cox_fit$coefficients))
    cox_ci <- exp(confint(cox_fit))
    cox_p <- cox_summary$coefficients[, "Pr(>|z|)"]
    
    # K-M curves for export
    km_high <- survfit(surv_obj[high_risk] ~ 1)
    km_low <- survfit(surv_obj[!high_risk] ~ 1)
    
    format_km_curve <- function(km_fit) {
      data.frame(
        time = km_fit$time,
        surv = km_fit$surv,
        lower = km_fit$lower,
        upper = km_fit$upper,
        n_risk = km_fit$n.risk,
        n_event = km_fit$n.event,
        n_censor = km_fit$n.censor
      )
    }
    
    list(
      model = "ensemble",
      stats = list(
        logrank_p = as.numeric(logrank_p),
        cox_hr = as.numeric(cox_hr),
        cox_hr_lower = as.numeric(cox_ci[1]),
        cox_hr_upper = as.numeric(cox_ci[2]),
        cox_p = as.numeric(cox_p)
      ),
      km_curve_high = format_km_curve(km_high),
      km_curve_low = format_km_curve(km_low)
    )
  }, error = function(e) {
    log_message(sprintf("Model risk survival failed: %s", e$message), "WARN")
    return(NULL)
  })
}

# =============================================================================
# UNIFIED SURVIVAL ANALYSIS RUNNER
# =============================================================================

#' Run complete survival analysis (per-gene and model-based)
#' @param X Expression data (samples x features)
#' @param y Target variable
#' @param sample_ids Sample IDs
#' @param ranking Profile ranking data frame
#' @param config Configuration list
#' @param annotation Annotation data frame
#' @param annot_sample_col Sample ID column name in annotation
#' @param selected_features Selected feature names
#' @return List with survival analysis results or NULL
run_survival_analysis <- function(X, y, sample_ids, ranking, config, annotation, annot_sample_col, selected_features) {
  if (!survival_available) {
    log_message("Survival package not available, skipping survival analysis", "WARN")
    return(NULL)
  }
  
  if (is.null(config$time_variable) || is.null(config$event_variable) ||
      config$time_variable == "" || config$event_variable == "") {
    log_message("No time/event variables specified, skipping survival analysis", "INFO")
    return(NULL)
  }
  
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("SURVIVAL ANALYSIS")
  
  # Per-gene survival analysis
  per_gene <- perform_survival_analysis(
    expr_data = X,
    annotation = annotation,
    time_col = config$time_variable,
    event_col = config$event_variable,
    features = selected_features,
    sample_ids = sample_ids
  )
  
  # Model risk score survival analysis
  model_risk <- perform_model_risk_survival(
    rankings = ranking,
    annotation = annotation,
    time_col = config$time_variable,
    event_col = config$event_variable,
    sample_ids = sample_ids
  )
  
  if (is.null(per_gene) && is.null(model_risk)) {
    log_message("No survival analysis results generated", "WARN")
    return(NULL)
  }
  
  result <- list(
    time_variable = config$time_variable,
    event_variable = config$event_variable,
    per_gene = if (!is.null(per_gene)) per_gene$per_gene else NULL,
    model_risk_scores = if (!is.null(model_risk)) list(model_risk) else NULL
  )
  
  log_message("Survival analysis completed")
  return(result)
}

# =============================================================================
# JSON EXPORT
# =============================================================================

export_to_json <- function(results, config, output_path) {
  log_message(sprintf("Exporting results to: %s", output_path))
  
  # Remove redundant info from config
  config_clean <- config[!vapply(config, is.list, logical(1))]
  
  output <- list(
    metadata = list(
      generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
      config = config_clean,
      r_version = R.version.string
    ),
    preprocessing = results$preprocessing_stats,
    model_performance = results$cv_summary,
    feature_importance = if (!is.null(results$feature_importance)) {
      head(results$feature_importance, 50)
    } else NULL,
    feature_importance_stability = results$feature_importance_stability,
    feature_boxplot_stats = results$feature_boxplot_stats,
    calibration_curves = results$calibration_curves,
    clustering = results$clustering,
    permutation_testing = if (!is.null(results$permutation)) {
      list(
        rf_oob_error = list(
          permuted_mean = mean(results$permutation$rf_oob_error, na.rm = TRUE),
          permuted_sd = sd(results$permutation$rf_oob_error, na.rm = TRUE),
          original = results$original_metrics$rf_oob_error,
          p_value = mean(results$permutation$rf_oob_error <= 
                           results$original_metrics$rf_oob_error, na.rm = TRUE)
        ),
        rf_auroc = list(
          permuted_mean = mean(results$permutation$rf_auroc, na.rm = TRUE),
          permuted_sd = sd(results$permutation$rf_auroc, na.rm = TRUE),
          original = results$original_metrics$rf_auroc,
          p_value = mean(results$permutation$rf_auroc >= 
                           results$original_metrics$rf_auroc, na.rm = TRUE)
        )
      )
    } else NULL,
    permutation_distributions = if (!is.null(results$permutation$per_model)) {
      results$permutation$per_model
    } else NULL,
    actual_distributions = results$actual_distributions,
    profile_ranking = if (!is.null(results$ranking)) {
      list(top_profiles = results$ranking[results$ranking$top_profile, ],
           all_rankings = results$ranking)
    } else NULL,
    selected_features = results$selected_features,
    survival_analysis = results$survival_analysis
  )
  
  json_output <- toJSON(output, auto_unbox = TRUE, pretty = TRUE, digits = 6)
  writeLines(json_output, output_path)
  
  log_message("Results exported successfully")
  return(invisible(output_path))
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

run_pipeline <- function(config) {
  # Apply fast mode settings if enabled
  
  log_message("Effective configuration:")
  print(config)
  
  config <- get_effective_config(config)
  
  start_time <- Sys.time()
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("Starting Multi-Method ML Diagnostic and Prognostic Classifier")
  if (config$analysis_mode == "fast") {
    log_message(">>> FAST ANALYSIS MODE (Testing Only) <<<", "WARN")
  }
  log_message(paste(rep("=", 60), collapse = ""))
  
  if (!dir.exists(config$output_dir)) dir.create(config$output_dir, recursive = TRUE)
  
  # Load data from expression matrix + annotation
  data <- load_data(config)
  
  # Feature selection
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("FEATURE SELECTION")
  
  selected_features <- perform_feature_selection(
    if (config$scale_data) data$X_scaled else data$X_raw, data$y, config$feature_selection_method,
    config$max_features, config$seed
  )
  log_message(sprintf("Selected %d features", length(selected_features)))
  
  # Cross-validation
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("CROSS-VALIDATION")
  
  cv_results <- run_cv_all_methods(
    X_raw    = data$X_raw,
    X_scaled = data$X_scaled,
    y        = data$y,
    config   = config,
    selected_features = selected_features
  )
  
  log_message(sprintf(
    "CV summary: attempted=%d | RF=%d | SVM=%d | KNN=%d | MLP=%d | HARD=%d | SOFT=%d",
    cv_results$attempted_folds,
    cv_results$valid_folds$rf,
    cv_results$valid_folds$svm,
    cv_results$valid_folds$knn,
    cv_results$valid_folds$mlp,
    cv_results$valid_folds$hard_vote,
    cv_results$valid_folds$soft_vote
  ))
  
  cv_summary <- aggregate_results(
    results_list    = cv_results$results,
    attempted_folds = cv_results$attempted_folds,
    valid_folds     = cv_results$valid_folds
  )
  
  # Derived exports from CV
  calibration_curves <- compute_calibration_curves_from_cv(cv_results$cv_predictions, n_bins = 10)
  feature_importance_stability <- compute_feature_importance_stability(cv_results$fold_importance, top_n = 50)
  
  # Train final models
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("TRAINING FINAL MODELS")
  
  X_selected <- if (config$scale_data) data$X_scaled else data$X_raw[, selected_features, drop = FALSE]
  
  final_models <- list(
    rf = tryCatch(train_rf(X_selected, data$y, config), error = function(e) NULL),
    svm = tryCatch(train_svm(X_selected, data$y, config), error = function(e) NULL),
    xgboost = tryCatch(train_xgboost(X_selected, data$y, config), error = function(e) NULL),
    knn = tryCatch(train_knn(X_selected, data$y, config), error = function(e) NULL),
    mlp = tryCatch(train_mlp(X_selected, data$y, config), error = function(e) NULL)
  )
  
  original_metrics <- list(
    rf_oob_error = if (!is.null(final_models$rf)) final_models$rf$oob_error else NA,
    rf_auroc = if (!is.null(cv_summary$rf$auroc)) cv_summary$rf$auroc$mean else NA
  )
  
  best_model <- select_best_model(cv_summary)
  
  use_soft <- !is.null(cv_summary$soft_vote) &&
    !is.null(cv_summary$soft_vote$accuracy) &&
    is.finite(cv_summary$soft_vote$accuracy$mean)
  
  use_hard <- !is.null(cv_summary$hard_vote) &&
    !is.null(cv_summary$hard_vote$accuracy) &&
    is.finite(cv_summary$hard_vote$accuracy$mean)
  
  use_ensemble <- cv_results$ensemble_valid && (use_soft || use_hard)
  
  
  if (!use_ensemble) {
    log_message(
      sprintf("Ensemble invalid — falling back to best model: %s", best_model),
      "WARN"
    )
  }
  
  log_message(sprintf("Best single model selected: %s", best_model))
  
  # Permutation testing
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("PERMUTATION TESTING")
  
  permutation_results <- run_permutation_test(if (config$scale_data) data$X_scaled else data$X_raw, data$y, config, selected_features)
  
  # Profile ranking
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("PROFILE RANKING")
  
  ranking <- rank_profiles(X_selected, data$y, final_models, config, sample_ids = data$sample_ids, annotation = data$annotation)
  
  # Generate final predictions
  final_predictions <- ranking$predicted_class
  final_probabilities <- ranking$ensemble_probability
  final_model_used <- "ensemble"
  
  if (!use_ensemble && !is.null(best_model)) {
    
    final_model_used <- best_model
    
    model_obj <- final_models[[best_model]]
    
    if (!best_model %in% c("rf", "svm", "xgboost", "knn", "mlp")) {
      stop(sprintf("Unknown model '%s' selected as best model", best_model))
    }
    
    pred_fun <- get(paste0("predict_", best_model))
    
    pred_res <- pred_fun(model_obj, X_selected)
    
    final_predictions <- pred_res$predictions
    final_probabilities <- pred_res$probabilities
  }
  
  if (use_ensemble) {
    final_model_used <- if (use_soft) "soft_vote" else "hard_vote"
  }
  
  # Clustering exports (PCA, t-SNE, UMAP)
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("DIMENSIONALITY REDUCTION (PCA, t-SNE, UMAP)")
  
  clustering <- list(
    pca  = if (ncol(X_selected) >= 2)
      compute_pca_embedding(
        if (isTRUE(config$model_scaling$dr)) data$X_scaled else data$X_raw,
        data$y,
        sample_ids = data$sample_ids
      )
    else NULL,
    tsne = compute_tsne_embedding(
      if (isTRUE(config$model_scaling$dr)) data$X_scaled else data$X_raw,
      data$y,
      sample_ids = data$sample_ids
    ),
    umap = compute_umap_embedding(
      if (isTRUE(config$model_scaling$dr)) data$X_scaled else data$X_raw,
      data$y,
      sample_ids = data$sample_ids
    )
  )
  
  if (is.null(clustering$tsne)) log_message("t-SNE not computed (Rtsne package not available or insufficient samples)", "WARN")
  if (is.null(clustering$umap)) log_message("UMAP not computed (umap package not available or insufficient samples)", "WARN")
  
  # Compute feature boxplot stats for top features
  top_feature_names <- if (!is.null(cv_results$feature_importance$feature)) {
    head(cv_results$feature_importance$feature, 20)
  } else {
    head(selected_features, 20)
  }
  feature_boxplot_stats <- compute_feature_boxplot_stats(data$X_raw, data$y, top_feature_names, top_n = 20)
  
  # Survival analysis (if time/event variables provided)
  survival_analysis <- run_survival_analysis(
    X = data$X_raw[, selected_features, drop = FALSE],
    y = data$y,
    sample_ids = data$sample_ids,
    ranking = ranking,
    config = config,
    annotation = data$annotation,
    annot_sample_col = data$annot_sample_col,
    selected_features = selected_features
  )
  
  # Compute actual distributions from CV folds for comparison charts
  actual_distributions <- compute_actual_distributions(cv_results)
  
  # Export results
  results <- list(
    cv_summary = cv_summary,
    feature_importance = cv_results$feature_importance,
    feature_importance_stability = feature_importance_stability,
    feature_boxplot_stats = feature_boxplot_stats,
    calibration_curves = calibration_curves,
    clustering = clustering,
    permutation = permutation_results,
    original_metrics = original_metrics,
    fold_metrics = cv_results$fold_metrics,
    actual_distributions = actual_distributions,
    ranking = ranking,
    selected_features = selected_features,
    preprocessing_stats = data$preprocessing_stats,
    dataset_name = basename(config$expression_matrix_file),
    survival_analysis = survival_analysis
  )
  
  results$final_prediction_strategy <- list(
    method_used = final_model_used,
    ensemble_valid = use_ensemble,
    best_single_model = best_model
  )
  
  output_path <- file.path(config$output_dir, config$output_json)
  export_to_json(results, config, output_path)
  
  # Save models with proper XGBoost serialization
  # XGBoost models need special serialization - convert to raw bytes
  if (!is.null(final_models$xgboost) && inherits(final_models$xgboost, "xgb.Booster")) {
    log_message("Serializing XGBoost model using xgb.save.raw()...")
    final_models$xgboost_raw <- xgb.save.raw(final_models$xgboost)
    final_models$xgboost <- NULL  # Remove the pointer-based object
    final_models$xgboost_serialized <- TRUE
  }
  
  models_path <- file.path(config$output_dir, "trained_models.rds")
  saveRDS(final_models, models_path)
  log_message(sprintf("Models saved to: %s", models_path))
  log_message("NOTE: To load XGBoost model, use: models$xgboost <- xgb.load.raw(models$xgboost_raw)")
  
  end_time <- Sys.time()
  log_message(sprintf("Pipeline completed in %.2f minutes",
                      as.numeric(difftime(end_time, start_time, units = "mins"))))
  
  return(invisible(results))
}

# =============================================================================
# BATCH PROCESSING
# =============================================================================

#' Run batch analysis on multiple datasets
#' @param batch_config List containing batch_datasets configuration
#' @return Combined results from all datasets
run_batch_pipeline <- function(batch_config) {
  if (is.null(batch_config$batch_datasets) || length(batch_config$batch_datasets) == 0) {
    stop("No batch datasets specified. Set config$batch_datasets to a list of dataset configurations.")
  }
  
  log_message(paste(rep("=", 60), collapse = ""))
  log_message(sprintf("BATCH PROCESSING: %d datasets", length(batch_config$batch_datasets)))
  log_message(paste(rep("=", 60), collapse = ""))
  
  batch_start <- Sys.time()
  all_results <- list()
  
  for (i in seq_along(batch_config$batch_datasets)) {
    dataset <- batch_config$batch_datasets[[i]]
    dataset_name <- dataset$name %||% paste0("Dataset_", i)
    
    log_message(paste(rep("-", 40), collapse = ""))
    log_message(sprintf("Processing dataset %d/%d: %s", i, length(batch_config$batch_datasets), dataset_name))
    log_message(paste(rep("-", 40), collapse = ""))
    
    # Create dataset-specific config
    dataset_config <- batch_config
    dataset_config$expression_matrix_file <- dataset$expr
    dataset_config$annotation_file <- dataset$annot
    dataset_config$output_json <- paste0(dataset_name, "_results.json")
    dataset_config$batch_datasets <- NULL  # Prevent recursion
    
    # Run pipeline for this dataset
    tryCatch({
      result <- run_pipeline(dataset_config)
      result$dataset_name <- dataset_name
      all_results[[dataset_name]] <- result
      log_message(sprintf("Dataset '%s' completed successfully", dataset_name))
    }, error = function(e) {
      log_message(sprintf("Dataset '%s' failed: %s", dataset_name, e$message), "ERROR")
      all_results[[dataset_name]] <- list(error = e$message, dataset_name = dataset_name)
    })
  }
  
  # Export combined batch results
  batch_output <- list(
    metadata = list(
      generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
      batch_mode = TRUE,
      n_datasets = length(batch_config$batch_datasets),
      r_version = R.version.string
    ),
    datasets = all_results,
    cv_summary = create_batch_summary(all_results)
  )
  
  batch_output_path <- file.path(batch_config$output_dir, "batch_results.json")
  json_output <- toJSON(batch_output, auto_unbox = TRUE, pretty = TRUE, digits = 6)
  writeLines(json_output, batch_output_path)
  
  batch_end <- Sys.time()
  log_message(paste(rep("=", 60), collapse = ""))
  log_message(sprintf("Batch processing completed in %.2f minutes", 
                      as.numeric(difftime(batch_end, batch_start, units = "mins"))))
  log_message(sprintf("Combined results exported to: %s", batch_output_path))
  
  return(invisible(batch_output))
}

#' Create summary of batch results for comparison
create_batch_summary <- function(all_results) {
  summary_df <- lapply(names(all_results), function(name) {
    result <- all_results[[name]]
    if (!is.null(result$error)) {
      return(data.frame(
        dataset = name,
        status = "failed",
        rf_auroc = NA,
        soft_vote_auroc = NA,
        n_features = NA
      ))
    }
    
    data.frame(
      dataset = name,
      status = "success",
      rf_auroc = if (!is.null(result$cv_summary$rf$auroc)) result$cv_summary$rf$auroc$mean else NA,
      soft_vote_auroc = if (!is.null(result$cv_summary$soft_vote$auroc)) result$cv_summary$soft_vote$auroc$mean else NA,
      n_features = length(result$selected_features)
    )
  })
  
  do.call(rbind, summary_df)
}

# Null-coalescing operator
`%||%` <- function(x, y) if (is.null(x)) y else x

# =============================================================================
# RUN
# =============================================================================

if (!interactive()) {
  if (!is.null(config$batch_datasets) && length(config$batch_datasets) > 0) {
    run_batch_pipeline(config)
  } else {
    run_pipeline(config)
  }
}