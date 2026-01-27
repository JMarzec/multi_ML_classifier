#!/usr/bin/env Rscript
# =============================================================================
# Multi-Method ML Classifier - FULL DATASET TRAINING VERSION v2
# =============================================================================
# Maintains exact JSON output format as original CV version
# Trains on 100% of data without cross-validation
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

# Optional survival analysis
survival_available <- requireNamespace("survival", quietly = TRUE)
if (survival_available) library(survival)

# =============================================================================
# PARAMETERS
# =============================================================================

MIN_VALID_FOLD_FRACTION <- 0.3
VAR_QUANTILE <- 0.25
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
  
  make_option(c("-p", "--n_permutations"),
              type = "integer",
              default = 100,
              help = "Number of permutations [default: %default]"),
  
  make_option(c("--time"),
              type = "character",
              default = NULL,
              help = "Time-to-event column for survival analysis"),
  
  make_option(c("--event"),
              type = "character",
              default = NULL,
              help = "Event/censoring indicator column for survival analysis (1=event, 0=censored)")
)

opt <- parse_args(OptionParser(option_list = option_list))

# Optional libraries
tsne_available <- requireNamespace("Rtsne", quietly = TRUE)
umap_available <- requireNamespace("umap", quietly = TRUE)

if (tsne_available) library(Rtsne)
if (umap_available) library(umap)

# =============================================================================
# CONFIGURATION
# =============================================================================

config <- list(
  expression_matrix_file = "expression_matrix.txt",
  annotation_file = "sample_annotation.txt",
  target_variable = "diagnosis",
  analysis_mode = "full",
  scale_data = SCALE_DATA_DEFAULT,
  model_scaling <- MODEL_SCALING,
  seed = 42,
  top_percent = 10,
  feature_selection_method = "stepwise",
  max_features = 50,
  n_permutations = 100,
  rf_ntree = 500,
  rf_mtry = NULL,
  svm_kernel = "radial",
  svm_cost = 1,
  svm_gamma = NULL,
  xgb_nrounds = 100,
  xgb_max_depth = 6,
  xgb_eta = 0.3,
  knn_k = 5,
  mlp_size = 10,
  mlp_decay = 0.01,
  mlp_maxit = 200,
  batch_datasets = NULL,
  time_variable = NULL,   # Survival analysis: time-to-event column
  event_variable = NULL,  # Survival analysis: event status column
  output_dir = "./results",
  output_json = NULL  # Will be auto-generated from annotation file if not specified
)

# Override from command-line
if (!is.null(opt$expr)) config$expression_matrix_file <- opt$expr
if (!is.null(opt$annot)) config$annotation_file <- opt$annot
if (!is.null(opt$target)) config$target_variable <- opt$target
if (!is.null(opt$mode)) config$analysis_mode <- opt$mode
if (!is.null(opt$scale)) config$scale_data <- opt$scale
if (!is.null(opt$outdir)) config$output_dir <- opt$outdir
if (!is.null(opt$seed)) config$seed <- opt$seed
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
# FAST MODE
# =============================================================================

get_effective_config <- function(config) {
  if (config$analysis_mode == "fast") {
    log_message("FAST MODE ENABLED", "WARN")
    config$n_permutations <- 10
    config$rf_ntree <- 50
    config$xgb_nrounds <- 20
    config$mlp_maxit <- 50
    config$max_features <- min(config$max_features, 10)
    config$feature_selection_method <- "none"
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

# =============================================================================
# DATA LOADING
# =============================================================================

load_and_preprocess_data <- function(config) {
  log_message("Loading expression matrix...")
  expr_mat <- read.delim(config$expression_matrix_file, 
                         stringsAsFactors = FALSE,
                         row.names = 1,
                         check.names = FALSE)
  
  log_message("Loading sample annotations...")
  annot <- read.delim(config$annotation_file, stringsAsFactors = FALSE)
  
  if (!(config$target_variable %in% colnames(annot))) {
    stop(sprintf("Target variable '%s' not found", config$target_variable))
  }
  
  # Get sample IDs from expression matrix
  sample_ids <- colnames(expr_mat)
  log_message(sprintf("Expression matrix has %d samples", length(sample_ids)))
  
  # Find sample ID column in annotation (first column or look for common pattern)
  annot_sample_col <- colnames(annot)[1]
  annot_samples <- annot[[annot_sample_col]]
  
  # Check for target variable
  if (!config$target_variable %in% colnames(annot)) {
    stop(sprintf("Target variable '%s' not found in annotation file. Available: %s",
                 config$target_variable, paste(colnames(annot), collapse = ", ")))
  }
  
  # Find common samples
  common_samples <- intersect(sample_ids, annot_samples)
  log_message(sprintf("Found %d common samples between expression matrix and annotation", length(common_samples)))
  
  if (length(common_samples) == 0) {
    stop("No common samples found. Ensure column names in expression matrix match sample IDs in annotation.")
  }
  
  # Filter and align data
  expr_mat <- expr_mat[, colnames(expr_mat) %in% common_samples, drop = FALSE]
  annot <- annot[annot[[annot_sample_col]] %in% common_samples, ]
  
  # Sort BOTH to ensure alignment, then update sample_ids AFTER sorting
  expr_mat <- expr_mat[, order(colnames(expr_mat)), drop = FALSE]
  annot <- annot[order(annot[[annot_sample_col]]), ]
  
  # Update sample_ids AFTER sorting to match the sorted order
  sample_ids <- colnames(expr_mat)
  
  # Verify alignment before extracting target
  if (!identical(sample_ids, annot[[annot_sample_col]])) {
    log_message("WARNING: Sample order mismatch detected, re-aligning...", "WARN")
    # Force exact alignment: reorder annot to match expr_mat column order
    annot <- annot[match(sample_ids, annot[[annot_sample_col]]), ]
  }
  
  log_message(sprintf("Sample alignment verified: %d samples", length(sample_ids)))
  
  # Extract target variable
  y <- factor(annot[[config$target_variable]])
  
  # Check for and remove samples with NA in target
  na_idx <- is.na(y)
  if (any(na_idx)) {
    log_message(sprintf("Removing %d samples with NA in target variable", sum(na_idx)), "WARN")
    keep_samples <- !na_idx
    expr_mat <- expr_mat[, keep_samples, drop = FALSE]
    y <- y[keep_samples]
    sample_ids <- sample_ids[keep_samples]
  }
  
  
  ## Transpose
  expr_mat <- as.data.frame(t(expr_mat))
  
  # Ensure expr_mat is numeric
  expr_mat <- as.data.frame(lapply(expr_mat, function(x) {
    if (is.character(x)) as.numeric(x) else x
  }))
  
  # Remove constant columns
  constant_cols <- sapply(expr_mat, function(x) length(unique(x[!is.na(x)])) <= 1)
  if (any(constant_cols)) {
    log_message(sprintf("Removing %d constant columns", sum(constant_cols)), "WARN")
    preprocessing_stats$constant_features_removed <- sum(constant_cols)
    expr_mat <- expr_mat[, !constant_cols]
  }
  
  # Handle missing values
  if (any(is.na(expr_mat))) {
    log_message("Imputing missing values with median", "WARN")
    expr_mat <- as.data.frame(lapply(expr_mat, function(x) {
      if (is.numeric(x)) x[is.na(x)] <- median(x, na.rm = TRUE)
      return(x)
    }))
  }
  
  # Store unscaled expression for boxplot export (subset to current features)
  X_raw <- expr_mat
  
  # --------------------------------------------------
  # Optional Z-score scaling (per feature)
  # --------------------------------------------------
  
  X_scaled <- expr_mat
  
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
  
  unscaled_expr <- X_raw[keep_features, , drop = FALSE]
  
  # Final consistency check
  if (nrow(X_raw) != length(y) || nrow(X_raw) != length(sample_ids)) {
    stop(sprintf("Data dimension mismatch: X_raw has %d rows, y has %d elements, sample_ids has %d elements",
                 nrow(X_raw), length(y), length(sample_ids)))
  }
  
  log_message(sprintf("Data dimensions verified: %d samples, %d features", nrow(X_raw), ncol(X_raw)))
  
  # Convert table to list for JSON compatibility
  class_dist <- table(y)
  class_dist_list <- setNames(as.numeric(class_dist), names(class_dist))
  
  preprocessing_stats <- list(
    original_samples = length(sample_ids),
    original_features = nrow(expr_mat),
    missing_values = 0,
    missing_pct = 0,
    class_distribution = class_dist_list,
    constant_features_removed = 0,
    # Full training mode - no train/test split
    full_training_mode = TRUE,
    cv_folds = NULL,
    cv_repeats = NULL,
    train_samples_per_fold = length(sample_ids),
    test_samples_per_fold = 0,
    train_class_distribution = class_dist_list,
    test_class_distribution = NULL
  )
  
  return(list(
    X_raw = X_raw,
    X_scaled = X_scaled,
    y = y,
    sample_ids = sample_ids,
    unscaled_expr = unscaled_expr,
    preprocessing_stats = preprocessing_stats,
    n_samples = nrow(X_raw),
    n_features = ncol(X_raw)
  ))
}

# =============================================================================
# FEATURE SELECTION
# =============================================================================

perform_feature_selection <- function(X, y, config) {
  # Enforce max_features limit (hard cap at 50)
  max_features <- min(config$max_features, 50)
  log_message(sprintf("Feature selection: method=%s, max_features=%d", config$feature_selection_method, max_features))
  
  if (config$feature_selection_method == "none") {
    # For "none" method, select top features by variance (capped at max_features)
    if (ncol(X) <= max_features) {
      selected_features <- colnames(X)
    } else {
      log_message(sprintf("Capping features from %d to %d using variance ranking", ncol(X), max_features))
      feature_var <- apply(X, 2, var, na.rm = TRUE)
      top_idx <- order(feature_var, decreasing = TRUE)[1:max_features]
      selected_features <- colnames(X)[top_idx]
    }
    importance_df <- data.frame(
      feature = selected_features,
      importance = rep(1, length(selected_features)),
      stringsAsFactors = FALSE
    )
    return(list(
      selected_features = selected_features,
      feature_importance = importance_df,
      feature_importance_stability = NULL
    ))
  }
  
  log_message("Performing feature selection...")
  
  # Use RF for feature importance
  rf_temp <- randomForest(x = X, y = y, ntree = 100, importance = TRUE)
  importance_scores <- importance(rf_temp)[, "MeanDecreaseGini"]
  
  n_select <- min(max_features, ncol(X))
  top_indices <- order(importance_scores, decreasing = TRUE)[1:n_select]
  selected_features <- colnames(X)[top_indices]
  
  importance_df <- data.frame(
    feature = names(sort(importance_scores, decreasing = TRUE)),
    importance = sort(importance_scores, decreasing = TRUE),
    stringsAsFactors = FALSE
  )
  
  # Create stability metrics (single value since no CV)
  stability_df <- lapply(selected_features, function(feat) {
    list(
      feature = feat,
      mean_rank = which(importance_df$feature == feat),
      sd_rank = 0,
      top_n_frequency = 1
    )
  })
  
  log_message(sprintf("Selected %d features (max allowed: %d)", length(selected_features), max_features))
  
  return(list(
    selected_features = selected_features,
    feature_importance = head(importance_df, max_features),
    feature_importance_stability = stability_df
  ))
}

# =============================================================================
# MODEL TRAINING
# =============================================================================

train_all_models <- function(X_raw, X_scaled, y, config) {
  log_message("Training models on full dataset...")
  set.seed(config$seed)
  
  results <- list()
  
  X_rf  <- if (isTRUE(config$model_scaling$rf)) X_scaled else X_raw
  X_svm <- if (isTRUE(config$model_scaling$svm)) X_scaled else X_raw
  X_xgb <- if (isTRUE(config$model_scaling$xgboost)) X_scaled else X_raw
  X_knn <- if (isTRUE(config$model_scaling$knn)) X_scaled else X_raw
  X_mlp <- if (isTRUE(config$model_scaling$mlp)) X_scaled else X_raw
  
  
  log_message(sprintf(
    "Scaling settings: RF=%s | SVM=%s | XGB=%s | KNN=%s | MLP=%s | DR=%s",
    config$model_scaling$rf,
    config$model_scaling$svm,
    config$model_scaling$xgboost,
    config$model_scaling$knn,
    config$model_scaling$mlp,
    config$model_scaling$dr
  ))
  
  # Random Forest
  log_message("  Training Random Forest...")
  mtry_val <- if (is.null(config$rf_mtry)) max(1, floor(sqrt(ncol(X_raw)))) else config$rf_mtry
  rf_model <- randomForest(x = X_rf, y = y, ntree = config$rf_ntree, mtry = mtry_val, importance = TRUE)
  rf_pred <- rf_model$predicted
  rf_prob <- rf_model$votes[, levels(y)[2]]
  results$rf <- compute_metrics(rf_pred, y, rf_prob)
  results$rf$model <- rf_model
  
  # SVM
  log_message("  Training SVM...")
  gamma_val <- if (is.null(config$svm_gamma)) 1/ncol(X_raw) else config$svm_gamma
  svm_model <- svm(x = X_svm, y = y, kernel = config$svm_kernel, 
                   cost = config$svm_cost, gamma = gamma_val, probability = TRUE)
  svm_pred <- predict(svm_model, X_raw, probability = TRUE)
  svm_prob <- attr(svm_pred, "probabilities")[, levels(y)[2]]
  results$svm <- compute_metrics(svm_pred, y, svm_prob)
  results$svm$model <- svm_model
  
  # XGBoost
  log_message("  Training XGBoost...")
  y_numeric <- as.numeric(y) - 1
  xgb_matrix <- xgb.DMatrix(data = as.matrix(X_xgb), label = y_numeric)
  xgb_model <- xgb.train(
    data = xgb_matrix,  # xgb_matrix should be created with label included
    params = list(
      objective = "binary:logistic",
      learning_rate = 0.3  # changed from 'eta'
    ),
    nrounds = config$xgb_nrounds,
    verbose = 0  # or remove this line
  )
  xgb_prob <- predict(xgb_model, xgb_matrix)
  xgb_pred <- factor(ifelse(xgb_prob > 0.5, levels(y)[2], levels(y)[1]), levels = levels(y))
  results$xgboost <- compute_metrics(xgb_pred, y, xgb_prob)
  results$xgboost$model <- xgb_model
  
  # KNN
  log_message("  Training KNN...")
  knn_pred <- knn(train = X_knn, test = X_raw, cl = y, k = config$knn_k, prob = TRUE)
  knn_prob <- attr(knn_pred, "prob")
  knn_prob <- ifelse(knn_pred == levels(y)[2], knn_prob, 1 - knn_prob)
  results$knn <- compute_metrics(knn_pred, y, knn_prob)
  results$knn$model <- list(X_train = X_raw, y_train = y, k = config$knn_k)
  
  # MLP
  log_message("  Training MLP...")
  y_mlp <- class.ind(y)
  mlp_model <- nnet(x = X_mlp, y = y_mlp, size = config$mlp_size,
                    decay = config$mlp_decay, maxit = config$mlp_maxit, trace = FALSE)
  mlp_pred_mat <- predict(mlp_model, X_raw)
  
  # Identify positive class (same convention as elsewhere)
  pos_class <- levels(y)[2]
  
  # Use column name if present
  if (!is.null(colnames(mlp_pred_mat)) && pos_class %in% colnames(mlp_pred_mat)) {
    mlp_prob <- mlp_pred_mat[, pos_class]
  } else {
    # fallback: assume 2nd column is positive
    mlp_prob <- mlp_pred_mat[, 2]
  }
  mlp_pred <- factor(
    ifelse(mlp_prob > 0.5, levels(y)[2], levels(y)[1]),
    levels = levels(y)
  )
  results$mlp <- compute_metrics(mlp_pred, y, mlp_prob)
  results$mlp$model <- mlp_model
  
  # Ensemble
  log_message("  Creating ensemble...")
  all_preds <- data.frame(
    rf = rf_pred, svm = svm_pred, xgboost = xgb_pred,
    knn = knn_pred, mlp = mlp_pred
  )
  hard_vote <- apply(all_preds, 1, function(row) names(which.max(table(row))))
  hard_vote <- factor(hard_vote, levels = levels(y))
  
  all_probs <- data.frame(rf = rf_prob, svm = svm_prob, xgboost = xgb_prob,
                          knn = knn_prob, mlp = mlp_prob)
  soft_vote_prob <- rowMeans(all_probs, na.rm = TRUE)
  soft_vote <- factor(ifelse(soft_vote_prob > 0.5, levels(y)[2], levels(y)[1]), levels = levels(y))
  
  results$hard_vote <- compute_metrics(hard_vote, y, soft_vote_prob)
  results$soft_vote <- compute_metrics(soft_vote, y, soft_vote_prob)
  
  return(results)
}

compute_metrics <- function(pred, actual, prob) {
  cm <- confusionMatrix(pred, actual)
  
  # Calculate all metrics
  accuracy <- as.numeric(cm$overall["Accuracy"])
  sensitivity <- as.numeric(cm$byClass["Sensitivity"])
  specificity <- as.numeric(cm$byClass["Specificity"])
  precision <- as.numeric(cm$byClass["Pos Pred Value"])
  
  f1 <- 2 * (precision * sensitivity) / (precision + sensitivity)
  if (is.na(f1) || !is.finite(f1)) f1 <- 0
  
  balanced_accuracy <- (sensitivity + specificity) / 2
  kappa <- as.numeric(cm$overall["Kappa"])
  
  # ROC and AUROC
  roc_obj <- tryCatch({
    roc(actual, prob, quiet = TRUE)
  }, error = function(e) NULL)
  
  auroc <- if (!is.null(roc_obj)) as.numeric(auc(roc_obj)) else NA
  
  # ROC curve points
  roc_curve <- if (!is.null(roc_obj)) {
    coords_df <- coords(roc_obj, x = seq(0, 1, by = 0.02), input = "specificity", ret = c("sensitivity", "specificity"))
    lapply(1:nrow(coords_df), function(i) {
      list(fpr = 1 - coords_df$specificity[i], tpr = coords_df$sensitivity[i])
    })
  } else NULL
  
  # Confusion matrix
  cm_table <- cm$table
  confusion <- list(
    tp = as.numeric(cm_table[2, 2]),
    tn = as.numeric(cm_table[1, 1]),
    fp = as.numeric(cm_table[2, 1]),
    fn = as.numeric(cm_table[1, 2])
  )
  
  # Return in format matching CV output (single values replicated for mean/sd/etc)
  list(
    accuracy = accuracy,
    sensitivity = sensitivity,
    specificity = specificity,
    precision = precision,
    f1_score = f1,
    balanced_accuracy = balanced_accuracy,
    auroc = auroc,
    kappa = kappa,
    confusion_matrix = confusion,
    roc_curve = roc_curve,
    predictions = pred,
    probabilities = prob
  )
}

# =============================================================================
# PERMUTATION TESTING
# =============================================================================

run_permutation_test <- function(X_raw, X_scaled, y, config) {
  log_message(sprintf("Running permutation test (%d permutations)...", config$n_permutations))
  set.seed(config$seed)
  
  null_aurocs <- list(rf = numeric(), svm = numeric(), xgboost = numeric(),
                      knn = numeric(), mlp = numeric(), soft_vote = numeric())
  null_accuracy <- list(rf = numeric(), svm = numeric(), xgboost = numeric(),
                        knn = numeric(), mlp = numeric(), soft_vote = numeric())
  
  for (i in 1:config$n_permutations) {
    show_progress(i, config$n_permutations, "Permutation test")
    
    y_perm <- sample(y)
    
    # Train models on permuted data
    perm_results <- train_all_models(X_raw, X_scaled, y_perm, config)
    
    for (method in names(null_aurocs)) {
      if (!is.null(perm_results[[method]])) {
        null_aurocs[[method]] <- c(null_aurocs[[method]], perm_results[[method]]$auroc)
        null_accuracy[[method]] <- c(null_accuracy[[method]], perm_results[[method]]$accuracy)
      }
    }
  }
  
  return(list(
    null_aurocs = null_aurocs,
    null_accuracy = null_accuracy
  ))
}

# =============================================================================
# CALIBRATION CURVES
# =============================================================================

compute_calibration_curves <- function(results, y) {
  log_message("Computing calibration curves...")
  
  calibration_curves <- list()
  methods <- c("rf", "svm", "xgboost", "knn", "mlp", "soft_vote")
  
  for (method in methods) {
    if (is.null(results[[method]])) next
    
    prob <- results[[method]]$probabilities
    actual <- as.numeric(y) - 1
    
    # Create bins
    bins <- cut(prob, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)
    bin_stats <- tapply(1:length(prob), bins, function(idx) {
      list(
        bin = levels(bins)[unique(as.numeric(bins[idx]))],
        mean_pred = mean(prob[idx]),
        frac_pos = mean(actual[idx]),
        n = length(idx),
        bin_center = mean(prob[idx]),
        mean_pred_pct = mean(prob[idx]) * 100,
        frac_pos_pct = mean(actual[idx]) * 100
      )
    })
    
    calibration_curves[[method]] <- bin_stats[!sapply(bin_stats, is.null)]
  }
  
  return(calibration_curves)
}

# =============================================================================
# DIMENSIONALITY REDUCTION
# =============================================================================

compute_pca_embedding <- function(X, y, sample_ids) {
  if (ncol(X) < 2) return(NULL)
  
  # Verify dimensions match
  if (nrow(X) != length(y) || nrow(X) != length(sample_ids)) {
    log_message(sprintf("PCA dimension mismatch: X=%d rows, y=%d, sample_ids=%d", 
                        nrow(X), length(y), length(sample_ids)), "ERROR")
    log_message("Attempting to fix by using X dimensions...", "WARN")
    
    # Use only the samples that exist in X
    n_samples <- nrow(X)
    if (length(y) > n_samples) y <- y[1:n_samples]
    if (length(sample_ids) > n_samples) sample_ids <- sample_ids[1:n_samples]
  }
  
  pca_result <- prcomp(X, center = TRUE, scale. = TRUE)
  pca_df <- data.frame(
    x = pca_result$x[, 1],
    y = pca_result$x[, 2],
    sample_id = sample_ids,
    actual_class = as.character(y),
    stringsAsFactors = FALSE
  )
  
  list(points = lapply(1:nrow(pca_df), function(i) as.list(pca_df[i,])))
}

compute_tsne_embedding <- function(X, y, sample_ids) {
  if (!tsne_available || nrow(X) < 10) return(NULL)
  
  # Verify dimensions match
  if (nrow(X) != length(y) || nrow(X) != length(sample_ids)) {
    log_message(sprintf("t-SNE dimension mismatch: X=%d rows, y=%d, sample_ids=%d", 
                        nrow(X), length(y), length(sample_ids)), "WARN")
    n_samples <- nrow(X)
    if (length(y) > n_samples) y <- y[1:n_samples]
    if (length(sample_ids) > n_samples) sample_ids <- sample_ids[1:n_samples]
  }
  
  tryCatch({
    set.seed(42)
    tsne_result <- Rtsne(X, dims = 2, perplexity = min(30, floor((nrow(X)-1)/3)),
                         check_duplicates = FALSE, pca = TRUE, verbose = FALSE)
    tsne_df <- data.frame(
      x = tsne_result$Y[, 1],
      y = tsne_result$Y[, 2],
      sample_id = sample_ids,
      actual_class = as.character(y),
      stringsAsFactors = FALSE
    )
    list(points = lapply(1:nrow(tsne_df), function(i) as.list(tsne_df[i,])))
  }, error = function(e) NULL)
}

compute_umap_embedding <- function(X, y, sample_ids) {
  if (!umap_available || nrow(X) < 10) return(NULL)
  
  # Verify dimensions match
  if (nrow(X) != length(y) || nrow(X) != length(sample_ids)) {
    log_message(sprintf("UMAP dimension mismatch: X=%d rows, y=%d, sample_ids=%d", 
                        nrow(X), length(y), length(sample_ids)), "WARN")
    n_samples <- nrow(X)
    if (length(y) > n_samples) y <- y[1:n_samples]
    if (length(sample_ids) > n_samples) sample_ids <- sample_ids[1:n_samples]
  }
  
  tryCatch({
    umap_result <- umap::umap(X)
    umap_df <- data.frame(
      x = umap_result$layout[, 1],
      y = umap_result$layout[, 2],
      sample_id = sample_ids,
      actual_class = as.character(y),
      stringsAsFactors = FALSE
    )
    list(points = lapply(1:nrow(umap_df), function(i) as.list(umap_df[i,])))
  }, error = function(e) NULL)
}

# =============================================================================
# PROFILE RANKING WITH CLASS-SPECIFIC RISK SCORES
# =============================================================================

rank_profiles <- function(results, y, sample_ids, top_percent = 10, annotation = NULL, config = NULL) {
  log_message("Ranking sample profiles with class-specific risk scores...")
  
  ensemble_prob <- results$soft_vote$probabilities
  predicted_class <- as.character(results$soft_vote$predictions)
  actual_class <- as.character(y)
  
  # Get class levels
  class_levels <- levels(y)
  pos_class <- class_levels[2]  # Usually "1"
  neg_class <- class_levels[1]  # Usually "0"
  
  # Calculate risk scores for each class
  risk_score_positive <- ensemble_prob * 100
  risk_score_negative <- (1 - ensemble_prob) * 100
  
  # Calculate confidence and rank
  confidence <- abs(ensemble_prob - 0.5) * 2
  
  # Build data frame for consistent structure with CV script
  ranking <- data.frame(
    sample_index = 1:length(ensemble_prob),
    sample_id = if (length(sample_ids) >= length(ensemble_prob)) sample_ids else paste0("Sample_", 1:length(ensemble_prob)),
    actual_class = actual_class,
    ensemble_probability = ensemble_prob,
    predicted_class = predicted_class,
    confidence = confidence,
    correct = predicted_class == actual_class,
    risk_score_class_0 = round(risk_score_negative, 2),
    risk_score_class_1 = round(risk_score_positive, 2),
    stringsAsFactors = FALSE
  )
  
  # Add survival data if annotation is provided and config has time/event variables
  if (!is.null(annotation) && !is.null(config) && !is.null(config$time_variable) && !is.null(config$event_variable)) {
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
  
  # Order by confidence (descending) and assign ranks
  ranking <- ranking[order(-ranking$confidence), ]
  ranking$rank <- 1:nrow(ranking)
  
  # Mark top profiles
  top_n <- ceiling(nrow(ranking) * (top_percent / 100))
  ranking$top_profile <- ranking$rank <= top_n
  
  # Return consistent structure with CV script (top_profiles + all_rankings)
  list(
    top_profiles = ranking[ranking$top_profile, ],
    all_rankings = ranking
  )
}

# =============================================================================
# FEATURE BOXPLOT STATS
# =============================================================================

compute_feature_boxplot_stats <- function(unscaled_expr, y, features) {
  stats_list <- list()
  
  for (feat in features) {
    if (!feat %in% colnames(unscaled_expr)) next
    
    expr <- unscaled_expr[[feat]]
    stats_by_class <- lapply(levels(y), function(cls) {
      vals <- expr[y == cls]
      list(
        class = cls,
        min = min(vals, na.rm = TRUE),
        q1 = quantile(vals, 0.25, na.rm = TRUE),
        median = median(vals, na.rm = TRUE),
        q3 = quantile(vals, 0.75, na.rm = TRUE),
        max = max(vals, na.rm = TRUE),
        mean = mean(vals, na.rm = TRUE),
        n = length(vals)
      )
    })
    
    stats_list[[feat]] <- stats_by_class
  }
  
  return(stats_list)
}

# =============================================================================
# SURVIVAL ANALYSIS (Kaplan-Meier & Cox Proportional Hazards)
# =============================================================================

run_survival_analysis <- function(X, y, sample_ids, results, config, annot, annot_sample_col) {
  if (!survival_available) {
    log_message("Survival package not available, skipping survival analysis", "WARN")
    return(NULL)
  }
  
  if (is.null(config$time_variable) || is.null(config$event_variable)) {
    log_message("No time/event variables specified, skipping survival analysis", "INFO")
    return(NULL)
  }
  
  log_message("Running survival analysis...")
  
  # Check if columns exist
  if (!config$time_variable %in% colnames(annot)) {
    log_message(sprintf("Time variable '%s' not found in annotation. Available columns: %s", 
                        config$time_variable, paste(colnames(annot), collapse = ", ")), "WARN")
    return(NULL)
  }
  if (!config$event_variable %in% colnames(annot)) {
    log_message(sprintf("Event variable '%s' not found in annotation. Available columns: %s", 
                        config$event_variable, paste(colnames(annot), collapse = ", ")), "WARN")
    return(NULL)
  }
  
  log_message(sprintf("Time variable: '%s', Event variable: '%s'", 
                      config$time_variable, config$event_variable))
  
  # Extract survival data aligned with samples
  log_message(sprintf("Matching %d sample IDs to annotation (%d rows)", length(sample_ids), nrow(annot)))
  surv_annot <- annot[annot[[annot_sample_col]] %in% sample_ids, ]
  log_message(sprintf("Found %d matching samples in annotation", nrow(surv_annot)))
  
  if (nrow(surv_annot) == 0) {
    log_message("No matching samples found between expression data and annotation", "WARN")
    log_message(sprintf("Sample ID column in annotation: '%s'", annot_sample_col), "INFO")
    log_message(sprintf("First 5 sample IDs in expression: %s", paste(head(sample_ids, 5), collapse = ", ")), "INFO")
    log_message(sprintf("First 5 sample IDs in annotation: %s", paste(head(annot[[annot_sample_col]], 5), collapse = ", ")), "INFO")
    return(NULL)
  }
  
  surv_annot <- surv_annot[match(sample_ids, surv_annot[[annot_sample_col]]), ]
  
  # Convert to numeric with robust handling (suppress coercion warnings)
  time_vals <- suppressWarnings(as.numeric(as.character(surv_annot[[config$time_variable]])))
  event_vals <- suppressWarnings(as.numeric(as.character(surv_annot[[config$event_variable]])))
  
  # Log conversion results
  na_time_orig <- sum(is.na(surv_annot[[config$time_variable]]))
  na_event_orig <- sum(is.na(surv_annot[[config$event_variable]]))
  na_time_after <- sum(is.na(time_vals))
  na_event_after <- sum(is.na(event_vals))
  
  log_message(sprintf("Time values: %d total, %d NA in original, %d NA after conversion", 
                      length(time_vals), na_time_orig, na_time_after))
  log_message(sprintf("Event values: %d total, %d NA in original, %d NA after conversion", 
                      length(event_vals), na_event_orig, na_event_after))
  
  # Check for valid time values (must be positive)
  positive_time <- sum(!is.na(time_vals) & time_vals > 0)
  log_message(sprintf("Positive time values: %d", positive_time))
  
  # Remove samples with missing survival data
  valid_idx <- !is.na(time_vals) & !is.na(event_vals) & time_vals > 0
  log_message(sprintf("Valid samples for survival analysis: %d / %d", sum(valid_idx), length(valid_idx)))
  
  if (sum(valid_idx) < 10) {
    log_message("Insufficient valid survival data (< 10 samples)", "WARN")
    return(NULL)
  }
  
  log_message(sprintf("Survival analysis: %d samples with valid time/event data", sum(valid_idx)))
  
  time_valid <- time_vals[valid_idx]
  event_valid <- event_vals[valid_idx]
  X_valid <- X[valid_idx, , drop = FALSE]
  
  # Per-gene survival analysis (top features)
  per_gene_results <- list()
  top_features <- colnames(X_valid)[1:min(50, ncol(X_valid))]
  
  for (gene in top_features) {
    tryCatch({
      expr <- X_valid[, gene]
      median_expr <- median(expr, na.rm = TRUE)
      group <- ifelse(expr > median_expr, "High", "Low")
      group <- factor(group, levels = c("Low", "High"))
      
      surv_obj <- Surv(time_valid, event_valid)
      
      # Log-rank test
      logrank <- survdiff(surv_obj ~ group)
      logrank_p <- 1 - pchisq(logrank$chisq, df = 1)
      
      # Cox proportional hazards
      cox_fit <- coxph(surv_obj ~ expr)
      cox_summary <- summary(cox_fit)
      cox_hr <- cox_summary$conf.int[1]
      cox_hr_lower <- cox_summary$conf.int[3]
      cox_hr_upper <- cox_summary$conf.int[4]
      cox_p <- cox_summary$coefficients[5]
      
      # Kaplan-Meier for median survival
      km_fit <- survfit(surv_obj ~ group)
      km_summary <- summary(km_fit)
      
      # Get median survival times
      high_median <- tryCatch({
        sf <- survfit(surv_obj[group == "High"] ~ 1)
        if (!is.na(sf$surv[1])) quantile(sf, 0.5)$quantile else NA
      }, error = function(e) NA)
      
      low_median <- tryCatch({
        sf <- survfit(surv_obj[group == "Low"] ~ 1)
        if (!is.na(sf$surv[1])) quantile(sf, 0.5)$quantile else NA
      }, error = function(e) NA)
      
      per_gene_results[[gene]] <- list(
        gene = gene,
        logrank_p = logrank_p,
        cox_hr = cox_hr,
        cox_hr_lower = cox_hr_lower,
        cox_hr_upper = cox_hr_upper,
        cox_p = cox_p,
        high_median_surv = if (is.na(high_median)) NULL else high_median,
        low_median_surv = if (is.na(low_median)) NULL else low_median
      )
    }, error = function(e) {
      log_message(sprintf("Survival analysis failed for %s: %s", gene, e$message), "WARN")
    })
  }
  
  # Model-based risk score survival (using ensemble probabilities)
  model_risk_results <- list()
  
  # Log sample count alignment for debugging
  log_message(sprintf("Model risk survival: valid_idx has %d TRUE values out of %d samples", 
                      sum(valid_idx), length(valid_idx)))
  
  for (model_name in c("rf", "svm", "xgboost", "soft_vote")) {
    if (is.null(results[[model_name]])) next
    
    tryCatch({
      probs <- results[[model_name]]$probabilities
      
      # Verify length matches sample_ids (probabilities should be in sample_ids order)
      if (length(probs) != length(sample_ids)) {
        log_message(sprintf("Probability length (%d) != sample_ids length (%d) for %s", 
                            length(probs), length(sample_ids), model_name), "WARN")
        next
      }
      
      # Apply same valid_idx to get probabilities for samples with valid survival data
      probs_valid <- probs[valid_idx]
      
      log_message(sprintf("Model %s: %d valid samples, prob range: %.3f - %.3f", 
                          model_name, length(probs_valid), min(probs_valid, na.rm=TRUE), max(probs_valid, na.rm=TRUE)))
      
      median_prob <- median(probs_valid, na.rm = TRUE)
      risk_group <- ifelse(probs_valid > median_prob, "High", "Low")
      
      # Log group sizes for debugging significance issues
      n_high <- sum(risk_group == "High", na.rm = TRUE)
      n_low <- sum(risk_group == "Low", na.rm = TRUE)
      log_message(sprintf("Model %s risk groups: High=%d, Low=%d (median cutoff=%.3f)", 
                          model_name, n_high, n_low, median_prob))
      risk_group <- factor(risk_group, levels = c("Low", "High"))
      
      # Log event distribution across risk groups for debugging
      events_high <- sum(event_valid[risk_group == "High"], na.rm = TRUE)
      events_low <- sum(event_valid[risk_group == "Low"], na.rm = TRUE)
      log_message(sprintf("Model %s events: High=%d/%d, Low=%d/%d", 
                          model_name, events_high, n_high, events_low, n_low))
      
      surv_obj <- Surv(time_valid, event_valid)
      
      # Log-rank test
      logrank <- survdiff(surv_obj ~ risk_group)
      logrank_p <- 1 - pchisq(logrank$chisq, df = 1)
      log_message(sprintf("Model %s: Log-rank chi-sq=%.3f, p=%.4e", model_name, logrank$chisq, logrank_p))
      
      # Cox model
      cox_fit <- coxph(surv_obj ~ probs_valid)
      cox_summary <- summary(cox_fit)
      cox_hr <- cox_summary$conf.int[1]
      cox_hr_lower <- cox_summary$conf.int[3]
      cox_hr_upper <- cox_summary$conf.int[4]
      cox_p <- cox_summary$coefficients[5]
      
      # KM curves for high/low risk
      km_fit <- survfit(surv_obj ~ risk_group)
      
      # Extract KM curve data
      extract_km_curve <- function(km_fit, strata_name) {
        if (is.null(km_fit$strata)) {
          # Single stratum
          data.frame(
            time = km_fit$time,
            surv = km_fit$surv,
            lower = km_fit$lower,
            upper = km_fit$upper,
            n_risk = km_fit$n.risk,
            n_event = km_fit$n.event,
            n_censor = km_fit$n.censor,
            stringsAsFactors = FALSE
          )
        } else {
          # Multiple strata
          strata_idx <- which(names(km_fit$strata) == strata_name)
          if (length(strata_idx) == 0) return(data.frame())
          
          start_idx <- if (strata_idx == 1) 1 else sum(km_fit$strata[1:(strata_idx-1)]) + 1
          end_idx <- sum(km_fit$strata[1:strata_idx])
          
          data.frame(
            time = km_fit$time[start_idx:end_idx],
            surv = km_fit$surv[start_idx:end_idx],
            lower = km_fit$lower[start_idx:end_idx],
            upper = km_fit$upper[start_idx:end_idx],
            n_risk = km_fit$n.risk[start_idx:end_idx],
            n_event = km_fit$n.event[start_idx:end_idx],
            n_censor = km_fit$n.censor[start_idx:end_idx],
            stringsAsFactors = FALSE
          )
        }
      }
      
      km_high <- extract_km_curve(km_fit, "risk_group=High")
      km_low <- extract_km_curve(km_fit, "risk_group=Low")
      
      model_risk_results[[model_name]] <- list(
        model = model_name,
        stats = list(
          logrank_p = logrank_p,
          cox_hr = cox_hr,
          cox_hr_lower = cox_hr_lower,
          cox_hr_upper = cox_hr_upper,
          cox_p = cox_p
        ),
        km_curve_high = lapply(1:nrow(km_high), function(i) as.list(km_high[i,])),
        km_curve_low = lapply(1:nrow(km_low), function(i) as.list(km_low[i,]))
      )
    }, error = function(e) {
      log_message(sprintf("Model risk survival failed for %s: %s", model_name, e$message), "WARN")
    })
  }
  
  log_message(sprintf("Survival analysis complete: %d genes, %d models", 
                      length(per_gene_results), length(model_risk_results)))
  
  return(list(
    time_variable = config$time_variable,
    event_variable = config$event_variable,
    per_gene = per_gene_results,
    model_risk_scores = model_risk_results
  ))
}

# =============================================================================
# BUILD JSON OUTPUT
# =============================================================================

build_json_output <- function(data, results, feature_selection, permutation_results, 
                              calibration_curves, clustering, profile_ranking, survival_analysis, config) {
  
  # Format model performance (matching CV output format)
  model_performance <- list()
  methods <- c("rf", "svm", "xgboost", "knn", "mlp", "hard_vote", "soft_vote")
  
  for (method in methods) {
    if (is.null(results[[method]])) next
    
    m <- results[[method]]
    # Since we have single training values, set mean=median=min=max=value, sd=0
    model_performance[[method]] <- list(
      accuracy = list(mean = m$accuracy, sd = 0, median = m$accuracy, 
                     q25 = m$accuracy, q75 = m$accuracy,
                     min = m$accuracy, max = m$accuracy),
      sensitivity = list(mean = m$sensitivity, sd = 0, median = m$sensitivity,
                        q25 = m$sensitivity, q75 = m$sensitivity,
                        min = m$sensitivity, max = m$sensitivity),
      specificity = list(mean = m$specificity, sd = 0, median = m$specificity,
                        q25 = m$specificity, q75 = m$specificity,
                        min = m$specificity, max = m$specificity),
      precision = list(mean = m$precision, sd = 0, median = m$precision,
                      q25 = m$precision, q75 = m$precision,
                      min = m$precision, max = m$precision),
      f1_score = list(mean = m$f1_score, sd = 0, median = m$f1_score,
                     q25 = m$f1_score, q75 = m$f1_score,
                     min = m$f1_score, max = m$f1_score),
      balanced_accuracy = list(mean = m$balanced_accuracy, sd = 0, median = m$balanced_accuracy,
                              q25 = m$balanced_accuracy, q75 = m$balanced_accuracy,
                              min = m$balanced_accuracy, max = m$balanced_accuracy),
      auroc = list(mean = m$auroc, sd = 0, median = m$auroc,
                  q25 = m$auroc, q75 = m$auroc,
                  min = m$auroc, max = m$auroc),
      kappa = list(mean = m$kappa, sd = 0, median = m$kappa,
                  q25 = m$kappa, q75 = m$kappa,
                  min = m$kappa, max = m$kappa),
      confusion_matrix = m$confusion_matrix,
      roc_curve = m$roc_curve
    )
  }
  
  # Permutation testing summary
  perm_testing <- list()
  for (method in c("rf", "svm", "xgboost", "knn", "mlp", "soft_vote")) {
    if (length(permutation_results$null_aurocs[[method]]) > 0) {
      null_auroc <- permutation_results$null_aurocs[[method]]
      actual_auroc <- results[[method]]$auroc
      p_val <- mean(null_auroc >= actual_auroc)
      
      perm_testing[[paste0(method, "_auroc")]] <- list(
        permuted_mean = mean(null_auroc),
        permuted_sd = sd(null_auroc),
        original = actual_auroc,
        p_value = p_val
      )
    }
  }
  
  # Feature boxplot stats
  top_features <- head(feature_selection$feature_importance$feature, 20)
  feature_boxplot_stats <- compute_feature_boxplot_stats(data$unscaled_expr, data$y, top_features)
  
  # Remove redundant info from config
  config_clean <- config[!vapply(config, is.list, logical(1))]
  
  # Build final output
  output <- list(
    metadata = list(
      generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
      config = config_clean,
      r_version = R.version.string
    ),
    preprocessing = data$preprocessing_stats,
    model_performance = model_performance,
    feature_importance = feature_selection$feature_importance,
    feature_importance_stability = feature_selection$feature_importance_stability,
    feature_boxplot_stats = feature_boxplot_stats,
    calibration_curves = calibration_curves,
    clustering = clustering,
    permutation_testing = perm_testing,
    permutation_distributions = list(
      rf = list(auroc = permutation_results$null_aurocs$rf,
                accuracy = permutation_results$null_accuracy$rf),
      svm = list(auroc = permutation_results$null_aurocs$svm,
                 accuracy = permutation_results$null_accuracy$svm),
      xgboost = list(auroc = permutation_results$null_aurocs$xgboost,
                     accuracy = permutation_results$null_accuracy$xgboost),
      knn = list(auroc = permutation_results$null_aurocs$knn,
                 accuracy = permutation_results$null_accuracy$knn),
      mlp = list(auroc = permutation_results$null_aurocs$mlp,
                 accuracy = permutation_results$null_accuracy$mlp),
      soft_vote = list(auroc = permutation_results$null_aurocs$soft_vote,
                       accuracy = permutation_results$null_accuracy$soft_vote)
    ),
    actual_distributions = list(),  # Empty as in original
    profile_ranking = profile_ranking,
    selected_features = feature_selection$selected_features,
    survival_analysis = survival_analysis
  )
  
  return(output)
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

run_pipeline <- function(config) {
  start_time <- Sys.time()
  
  config <- get_effective_config(config)
  set.seed(config$seed)
  
  if (!dir.exists(config$output_dir)) {
    dir.create(config$output_dir, recursive = TRUE)
  }
  
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("FULL DATASET TRAINING MODE")
  log_message("WARNING: Training metrics only - validate externally!")
  log_message(paste(rep("=", 60), collapse = ""))
  
  # Load data
  data <- load_and_preprocess_data(config)
  
  # Load annotation for survival analysis (need full annotation)
  annot <- read.delim(config$annotation_file, stringsAsFactors = FALSE)
  annot_sample_col <- colnames(annot)[1]
  
  # Feature selection
  feature_selection <- perform_feature_selection(data$X_scaled, data$y, config)
  X_selected_scaled <- data$X_scaled[, feature_selection$selected_features, drop = FALSE]
  X_selected_raw    <- data$X_raw[, feature_selection$selected_features, drop = FALSE]
  
  # Train models
  results <- train_all_models(
    X_raw    = X_selected_raw,
    X_scaled = X_selected_scaled,
    y        = data$y,
    config   = config
  )
  
  # Permutation testing
  permutation_results <- run_permutation_test(X_selected_raw, X_selected_scaled, data$y, config)
  
  # Calibration curves
  calibration_curves <- compute_calibration_curves(results, data$y)
  
  # Dimensionality reduction
  X_dr <- if (isTRUE(config$model_scaling$dr)) X_selected_scaled else X_selected_raw
  
  clustering <- list(
    pca  = compute_pca_embedding(X_dr, data$y, data$sample_ids),
    tsne = compute_tsne_embedding(X_dr, data$y, data$sample_ids),
    umap = compute_umap_embedding(X_dr, data$y, data$sample_ids)
  )
  
  # Profile ranking (consistent structure with CV script)
  profile_ranking <- rank_profiles(results, data$y, data$sample_ids, config$top_percent, annotation = annot, config = config)
  
  # Survival analysis (if time/event variables provided)
  survival_analysis <- run_survival_analysis(
    X = X_selected_raw,
    y = data$y,
    sample_ids = data$sample_ids,
    results = results,
    config = config,
    annot = annot,
    annot_sample_col = annot_sample_col
  )
  
  # Build JSON output
  output <- build_json_output(data, results, feature_selection, permutation_results,
                              calibration_curves, clustering, profile_ranking, survival_analysis, config)
  
  # Export
  output_path <- file.path(config$output_dir, config$output_json)
  log_message(sprintf("Exporting to: %s", output_path))
  json_output <- toJSON(output, auto_unbox = TRUE, pretty = TRUE, digits = 6, na = "null")
  writeLines(json_output, output_path)
  
  # Save models with proper XGBoost serialization
  models <- lapply(results, function(r) r$model)
  
  # XGBoost models need special serialization - convert to raw bytes
  if (!is.null(models$xgboost) && inherits(models$xgboost, "xgb.Booster")) {
    log_message("Serializing XGBoost model using xgb.save.raw()...")
    models$xgboost_raw <- xgb.save.raw(models$xgboost)
    models$xgboost <- NULL  # Remove the pointer-based object
    models$xgboost_serialized <- TRUE
  }
  
  models_path <- file.path(config$output_dir, "trained_models.rds")
  saveRDS(models, models_path)
  log_message(sprintf("Models saved to: %s", models_path))
  log_message("NOTE: To load XGBoost model, use: models$xgboost <- xgb.load.raw(models$xgboost_raw)")
  
  end_time <- Sys.time()
  log_message(sprintf("Completed in %.2f minutes",
                      as.numeric(difftime(end_time, start_time, units = "mins"))))
  
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("IMPORTANT: These are TRAINING metrics - validate externally!")
  log_message(paste(rep("=", 60), collapse = ""))
  
  return(invisible(output))
}

# =============================================================================
# RUN
# =============================================================================

if (!interactive()) {
  run_pipeline(config)
}
