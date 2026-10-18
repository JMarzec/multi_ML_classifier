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
  
  make_option(c("--train_ratio"),
              type = "double",
              default = NULL,
              help = "Training set ratio (0.5-0.9). Overrides n_folds for split calculation. E.g., 0.7 = 70%% training, 30%% testing [default: calculated from n_folds]"),
  
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
  model_scaling = MODEL_SCALING,
  
  # Model settings
  seed = 42,
  n_folds = 5,
  n_repeats = 10,
  top_percent = 10,
  train_ratio = NULL,  # Training set ratio (0.5-0.9). If set, overrides n_folds for split calculation
  
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
if (!is.null(opt$train_ratio)) {
  # Validate train_ratio is between 0.5 and 0.9
  if (opt$train_ratio < 0.5 || opt$train_ratio > 0.9) {
    stop("train_ratio must be between 0.5 and 0.9")
  }
  config$train_ratio <- opt$train_ratio
}
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
binary_levels <- function(y) {
  lev <- if (is.factor(y)) levels(y) else sort(unique(as.character(y)))
  if (length(lev) != 2L || anyNA(y)) stop("Exactly two nonmissing classes required")
  lev
}

fit_preprocessor <- function(X, y, config) {
  X <- as.data.frame(X, check.names = FALSE)
  med <- vapply(X, function(z) {
    z <- z[is.finite(z)]
    if (length(z)) median(z) else NA_real_
  }, numeric(1))
  X <- X[, is.finite(med), drop = FALSE]
  med <- med[colnames(X)]
  for (nm in names(X)) X[[nm]][!is.finite(X[[nm]])] <- med[[nm]]
  vv <- vapply(X, var, numeric(1))
  keep <- names(vv)[is.finite(vv) & vv > 0]
  if (!length(keep)) stop("No variable training features")
  cutoff <- quantile(vv[keep], VAR_QUANTILE, names = FALSE)
  keep <- keep[vv[keep] >= cutoff] # retain ties, including one-feature cases
  X <- X[, keep, drop = FALSE]
  nzv <- caret::nearZeroVar(X)
  if (length(nzv)) X <- X[, -nzv, drop = FALSE]
  if (!ncol(X)) stop("No training features after NZV filtering")
  # Existing selection routines see only this outer training set.
  # With <3 samples per class their fixed 3-fold inner CV is unsafe.
  method <- config$feature_selection_method
  if (method != "none" && min(table(y)) < 3L)
    stop("Supervised selection requires >=3 training samples per class; use none")
  sel <- perform_feature_selection(X, y, method, config$max_features, config$seed)
  if (!length(sel)) stop("Feature selection returned no features")
  X <- X[, sel, drop = FALSE]
  list(features = sel, median = med[sel],
       center = vapply(X, mean, numeric(1)),
       scale = vapply(X, sd, numeric(1)))
}

apply_preprocessor <- function(X, pp, scaled = FALSE) {
  X <- as.data.frame(X[, pp$features, drop = FALSE], check.names = FALSE)
  for (nm in names(X)) X[[nm]][!is.finite(X[[nm]])] <- pp$median[[nm]]
  if (scaled) {
    X <- as.data.frame(sweep(sweep(as.matrix(X), 2, pp$center, "-"),
                             2, pp$scale, "/"), check.names = FALSE)
  }
  X
}

make_splits <- function(y, config) {
  binary_levels(y)
  if (config$n_repeats < 1L) stop("n_repeats must be positive")
  if (min(table(y)) < 2L) stop("Each class needs >=2 samples for resampling")
  out <- list()
  for (r in seq_len(config$n_repeats)) {
    set.seed(config$seed + r)
    if (is.null(config$train_ratio)) {
      k <- config$n_folds
      if (k < 2L || k > min(table(y)))
        stop("n_folds must be between 2 and the minority class count")
      ff <- caret::createFolds(y, k = k, returnTrain = TRUE)
    } else {
      if (!is.finite(config$train_ratio) || config$train_ratio < .5 ||
          config$train_ratio > .9) stop("train_ratio must be in [0.5,0.9]")
      ff <- list(as.vector(caret::createDataPartition(
        y, p = config$train_ratio, list = FALSE)))
    }
    for (tr in ff) {
      te <- setdiff(seq_along(y), tr)
      if (!length(te) || length(unique(y[tr])) != 2L)
        stop("Invalid train/test split")
      out[[length(out) + 1L]] <- list(train = tr, test = te, repeat_id = r)
    }
  }
  out
}

fit_one <- function(method, X, y, config) {
  lev <- binary_levels(y)
  yy <- factor(ifelse(as.character(y) == lev[2], "1", "0"), levels = c("0", "1"))
  if (method == "knn") config$knn_k <- min(config$knn_k, nrow(X))
  obj <- get(paste0("train_", method))(X, yy, config)
  obj$class_levels <- lev
  obj
}

predict_one <- function(method, obj, X) {
  z <- get(paste0("predict_", method))(obj, X)
  z$predictions <- factor(obj$class_levels[match(as.character(z$predictions), c("0", "1"))],
                          levels = obj$class_levels)
  z
}

permutation_p <- function(null, observed) {
  if (length(observed) != 1L || !is.finite(observed) || !length(null) ||
      any(!is.finite(null))) return(NA_real_)
  (1 + sum(null >= observed))/(1 + length(null))
}

calculate_metrics <- function(actual, predicted, probabilities = NULL) {
  lev <- binary_levels(actual) # actual must retain full factor levels in test subsets
  a <- as.character(actual); p <- as.character(predicted)
  if (length(a) != length(p) || !length(a) || anyNA(a) || anyNA(p) ||
      any(!p %in% lev)) stop("Invalid metric inputs")
  pos <- lev[2]; neg <- lev[1]
  tp <- sum(a == pos & p == pos); tn <- sum(a == neg & p == neg)
  fp <- sum(a == neg & p == pos); fn <- sum(a == pos & p == neg)
  div <- function(x, d) if (d == 0) NA_real_ else x / d
  sens <- div(tp, tp + fn); spec <- div(tn, tn + fp)
  acc <- mean(a == p)
  expected <- ((tp + fn)*(tp + fp) + (tn + fp)*(tn + fn))/length(a)^2
  ans <- list(accuracy = acc, sensitivity = sens, specificity = spec,
    precision = div(tp, tp + fp), f1_score = div(2*tp, 2*tp + fp + fn),
    balanced_accuracy = (sens + spec)/2, kappa = div(acc - expected, 1 - expected),
    confusion_matrix = list(tp = tp, tn = tn, fp = fp, fn = fn),
    auroc = NA_real_, roc_curve = NULL)
  if (!is.null(probabilities) && length(probabilities) == length(a) &&
      all(is.finite(probabilities)) && all(probabilities >= 0 & probabilities <= 1) &&
      length(unique(a)) == 2L) {
    rr <- pROC::roc(factor(a, levels = lev), probabilities,
                     levels = lev, direction = "<", quiet = TRUE)
    ans$auroc <- as.numeric(pROC::auc(rr))
    ans$roc_curve <- data.frame(fpr = 1 - rr$specificities, tpr = rr$sensitivities)
  }
  ans
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
  annotation[[annot_sample_col]] <- as.character(annotation[[annot_sample_col]])
  annot_samples <- annotation[[annot_sample_col]]
  if (anyNA(annot_samples) || any(!nzchar(annot_samples)) || anyDuplicated(annot_samples) || anyDuplicated(sample_ids)) stop("Sample IDs must be nonmissing and unique")
  
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
  
  expr_matrix <- expr_matrix[common_samples, , drop = FALSE]
  annotation <- annotation[match(common_samples, annot_samples), , drop = FALSE]
  sample_ids <- rownames(expr_matrix)
  stopifnot(identical(sample_ids, annotation[[annot_sample_col]]))
  X <- expr_matrix
  y <- factor(annotation[[config$target_variable]])
  if (anyNA(y) || nlevels(y) != 2L) stop("Exactly two nonmissing target classes are required")

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
  # Calculate train/test split info based on config
  n_folds <- if (!is.null(config$n_folds)) config$n_folds else 5
  n_repeats <- if (!is.null(config$n_repeats)) config$n_repeats else 3
  
  # Determine effective training ratio: use train_ratio if specified, otherwise calculate from n_folds
  effective_train_ratio <- if (!is.null(config$train_ratio)) {
    config$train_ratio
  } else {
    (n_folds - 1) / n_folds  # e.g., 5-fold = 80% training
  }
  
  train_per_fold <- round(length(sample_ids) * effective_train_ratio)
  test_per_fold <- length(sample_ids) - train_per_fold
  
  # Calculate approximate class distribution per fold (stratified)
  class_table <- table(y)
  train_class_dist <- as.list(round(class_table * effective_train_ratio))
  test_class_dist <- as.list(round(class_table * (1 - effective_train_ratio)))
  
  preprocessing_stats <- list(
    original_samples = length(sample_ids),
    original_features = ncol(X),
    missing_values = sum(is.na(X)),
    missing_pct = round(sum(is.na(X)) / (nrow(X) * ncol(X)) * 100, 2),
    class_distribution = as.list(class_table),
    constant_features_removed = 0,
    cv_folds = n_folds,
    cv_repeats = n_repeats,
    train_ratio = effective_train_ratio,
    train_samples_per_fold = train_per_fold,
    test_samples_per_fold = test_per_fold,
    train_class_distribution = train_class_dist,
    test_class_distribution = test_class_dist,
    full_training_mode = FALSE
  )
  
  X <- as.data.frame(lapply(X, function(z) as.numeric(as.character(z))), check.names = FALSE)
  rownames(X) <- sample_ids
  X[] <- lapply(X, function(z) { z[!is.finite(z)] <- NA_real_; z })
  X_raw <- X
  X_scaled <- X # Compatibility only: transformations are fitted within each split.

  return(list(
    X_raw    = X_raw,
    X_scaled = X_scaled,
    y = y,
    sample_ids = sample_ids,
    feature_names = colnames(X_raw),
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
  
  model <- svm(x = as.matrix(X_train), y = y_train, scale = FALSE, kernel = config$svm_kernel,
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
  lev <- levels(predictions_list[[1]])
  mm <- do.call(cbind, lapply(predictions_list, as.character))
  factor(ifelse(rowMeans(mm == lev[2]) > .5, lev[2], lev[1]), levels = lev)
}

soft_voting <- function(probabilities_list, weights = NULL, class_levels = c("0", "1")) {
  n <- length(probabilities_list)
  if (!n) stop("No probabilities")
  if (is.null(weights)) weights <- rep(1/n, n)
  weights <- weights/sum(weights)
  prob <- Reduce(`+`, Map(`*`, probabilities_list, weights))
  list(predictions = factor(ifelse(prob > .5, class_levels[2], class_levels[1]),
                            levels = class_levels), probabilities = prob)
}

is_valid_prediction <- function(pred, y_test) {
  !is.null(pred) && length(pred) == length(y_test) &&
    length(pred) > 0L && !anyNA(pred) &&
    all(as.character(pred) %in% binary_levels(y_test))
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
  if (!is.null(selected_features))
    stop("Do not pass globally selected features into CV")
  lev <- binary_levels(y)
  methods <- c("rf", "svm", "xgboost", "knn", "mlp")
  all_methods <- c(methods, "hard_vote", "soft_vote")
  splits <- make_splits(y, config)
  results <- setNames(lapply(all_methods, function(m) vector("list", length(splits))), all_methods)
  counts <- setNames(as.list(rep(0L, length(all_methods))), all_methods)
  cvp <- setNames(vector("list", length(methods) + 1L), c(methods, "soft_vote"))
  imps <- list()
  for (i in seq_along(splits)) {
    tr <- splits[[i]]$train; te <- splits[[i]]$test
    set.seed(config$seed + i)
    pp <- fit_preprocessor(X_raw[tr, , drop = FALSE], y[tr], config)
    preds <- list(); probs <- list()
    for (m in methods) {
      scaled <- isTRUE(config$scale_data) || isTRUE(config$model_scaling[[m]])
      train_x <- apply_preprocessor(X_raw[tr, , drop = FALSE], pp, scaled)
      test_x <- apply_preprocessor(X_raw[te, , drop = FALSE], pp, scaled)
      z <- tryCatch({
        obj <- fit_one(m, train_x, y[tr], config)
        pred <- predict_one(m, obj, test_x)
        list(obj = obj, pred = pred)
      }, error = function(e) {
        log_message(sprintf("Fold %d %s failed: %s", i, m, conditionMessage(e)), "WARN")
        NULL
      })
      if (is.null(z)) next
      pr <- z$pred
      if (!is_valid_prediction(pr$predictions, y[te])) next
      preds[[m]] <- pr$predictions
      if (length(pr$probabilities) == length(te) &&
          all(is.finite(pr$probabilities)) &&
          all(pr$probabilities >= 0 & pr$probabilities <= 1))
        probs[[m]] <- pr$probabilities
      results[[m]][[i]] <- calculate_metrics(y[te], preds[[m]], probs[[m]])
      counts[[m]] <- counts[[m]] + 1L
      if (!is.null(probs[[m]])) cvp[[m]] <- rbind(cvp[[m]], data.frame(
        actual = as.integer(y[te] == lev[2]), prob = probs[[m]],
        sample_index = te, split = i, repeat_id = splits[[i]]$repeat_id))
      if (m == "rf" && !is.null(z$obj$importance))
        imps[[length(imps) + 1L]] <- z$obj$importance
    }
    if (length(preds) >= 2L) {
      mat <- do.call(cbind, lapply(preds, as.character))
      # Deterministic tie break to the negative class; keep the same in final fits.
      hp <- factor(ifelse(rowMeans(mat == lev[2]) > .5, lev[2], lev[1]), levels = lev)
      results$hard_vote[[i]] <- calculate_metrics(y[te], hp)
      counts$hard_vote <- counts$hard_vote + 1L
    }
    if (length(probs) >= 2L) {
      sp <- rowMeans(do.call(cbind, probs))
      pred <- factor(ifelse(sp > .5, lev[2], lev[1]), levels = lev)
      results$soft_vote[[i]] <- calculate_metrics(y[te], pred, sp)
      counts$soft_vote <- counts$soft_vote + 1L
      cvp$soft_vote <- rbind(cvp$soft_vote, data.frame(
        actual = as.integer(y[te] == lev[2]), prob = sp,
        sample_index = te, split = i, repeat_id = splits[[i]]$repeat_id))
    }
  }
  # Align importances by feature NAME; selections differ across folds.
  fi <- data.frame(feature = character(), importance = numeric())
  if (length(imps)) {
    feats <- unique(unlist(lapply(imps, names)))
    mm <- vapply(imps, function(v) {
      out <- setNames(rep(0, length(feats)), feats)
      out[names(v)] <- v
      out
    }, numeric(length(feats)))
    fi <- data.frame(feature = feats, importance = rowMeans(mm))
    fi <- fi[order(-fi$importance), , drop = FALSE]
  }
  list(results = results, fold_metrics = results, feature_importance = fi,
       cv_predictions = cvp, fold_importance = imps, valid_folds = counts,
       attempted_folds = length(splits),
       ensemble_valid = counts$hard_vote > 0L || counts$soft_vote > 0L)
}

# =============================================================================
# PERMUTATION TESTING
# =============================================================================

run_permutation_test <- function(X, y, config, selected_features = NULL) {
  if (!is.null(selected_features)) stop("Permutation must repeat feature selection")
  B <- config$n_permutations
  if (length(B) != 1L || !is.finite(B) || B < 0 || B != as.integer(B))
    stop("n_permutations must be a nonnegative integer")
  methods <- c("rf", "svm", "xgboost", "knn", "mlp", "soft_vote")
  out <- list(rf_oob_error = rep(NA_real_, B), rf_auroc = rep(NA_real_, B),
    per_model = setNames(lapply(methods, function(m)
      list(auroc = rep(NA_real_, B), accuracy = rep(NA_real_, B))), methods))
  # Generate all permutations before CV resets the RNG.
  set.seed(config$seed + 100000L)
  ys <- lapply(seq_len(B), function(i) y[sample.int(length(y))])
  for (b in seq_len(B)) {
    z <- run_cv_all_methods(X, X, ys[[b]], config)
    s <- aggregate_results(z$results, z$attempted_folds, z$valid_folds)
    for (m in methods) for (metric in c("auroc", "accuracy")) {
      val <- s[[m]][[metric]]$mean
      if (length(val) == 1L && is.finite(val)) out$per_model[[m]][[metric]][b] <- val
    }
    out$rf_auroc[b] <- out$per_model$rf$auroc[b]
  }
  out
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

# A more robust ranking builder that does NOT re-predict per-model probabilities.
# This avoids failures when some models were trained on scaled data but predictions
# are attempted on unscaled matrices (common source of empty Rankings/Risk Scores
# in the exported JSON).
build_profile_ranking <- function(
  y,
  probabilities,
  predicted_class,
  config,
  sample_ids = NULL,
  annotation = NULL,
  model_name = "soft_vote"
) {
  log_message("Building profile ranking (Rankings/Risk Scores export)...")

  if (is.null(probabilities) || length(probabilities) == 0) {
    log_message("No probabilities available to build profile ranking", "WARN")
    return(NULL)
  }

  n <- length(probabilities)
  if (length(y) != n) {
    log_message(sprintf("Ranking length mismatch: y=%d, probs=%d", length(y), n), "WARN")
    n <- min(length(y), n)
    probabilities <- probabilities[1:n]
    y <- y[1:n]
    if (!is.null(predicted_class)) predicted_class <- predicted_class[1:n]
    if (!is.null(sample_ids)) sample_ids <- sample_ids[1:n]
  }

  class_levels <- levels(y)
  pos_class <- class_levels[2]
  neg_class <- class_levels[1]

  pred <- if (!is.null(predicted_class)) {
    as.character(predicted_class)
  } else {
    ifelse(probabilities > 0.5, pos_class, neg_class)
  }

  # For consistency with full-training JSON: class_0=(1-prob), class_1=prob
  risk_score_class_1 <- as.numeric(probabilities) * 100
  risk_score_class_0 <- (1 - as.numeric(probabilities)) * 100
  confidence <- abs(as.numeric(probabilities) - 0.5) * 2

  ranking <- data.frame(
    sample_index = 1:n,
    sample_id = if (!is.null(sample_ids) && length(sample_ids) >= n) sample_ids[1:n] else paste0("Sample_", 1:n),
    actual_class = as.character(y),
    ensemble_probability = as.numeric(probabilities),
    predicted_class = pred,
    confidence = confidence,
    correct = pred == as.character(y),
    risk_score_class_0 = round(risk_score_class_0, 2),
    risk_score_class_1 = round(risk_score_class_1, 2),
    stringsAsFactors = FALSE
  )

  # Add survival columns (optional)
  if (!is.null(annotation) && !is.null(config$time_variable) && !is.null(config$event_variable)) {
    time_col <- config$time_variable
    event_col <- config$event_variable

    if (time_col %in% colnames(annotation) && event_col %in% colnames(annotation)) {
      annot_sample_col <- colnames(annotation)[1]
      surv_lookup <- data.frame(
        sample_id = annotation[[annot_sample_col]],
        surv_time = suppressWarnings(as.numeric(as.character(annotation[[time_col]]))),
        surv_event = suppressWarnings(as.numeric(as.character(annotation[[event_col]]))),
        stringsAsFactors = FALSE
      )

      ranking$surv_time <- surv_lookup$surv_time[match(ranking$sample_id, surv_lookup$sample_id)]
      ranking$surv_event <- surv_lookup$surv_event[match(ranking$sample_id, surv_lookup$sample_id)]
    }
  }

  ranking <- ranking[order(-ranking$confidence), ]
  ranking$rank <- 1:nrow(ranking)

  top_n <- ceiling(nrow(ranking) * (config$top_percent / 100))
  ranking$top_profile <- ranking$rank <= top_n

  attr(ranking, "model_name") <- model_name
  ranking
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
perform_model_risk_survival <- function(rankings, annotation, time_col, event_col, sample_ids, model_name = "soft_vote") {
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
    cutoff <- median(risk_vals, na.rm = TRUE)
    high_risk <- risk_vals >= cutoff
    
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
      model = as.character(model_name),
      cutoff = as.numeric(cutoff),
      high_group_rule = "positive_class_probability >= cutoff",
      cox_predictor = "continuous_positive_class_probability",
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
  model_name <- attr(ranking, "model_name")
  if (is.null(model_name) || model_name == "") model_name <- "soft_vote"

  model_risk <- perform_model_risk_survival(
    rankings = ranking,
    annotation = annotation,
    time_col = config$time_variable,
    event_col = config$event_variable,
    sample_ids = sample_ids,
    model_name = model_name
  )
  
  if (is.null(per_gene) && is.null(model_risk)) {
    log_message("No survival analysis results generated", "WARN")
    return(NULL)
  }
  
  model_risk_scores <- if (!is.null(model_risk) && !is.null(model_risk$model)) {
    setNames(list(model_risk), model_risk$model)
  } else {
    NULL
  }

  result <- list(
    time_variable = config$time_variable,
    event_variable = config$event_variable,
    per_gene = if (!is.null(per_gene)) per_gene$per_gene else NULL,
    model_risk_scores = model_risk_scores
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
          p_value = NA_real_
        ),
        rf_auroc = list(
          permuted_mean = mean(results$permutation$rf_auroc, na.rm = TRUE),
          permuted_sd = sd(results$permutation$rf_auroc, na.rm = TRUE),
          original = results$original_metrics$rf_auroc,
          p_value = permutation_p(results$permutation$rf_auroc, results$original_metrics$rf_auroc)
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
  
  json_output <- toJSON(output, auto_unbox = TRUE, pretty = TRUE, digits = 6, na = "null")
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
  
  # Cross-validation
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("CROSS-VALIDATION")
  
  cv_results <- run_cv_all_methods(data$X_raw, data$X_raw, data$y, config)

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
  
  final_pp <- fit_preprocessor(data$X_raw, data$y, config)
  selected_features <- final_pp$features
  final_X <- setNames(lapply(c("rf", "svm", "xgboost", "knn", "mlp"), function(m)
    apply_preprocessor(data$X_raw, final_pp, isTRUE(config$scale_data) || isTRUE(config$model_scaling[[m]]))),
    c("rf", "svm", "xgboost", "knn", "mlp"))
  final_models <- setNames(lapply(names(final_X), function(m)
    tryCatch(fit_one(m, final_X[[m]], data$y, config), error = function(e) {
      log_message(conditionMessage(e), "WARN"); NULL
    })), names(final_X))
  X_selected <- apply_preprocessor(data$X_raw, final_pp, FALSE)
  data$X_scaled <- apply_preprocessor(data$X_raw, final_pp, TRUE)
  data$preprocessing_stats$resampling_method <- if (is.null(config$train_ratio)) "repeated_kfold" else "repeated_holdout"
  config$resampling_method <- data$preprocessing_stats$resampling_method
  config$positive_class <- levels(data$y)[2]
  config$prediction_regime <- "final_models_in_sample; performance_metrics_held_out"

  original_metrics <- list(
    rf_oob_error = NA_real_,
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
  
  permutation_results <- run_permutation_test(data$X_raw, data$y, config)
  
  # Profile ranking / final predictions
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("PROFILE RANKING")
  
  # NOTE: In CV mode, re-predicting per-model probabilities here can fail when
  # some models were trained on scaled matrices but predictions are attempted on
  # unscaled data (leading to empty Rankings/Risk Scores in the JSON).
  # We therefore build the ranking from the *final* probabilities we already
  # compute below.
  ranking <- NULL
  final_predictions <- NULL
  final_probabilities <- NULL
  final_model_used <- "ensemble"
  
  if (!use_ensemble && !is.null(best_model)) {
    
    final_model_used <- best_model
    
    model_obj <- final_models[[best_model]]
    
    if (!best_model %in% c("rf", "svm", "xgboost", "knn", "mlp")) {
      stop(sprintf("Unknown model '%s' selected as best model", best_model))
    }
    
    pred_fun <- get(paste0("predict_", best_model))
    
    pred_res <- predict_one(best_model, model_obj, final_X[[best_model]])
    
    final_predictions <- pred_res$predictions
    final_probabilities <- pred_res$probabilities
  }
  
  if (use_ensemble) {
    final_model_used <- if (use_soft) "soft_vote" else "hard_vote"
  }

  # If we selected an ensemble strategy, compute final predictions/probabilities
  # directly from the final models (trained on X_selected) using the same
  # hard/soft voting logic as in CV.
  if (use_ensemble && (is.null(final_predictions) || is.null(final_probabilities))) {
    preds <- list()
    probs <- list()

    if (!is.null(final_models$rf)) {
      res <- predict_one("rf", final_models$rf, final_X$rf)
      preds$rf <- res$predictions
      probs$rf <- res$probabilities
    }
    if (!is.null(final_models$svm)) {
      res <- predict_one("svm", final_models$svm, final_X$svm)
      preds$svm <- res$predictions
      probs$svm <- res$probabilities
    }
    if (!is.null(final_models$xgboost)) {
      res <- predict_one("xgboost", final_models$xgboost, final_X$xgboost)
      preds$xgboost <- res$predictions
      probs$xgboost <- res$probabilities
    }
    if (!is.null(final_models$knn)) {
      res <- predict_one("knn", final_models$knn, final_X$knn)
      preds$knn <- res$predictions
      probs$knn <- res$probabilities
    }
    if (!is.null(final_models$mlp)) {
      res <- predict_one("mlp", final_models$mlp, final_X$mlp)
      preds$mlp <- res$predictions
      probs$mlp <- res$probabilities
    }

    valid_pred_names <- names(preds)[sapply(names(preds), function(m) {
      is_valid_prediction(preds[[m]], data$y)
    })]
    valid_prob_names <- names(probs)[sapply(names(probs), function(m) {
      !is.null(probs[[m]]) && length(probs[[m]]) == length(data$y) && all(is.finite(probs[[m]])) && all(probs[[m]] >= 0 & probs[[m]] <= 1)
    })]

    if (final_model_used == "soft_vote" && length(valid_prob_names) >= 2) {
      weights <- rep(1 / length(valid_prob_names), length(valid_prob_names))
      soft_res <- soft_voting(probabilities_list = probs[valid_prob_names], weights = weights, class_levels = levels(data$y))
      final_predictions <- soft_res$predictions
      final_probabilities <- soft_res$probabilities
    } else if (final_model_used == "hard_vote" && length(valid_pred_names) >= 2) {
      final_predictions <- hard_voting(preds[valid_pred_names])
      # No probabilities for hard vote; approximate using mean prob if available
      if (length(valid_prob_names) >= 1) {
        final_probabilities <- as.numeric(rowMeans(do.call(cbind, probs[valid_prob_names]), na.rm = TRUE))
      }
    }
  }

  # ============================================================================
  # FALLBACK: If no ensemble/voting worked, use best single model for ranking
  # ============================================================================
  if (is.null(final_probabilities) || length(final_probabilities) == 0) {
    log_message("No ensemble probabilities available, attempting single model fallback for ranking...", "WARN")
    
    # Try each model in order of preference
    fallback_models <- c("rf", "svm", "xgboost", "knn", "mlp")
    for (fb_model in fallback_models) {
      if (!is.null(final_models[[fb_model]])) {
        tryCatch({
          pred_fun <- get(paste0("predict_", fb_model))
          fb_res <- predict_one(fb_model, final_models[[fb_model]], final_X[[fb_model]])
          if (!is.null(fb_res$probabilities) && length(fb_res$probabilities) == nrow(X_selected)) {
            final_probabilities <- fb_res$probabilities
            final_predictions <- fb_res$predictions
            final_model_used <- fb_model
            log_message(sprintf("Using %s as fallback for profile ranking", toupper(fb_model)))
            break
          }
        }, error = function(e) {
          log_message(sprintf("Fallback %s failed: %s", fb_model, e$message), "WARN")
        })
      }
    }
  }

  ranking <- build_profile_ranking(
    y = data$y,
    probabilities = final_probabilities,
    predicted_class = final_predictions,
    config = config,
    sample_ids = data$sample_ids,
    annotation = data$annotation,
    model_name = final_model_used
  )
  
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
  feature_boxplot_stats <- compute_feature_boxplot_stats(X_selected, data$y, top_feature_names, top_n = 20)
  
  # Survival analysis (if time/event variables provided)
  survival_analysis <- run_survival_analysis(
    X = X_selected,
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
  if (!is.null(final_models$xgboost) && inherits(final_models$xgboost$model, "xgb.Booster")) {
    log_message("Serializing XGBoost model using xgb.save.raw()...")
    final_models$xgboost_raw <- xgb.save.raw(final_models$xgboost$model)
    final_models$xgboost$model <- NULL  # Preserve wrapper class mapping
    final_models$xgboost_serialized <- TRUE
  }
  
  models_path <- file.path(config$output_dir, "trained_models.rds")
  final_models$preprocessor <- final_pp
  final_models$class_levels <- levels(data$y)
  final_models$model_scaling <- config$model_scaling
  final_models$scale_data <- config$scale_data
  saveRDS(final_models, models_path)
  log_message(sprintf("Models saved to: %s", models_path))
  log_message("NOTE: To load XGBoost model, use: models$xgboost$model <- xgb.load.raw(models$xgboost_raw)")
  
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
  json_output <- toJSON(batch_output, auto_unbox = TRUE, pretty = TRUE, digits = 6, na = "null")
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