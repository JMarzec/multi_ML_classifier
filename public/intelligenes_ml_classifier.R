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
  annotation_file = "sample_annotation.txt",         # or .tsv
  
  # Column name for target variable in annotation file
  target_variable = "diagnosis",
  
  # Analysis mode: "full" (default) or "fast" (for testing, reduced accuracy)
  analysis_mode = "full",  # "full" or "fast"
  
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
  
  # Output
  output_dir = "./results",
  output_json = "ml_results.json"
)

# =============================================================================
# FAST MODE CONFIGURATION
# =============================================================================
# When analysis_mode = "fast", these settings override defaults for quick testing
# This reduces accuracy but provides rapid feedback for testing pipelines

get_effective_config <- function(config) {
  if (config$analysis_mode == "fast") {
    log_message("FAST MODE ENABLED - Using reduced settings for quick testing", "WARN")
    config$n_folds <- 2
    config$n_repeats <- 1
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
    confusionMatrix(predicted_factor, actual_factor, positive = "1")
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
  preprocessing_stats <- list(
    original_samples = length(sample_ids),
    original_features = ncol(X),
    missing_values = sum(is.na(X)),
    missing_pct = round(sum(is.na(X)) / (nrow(X) * ncol(X)) * 100, 2),
    class_distribution = as.list(table(y)),
    constant_features_removed = 0
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
  unscaled_expr <- X
  
  # Scale numeric features
  numeric_cols <- sapply(X, is.numeric)
  X[, numeric_cols] <- scale(X[, numeric_cols])
  
  log_message(sprintf("Final data: %d samples, %d features, %d classes",
                      nrow(X), ncol(X), length(levels(y))))
  log_message(sprintf("Class distribution: %s",
                      paste(names(table(y)), table(y), sep = "=", collapse = ", ")))
  
  return(list(
    X = X,
    y = y,
    sample_ids = sample_ids,
    feature_names = colnames(X),
    unscaled_expr = unscaled_expr,
    preprocessing_stats = preprocessing_stats
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

backward_elimination <- function(X, y, min_features = 5, seed = 42) {
  set.seed(seed)
  log_message("Performing backward elimination...")
  
  selected <- colnames(X)
  ctrl <- trainControl(method = "cv", number = 3)
  model <- train(x = X, y = y, method = "rf",
                trControl = ctrl, ntree = 100, tuneLength = 1)
  best_accuracy <- max(model$results$Accuracy)
  
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

perform_feature_selection <- function(X, y, method, max_features, seed) {
  if (method == "none" || ncol(X) <= max_features) {
    return(colnames(X))
  }
  
  switch(method,
    "forward" = forward_selection(X, y, max_features, seed),
    "backward" = backward_elimination(X, y, min_features = max_features, seed = seed),
    "stepwise" = stepwise_selection(X, y, max_features, seed),
    colnames(X)
  )
}

# =============================================================================
# MODEL TRAINING
# =============================================================================

#' Safe get probability for class "1" from various probability outputs
#' Handles matrix, vector, and NA cases gracefully
safe_get_prob <- function(prob_output, target_class = "1", n = NULL) {
  if (is.null(prob_output)) {
    if (!is.null(n)) return(rep(0.5, n))
    return(NA_real_)
  }
  
  # If it's a matrix or data frame with class columns
  if (is.matrix(prob_output) || is.data.frame(prob_output)) {
    if (target_class %in% colnames(prob_output)) {
      return(as.numeric(prob_output[, target_class]))
    } else if (ncol(prob_output) >= 2) {
      return(as.numeric(prob_output[, 2]))  # Assume second column is positive class
    } else if (ncol(prob_output) == 1) {
      return(as.numeric(prob_output[, 1]))
    }
  }
  
  # If it's already a vector
  if (is.numeric(prob_output) && length(prob_output) > 0) {
    return(as.numeric(prob_output))
  }
  
  if (!is.null(n)) return(rep(0.5, n))
  return(NA_real_)
}

train_rf <- function(X_train, y_train, config) {
  mtry <- config$rf_mtry %||% floor(sqrt(ncol(X_train)))
  
  model <- randomForest(
    x = X_train,
    y = y_train,
    ntree = config$rf_ntree,
    mtry = mtry,
    importance = TRUE
  )
  
  importance <- model$importance[, "MeanDecreaseGini"]
  names(importance) <- rownames(model$importance)
  
  return(list(model = model, importance = importance, oob_error = model$err.rate[nrow(model$err.rate), "OOB"]))
}

train_svm <- function(X_train, y_train, config) {
  gamma <- config$svm_gamma %||% (1 / ncol(X_train))
  
  model <- svm(
    x = as.matrix(X_train),
    y = y_train,
    kernel = config$svm_kernel,
    cost = config$svm_cost,
    gamma = gamma,
    probability = TRUE
  )
  
  return(list(model = model))
}

train_xgboost <- function(X_train, y_train, config) {
  # Convert labels to numeric (0/1)
  y_numeric <- as.numeric(y_train) - 1
  
  dtrain <- xgb.DMatrix(data = as.matrix(X_train), label = y_numeric)
  
  model <- xgb.train(
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      max_depth = config$xgb_max_depth,
      eta = config$xgb_eta,
      verbosity = 0
    ),
    data = dtrain,
    nrounds = config$xgb_nrounds,
    verbose = 0
  )
  
  return(list(model = model))
}

train_knn <- function(X_train, y_train, config) {
  return(list(
    X_train = X_train,
    y_train = y_train,
    k = config$knn_k
  ))
}

train_mlp <- function(X_train, y_train, config) {
  model <- nnet(
    x = as.matrix(X_train),
    y = class.ind(y_train),
    size = config$mlp_size,
    decay = config$mlp_decay,
    maxit = config$mlp_maxit,
    softmax = TRUE,
    trace = FALSE
  )
  
  return(list(model = model))
}

# =============================================================================
# MODEL PREDICTION
# =============================================================================

predict_rf <- function(model_obj, X_test) {
  pred <- predict(model_obj$model, X_test)
  prob_matrix <- predict(model_obj$model, X_test, type = "prob")
  prob <- safe_get_prob(prob_matrix, "1")
  return(list(predictions = pred, probabilities = prob))
}

predict_svm <- function(model_obj, X_test) {
  pred <- predict(model_obj$model, as.matrix(X_test), probability = TRUE)
  prob_matrix <- attr(pred, "probabilities")
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

# =============================================================================
# CROSS-VALIDATION
# =============================================================================

run_cv_all_methods <- function(X, y, config, selected_features = NULL) {
  set.seed(config$seed)
  if (!is.null(selected_features)) X <- X[, selected_features, drop = FALSE]
  
  n_samples <- nrow(X)
  folds <- createMultiFolds(y, k = config$n_folds, times = config$n_repeats)
  
  all_results <- list(rf = list(), svm = list(), xgboost = list(), 
                      knn = list(), mlp = list(), 
                      hard_vote = list(), soft_vote = list())
  
  importance_scores <- list()
  
  # Collect per-fold predictions for calibration curves export
  cv_predictions <- list(rf = NULL, svm = NULL, xgboost = NULL, knn = NULL, mlp = NULL, soft_vote = NULL)
  
  # Collect per-fold importance for stability analysis
  fold_importance <- list()
  
  # Collect per-fold AUROC and accuracy for distribution plots (actual data)
  fold_metrics <- list(
    rf = list(auroc = numeric(), accuracy = numeric()),
    svm = list(auroc = numeric(), accuracy = numeric()),
    xgboost = list(auroc = numeric(), accuracy = numeric()),
    knn = list(auroc = numeric(), accuracy = numeric()),
    mlp = list(auroc = numeric(), accuracy = numeric()),
    soft_vote = list(auroc = numeric(), accuracy = numeric())
  )
  
  log_message(sprintf("Running %d-fold CV with %d repeats", config$n_folds, config$n_repeats))
  n_folds_total <- length(folds)
  
  for (i in seq_along(folds)) {
    train_idx <- folds[[i]]
    test_idx <- setdiff(1:n_samples, train_idx)
    
    X_train <- X[train_idx, , drop = FALSE]
    X_test <- X[test_idx, , drop = FALSE]
    y_train <- y[train_idx]
    y_test <- y[test_idx]
    
    # Show progress bar
    show_progress(i, n_folds_total, "CV Progress")
    
    # Skip fold if single class in training set
    if (length(unique(as.character(y_train))) < 2) {
      log_message(sprintf("Skipping fold %d: single class in training set", i), "WARN")
      next
    }
    
    models <- list(
      rf = tryCatch(train_rf(X_train, y_train, config), error = function(e) NULL),
      svm = tryCatch(train_svm(X_train, y_train, config), error = function(e) NULL),
      xgboost = tryCatch(train_xgboost(X_train, y_train, config), error = function(e) NULL),
      knn = tryCatch(train_knn(X_train, y_train, config), error = function(e) NULL),
      mlp = tryCatch(train_mlp(X_train, y_train, config), error = function(e) NULL)
    )
    
    preds <- list()
    probs <- list()
    
    if (!is.null(models$rf)) {
      result <- predict_rf(models$rf, X_test)
      preds$rf <- result$predictions
      probs$rf <- result$probabilities
      metrics <- calculate_metrics(y_test, preds$rf, probs$rf)
      all_results$rf[[i]] <- metrics
      if (!is.null(models$rf$importance)) {
        importance_scores[[length(importance_scores) + 1]] <- models$rf$importance
        fold_importance[[length(fold_importance) + 1]] <- models$rf$importance
      }
      # Collect for calibration
      cv_predictions$rf <- rbind(cv_predictions$rf, data.frame(actual = as.character(y_test), prob = probs$rf))
      # Collect fold metrics
      if (!is.na(metrics$auroc)) fold_metrics$rf$auroc <- c(fold_metrics$rf$auroc, metrics$auroc)
      if (!is.na(metrics$accuracy)) fold_metrics$rf$accuracy <- c(fold_metrics$rf$accuracy, metrics$accuracy)
    }
    
    if (!is.null(models$svm)) {
      result <- predict_svm(models$svm, X_test)
      preds$svm <- result$predictions
      probs$svm <- result$probabilities
      metrics <- calculate_metrics(y_test, preds$svm, probs$svm)
      all_results$svm[[i]] <- metrics
      cv_predictions$svm <- rbind(cv_predictions$svm, data.frame(actual = as.character(y_test), prob = probs$svm))
      if (!is.na(metrics$auroc)) fold_metrics$svm$auroc <- c(fold_metrics$svm$auroc, metrics$auroc)
      if (!is.na(metrics$accuracy)) fold_metrics$svm$accuracy <- c(fold_metrics$svm$accuracy, metrics$accuracy)
    }
    
    if (!is.null(models$xgboost)) {
      result <- predict_xgboost(models$xgboost, X_test)
      preds$xgboost <- result$predictions
      probs$xgboost <- result$probabilities
      metrics <- calculate_metrics(y_test, preds$xgboost, probs$xgboost)
      all_results$xgboost[[i]] <- metrics
      cv_predictions$xgboost <- rbind(cv_predictions$xgboost, data.frame(actual = as.character(y_test), prob = probs$xgboost))
      if (!is.na(metrics$auroc)) fold_metrics$xgboost$auroc <- c(fold_metrics$xgboost$auroc, metrics$auroc)
      if (!is.na(metrics$accuracy)) fold_metrics$xgboost$accuracy <- c(fold_metrics$xgboost$accuracy, metrics$accuracy)
    }
    
    if (!is.null(models$knn)) {
      result <- predict_knn(models$knn, X_test)
      preds$knn <- result$predictions
      probs$knn <- result$probabilities
      metrics <- calculate_metrics(y_test, preds$knn, probs$knn)
      all_results$knn[[i]] <- metrics
      cv_predictions$knn <- rbind(cv_predictions$knn, data.frame(actual = as.character(y_test), prob = probs$knn))
      if (!is.na(metrics$auroc)) fold_metrics$knn$auroc <- c(fold_metrics$knn$auroc, metrics$auroc)
      if (!is.na(metrics$accuracy)) fold_metrics$knn$accuracy <- c(fold_metrics$knn$accuracy, metrics$accuracy)
    }
    
    if (!is.null(models$mlp)) {
      result <- predict_mlp(models$mlp, X_test)
      preds$mlp <- result$predictions
      probs$mlp <- result$probabilities
      metrics <- calculate_metrics(y_test, preds$mlp, probs$mlp)
      all_results$mlp[[i]] <- metrics
      cv_predictions$mlp <- rbind(cv_predictions$mlp, data.frame(actual = as.character(y_test), prob = probs$mlp))
      if (!is.na(metrics$auroc)) fold_metrics$mlp$auroc <- c(fold_metrics$mlp$auroc, metrics$auroc)
      if (!is.na(metrics$accuracy)) fold_metrics$mlp$accuracy <- c(fold_metrics$mlp$accuracy, metrics$accuracy)
    }
    
    if (length(preds) > 1) {
      hard_pred <- hard_voting(preds)
      all_results$hard_vote[[i]] <- calculate_metrics(y_test, hard_pred)
      
      soft_result <- soft_voting(probs)
      soft_metrics <- calculate_metrics(y_test, soft_result$predictions, soft_result$probabilities)
      all_results$soft_vote[[i]] <- soft_metrics
      cv_predictions$soft_vote <- rbind(cv_predictions$soft_vote, 
                                         data.frame(actual = as.character(y_test), prob = soft_result$probabilities))
      if (!is.na(soft_metrics$auroc)) fold_metrics$soft_vote$auroc <- c(fold_metrics$soft_vote$auroc, soft_metrics$auroc)
      if (!is.na(soft_metrics$accuracy)) fold_metrics$soft_vote$accuracy <- c(fold_metrics$soft_vote$accuracy, soft_metrics$accuracy)
    }
  }
  
  feature_importance <- data.frame(feature = colnames(X))
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
    fold_metrics = fold_metrics
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
      
      X_train <- X[train_idx, , drop = FALSE]
      X_test <- X[test_idx, , drop = FALSE]
      y_train <- y_permuted[train_idx]
      y_test <- y_permuted[test_idx]
      
      probs_list <- list()
      
      # RF
      rf_fold <- tryCatch(train_rf(X_train, y_train, config), error = function(e) NULL)
      if (!is.null(rf_fold)) {
        res <- predict_rf(rf_fold, X_test)
        cv_probs_rf[test_idx] <- res$probabilities
        cv_preds_rf[test_idx] <- as.character(res$predictions)
        probs_list$rf <- res$probabilities
      }
      
      # SVM
      svm_fold <- tryCatch(train_svm(X_train, y_train, config), error = function(e) NULL)
      if (!is.null(svm_fold)) {
        res <- predict_svm(svm_fold, X_test)
        cv_probs_svm[test_idx] <- res$probabilities
        cv_preds_svm[test_idx] <- as.character(res$predictions)
        probs_list$svm <- res$probabilities
      }
      
      # XGBoost
      xgb_fold <- tryCatch(train_xgboost(X_train, y_train, config), error = function(e) NULL)
      if (!is.null(xgb_fold)) {
        res <- predict_xgboost(xgb_fold, X_test)
        cv_probs_xgb[test_idx] <- res$probabilities
        cv_preds_xgb[test_idx] <- as.character(res$predictions)
        probs_list$xgb <- res$probabilities
      }
      
      # KNN
      knn_fold <- tryCatch(train_knn(X_train, y_train, config), error = function(e) NULL)
      if (!is.null(knn_fold)) {
        res <- predict_knn(knn_fold, X_test)
        cv_probs_knn[test_idx] <- res$probabilities
        cv_preds_knn[test_idx] <- as.character(res$predictions)
        probs_list$knn <- res$probabilities
      }
      
      # MLP
      mlp_fold <- tryCatch(train_mlp(X_train, y_train, config), error = function(e) NULL)
      if (!is.null(mlp_fold)) {
        res <- predict_mlp(mlp_fold, X_test)
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
# PROFILE RANKING
# =============================================================================

rank_profiles <- function(X, y, models, config, sample_ids = NULL) {
  log_message("Ranking profiles by prediction confidence...")

  all_probs <- list()

  if (!is.null(models$rf)) {
    prob_matrix <- tryCatch(predict(models$rf$model, X, type = "prob"), error = function(e) NULL)
    all_probs$rf <- safe_get_prob(prob_matrix, "1", n = nrow(X))
  }

  if (!is.null(models$svm)) {
    svm_pred <- tryCatch(predict(models$svm$model, as.matrix(X), probability = TRUE), error = function(e) NULL)
    prob_matrix <- if (!is.null(svm_pred)) attr(svm_pred, "probabilities") else NULL
    all_probs$svm <- safe_get_prob(prob_matrix, "1", n = nrow(X))
  }

  if (!is.null(models$xgboost)) {
    prob_vec <- tryCatch(predict(models$xgboost$model, as.matrix(X)), error = function(e) NULL)
    all_probs$xgboost <- safe_get_prob(prob_vec, n = nrow(X))
  }

  if (!is.null(models$mlp)) {
    prob_matrix <- tryCatch(predict(models$mlp$model, as.matrix(X)), error = function(e) NULL)
    all_probs$mlp <- safe_get_prob(prob_matrix, n = nrow(X))
  }

  # Keep only valid probability vectors with the expected length
  all_probs <- Filter(function(v) is.numeric(v) && length(v) == nrow(X), all_probs)

  avg_prob <- if (length(all_probs) == 0) {
    log_message("No usable probability outputs for profile ranking; using neutral 0.5", "WARN")
    rep(0.5, nrow(X))
  } else {
    as.numeric(rowMeans(do.call(cbind, all_probs), na.rm = TRUE))
  }

  confidence <- abs(avg_prob - 0.5) * 2

  # Use sample_ids if provided (for proper sample names)
  sample_names <- if (!is.null(sample_ids)) sample_ids else as.character(1:nrow(X))

  ranking <- data.frame(
    sample_index = 1:nrow(X),
    sample_id = sample_names,
    actual_class = as.character(y),
    ensemble_probability = avg_prob,
    predicted_class = ifelse(avg_prob > 0.5, "1", "0"),
    confidence = confidence,
    correct = as.character(y) == ifelse(avg_prob > 0.5, "1", "0")
  )

  ranking <- ranking[order(-ranking$confidence), ]
  ranking$rank <- 1:nrow(ranking)

  top_n <- ceiling(nrow(ranking) * (config$top_percent / 100))
  ranking$top_profile <- ranking$rank <= top_n

  return(ranking)
}

# =============================================================================
# RESULTS AGGREGATION
# =============================================================================

aggregate_results <- function(results_list) {
  methods <- names(results_list)
  
  summary <- lapply(methods, function(method) {
    if (length(results_list[[method]]) == 0) return(NULL)
    
    metrics <- c("accuracy", "sensitivity", "specificity", "precision", 
                "f1_score", "balanced_accuracy", "auroc", "kappa")
    
    stats <- lapply(metrics, function(m) {
      values <- sapply(results_list[[method]], function(r) r[[m]])
      values <- values[!is.na(values)]
      if (length(values) == 0) return(NULL)
      list(mean = mean(values), sd = sd(values), median = median(values),
           q25 = quantile(values, 0.25), q75 = quantile(values, 0.75),
           min = min(values), max = max(values))
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
  names(summary) <- methods
  
  return(summary)
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
  if (nrow(X) < 2) return(NULL)
  pca <- tryCatch(prcomp(X, center = TRUE, scale. = FALSE), error = function(e) NULL)
  if (is.null(pca)) return(NULL)

  coords <- as.data.frame(pca$x[, 1:2, drop = FALSE])
  colnames(coords) <- c("x", "y")
  coords$sample_id <- if (!is.null(sample_ids)) sample_ids else rownames(X)
  coords$actual_class <- as.character(y)

  var_expl <- (pca$sdev^2) / sum(pca$sdev^2)
  variance_explained <- list(pc1 = as.numeric(var_expl[1]), pc2 = as.numeric(var_expl[2]))

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

# =============================================================================
# JSON EXPORT
# =============================================================================

export_to_json <- function(results, config, output_path) {
  log_message(sprintf("Exporting results to: %s", output_path))
  
  output <- list(
    metadata = list(
      generated_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%S"),
      config = config,
      r_version = R.version.string
    ),
    preprocessing = results$preprocessing_stats,
    model_performance = results$summary,
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
    actual_distributions = results$fold_metrics,
    profile_ranking = if (!is.null(results$ranking)) {
      list(top_profiles = results$ranking[results$ranking$top_profile, ],
           all_rankings = results$ranking)
    } else NULL,
    selected_features = results$selected_features
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
    data$X, data$y, config$feature_selection_method,
    config$max_features, config$seed
  )
  log_message(sprintf("Selected %d features", length(selected_features)))

  # Cross-validation
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("CROSS-VALIDATION")

  cv_results <- run_cv_all_methods(data$X, data$y, config, selected_features)
  summary <- aggregate_results(cv_results$results)

  # Derived exports from CV
  calibration_curves <- compute_calibration_curves_from_cv(cv_results$cv_predictions, n_bins = 10)
  feature_importance_stability <- compute_feature_importance_stability(cv_results$fold_importance, top_n = 50)

  # Train final models
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("TRAINING FINAL MODELS")

  X_selected <- data$X[, selected_features, drop = FALSE]

  final_models <- list(
    rf = tryCatch(train_rf(X_selected, data$y, config), error = function(e) NULL),
    svm = tryCatch(train_svm(X_selected, data$y, config), error = function(e) NULL),
    xgboost = tryCatch(train_xgboost(X_selected, data$y, config), error = function(e) NULL),
    knn = tryCatch(train_knn(X_selected, data$y, config), error = function(e) NULL),
    mlp = tryCatch(train_mlp(X_selected, data$y, config), error = function(e) NULL)
  )

  original_metrics <- list(
    rf_oob_error = if (!is.null(final_models$rf)) final_models$rf$oob_error else NA,
    rf_auroc = if (!is.null(summary$rf$auroc)) summary$rf$auroc$mean else NA
  )

  # Permutation testing
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("PERMUTATION TESTING")

  permutation_results <- run_permutation_test(data$X, data$y, config, selected_features)

  # Profile ranking
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("PROFILE RANKING")

  ranking <- rank_profiles(X_selected, data$y, final_models, config, sample_ids = data$sample_ids)

  # Clustering exports (PCA, t-SNE, UMAP)
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("DIMENSIONALITY REDUCTION (PCA, t-SNE, UMAP)")
  
  clustering <- list(
    pca = compute_pca_embedding(X_selected, data$y, sample_ids = data$sample_ids),
    tsne = compute_tsne_embedding(X_selected, data$y, sample_ids = data$sample_ids),
    umap = compute_umap_embedding(X_selected, data$y, sample_ids = data$sample_ids)
  )
  
  if (is.null(clustering$tsne)) log_message("t-SNE not computed (Rtsne package not available or insufficient samples)", "WARN")
  if (is.null(clustering$umap)) log_message("UMAP not computed (umap package not available or insufficient samples)", "WARN")

  # Compute feature boxplot stats for top features
  top_feature_names <- if (!is.null(cv_results$feature_importance$feature)) {
    head(cv_results$feature_importance$feature, 20)
  } else {
    head(selected_features, 20)
  }
  feature_boxplot_stats <- compute_feature_boxplot_stats(data$unscaled_expr, data$y, top_feature_names, top_n = 20)

  # Export results
  results <- list(
    summary = summary,
    feature_importance = cv_results$feature_importance,
    feature_importance_stability = feature_importance_stability,
    feature_boxplot_stats = feature_boxplot_stats,
    calibration_curves = calibration_curves,
    clustering = clustering,
    permutation = permutation_results,
    original_metrics = original_metrics,
    fold_metrics = cv_results$fold_metrics,
    ranking = ranking,
    selected_features = selected_features,
    preprocessing_stats = data$preprocessing_stats,
    dataset_name = basename(config$expression_matrix_file)
  )

  output_path <- file.path(config$output_dir, config$output_json)
  export_to_json(results, config, output_path)

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
    summary = create_batch_summary(all_results)
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
      rf_auroc = if (!is.null(result$summary$rf$auroc)) result$summary$rf$auroc$mean else NA,
      soft_vote_auroc = if (!is.null(result$summary$soft_vote$auroc)) result$summary$soft_vote$auroc$mean else NA,
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
