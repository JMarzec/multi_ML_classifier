#!/usr/bin/env Rscript
# =============================================================================
# IntelliGenes-Style Multi-Method ML Classifier
# =============================================================================
# Implements various ML methods (RF, SVM, XGBoost, KNN, MLP) with voting
# classifiers, feature selection, and permutation testing for robust
# diagnostic prediction.
#
# INPUT FILES:
#   1. Expression matrix (tab-delimited .txt or .tsv)
#      - Rows: FEATURES, Columns: SAMPLES
#      - Column names are sample IDs (no separate sample ID column needed)
#      - First column can optionally be feature names (will be used as row names)
#   2. Sample annotation file (tab-delimited .txt or .tsv)
#      - Must contain sample IDs and target variable column
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
  
  # Output
  output_dir = "./results",
  output_json = "ml_results.json"
)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

log_message <- function(msg, level = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("[%s] %s: %s\n", timestamp, level, msg))
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

calculate_metrics <- function(actual, predicted, probabilities = NULL) {
  cm <- confusionMatrix(as.factor(predicted), as.factor(actual), positive = "1")
  
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
    roc_obj <- roc(actual, probabilities, quiet = TRUE)
    metrics$auroc <- as.numeric(auc(roc_obj))
    metrics$roc_curve <- data.frame(
      fpr = 1 - roc_obj$specificities,
      tpr = roc_obj$sensitivities
    )
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
  
  # Ensure X is numeric
  X <- as.data.frame(lapply(X, function(x) {
    if (is.character(x)) as.numeric(x) else x
  }))
  
  # Remove constant columns
  constant_cols <- sapply(X, function(x) length(unique(x[!is.na(x)])) <= 1)
  if (any(constant_cols)) {
    log_message(sprintf("Removing %d constant columns", sum(constant_cols)), "WARN")
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
    feature_names = colnames(X)
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

perform_feature_selection <- function(X, y, method, max_features, seed = 42) {
  switch(method,
    "forward" = forward_selection(X, y, max_features, seed),
    "backward" = backward_elimination(X, y, max_features, seed),
    "stepwise" = stepwise_selection(X, y, max_features, seed),
    "none" = colnames(X)
  )
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
predict_rf <- function(model_obj, X_test) {
  pred <- predict(model_obj$model, X_test)
  prob <- predict(model_obj$model, X_test, type = "prob")[, "1"]
  return(list(predictions = pred, probabilities = prob))
}

predict_svm <- function(model_obj, X_test) {
  pred <- predict(model_obj$model, as.matrix(X_test))
  prob_attr <- predict(model_obj$model, as.matrix(X_test), probability = TRUE)
  prob <- attr(prob_attr, "probabilities")[, "1"]
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
  prob <- prob_matrix[, 2]
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
  
  log_message(sprintf("Running %d-fold CV with %d repeats", config$n_folds, config$n_repeats))
  
  for (i in seq_along(folds)) {
    train_idx <- folds[[i]]
    test_idx <- setdiff(1:n_samples, train_idx)
    
    X_train <- X[train_idx, , drop = FALSE]
    X_test <- X[test_idx, , drop = FALSE]
    y_train <- y[train_idx]
    y_test <- y[test_idx]
    
    if (i %% 10 == 0) log_message(sprintf("  Fold %d/%d", i, length(folds)))
    
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
      all_results$rf[[i]] <- calculate_metrics(y_test, preds$rf, probs$rf)
      if (!is.null(models$rf$importance)) {
        importance_scores[[length(importance_scores) + 1]] <- models$rf$importance
      }
    }
    
    if (!is.null(models$svm)) {
      result <- predict_svm(models$svm, X_test)
      preds$svm <- result$predictions
      probs$svm <- result$probabilities
      all_results$svm[[i]] <- calculate_metrics(y_test, preds$svm, probs$svm)
    }
    
    if (!is.null(models$xgboost)) {
      result <- predict_xgboost(models$xgboost, X_test)
      preds$xgboost <- result$predictions
      probs$xgboost <- result$probabilities
      all_results$xgboost[[i]] <- calculate_metrics(y_test, preds$xgboost, probs$xgboost)
    }
    
    if (!is.null(models$knn)) {
      result <- predict_knn(models$knn, X_test)
      preds$knn <- result$predictions
      probs$knn <- result$probabilities
      all_results$knn[[i]] <- calculate_metrics(y_test, preds$knn, probs$knn)
    }
    
    if (!is.null(models$mlp)) {
      result <- predict_mlp(models$mlp, X_test)
      preds$mlp <- result$predictions
      probs$mlp <- result$probabilities
      all_results$mlp[[i]] <- calculate_metrics(y_test, preds$mlp, probs$mlp)
    }
    
    if (length(preds) > 1) {
      hard_pred <- hard_voting(preds)
      all_results$hard_vote[[i]] <- calculate_metrics(y_test, hard_pred)
      
      soft_result <- soft_voting(probs)
      all_results$soft_vote[[i]] <- calculate_metrics(y_test, soft_result$predictions, 
                                                       soft_result$probabilities)
    }
  }
  
  feature_importance <- data.frame(feature = colnames(X))
  if (length(importance_scores) > 0) {
    avg_importance <- rowMeans(do.call(cbind, importance_scores), na.rm = TRUE)
    feature_importance$importance <- avg_importance
    feature_importance <- feature_importance[order(-feature_importance$importance), ]
  }
  
  return(list(results = all_results, feature_importance = feature_importance))
}

# =============================================================================
# PERMUTATION TESTING
# =============================================================================

run_permutation_test <- function(X, y, config, selected_features = NULL) {
  n_permutations <- config$n_permutations
  log_message(sprintf("Running permutation testing with %d permutations", n_permutations))
  
  if (!is.null(selected_features)) X <- X[, selected_features, drop = FALSE]
  
  permutation_results <- list(rf_oob_error = numeric(n_permutations),
                              rf_auroc = numeric(n_permutations))
  
  set.seed(config$seed)
  folds <- createFolds(y, k = config$n_folds)
  
  for (p in 1:n_permutations) {
    if (p %% 10 == 0) log_message(sprintf("  Permutation %d/%d", p, n_permutations))
    
    y_permuted <- sample(y)
    
    rf_model <- tryCatch({
      randomForest(x = X, y = y_permuted, ntree = min(config$rf_ntree, 200), importance = FALSE)
    }, error = function(e) NULL)
    
    if (!is.null(rf_model)) {
      permutation_results$rf_oob_error[p] <- rf_model$err.rate[nrow(rf_model$err.rate), "OOB"]
    }
    
    cv_probs <- numeric(length(y))
    for (fold in folds) {
      train_idx <- setdiff(1:length(y), fold)
      rf_fold <- tryCatch({
        randomForest(x = X[train_idx, ], y = y_permuted[train_idx], ntree = 100, importance = FALSE)
      }, error = function(e) NULL)
      
      if (!is.null(rf_fold)) {
        cv_probs[fold] <- predict(rf_fold, X[fold, ], type = "prob")[, "1"]
      }
    }
    
    if (any(cv_probs > 0)) {
      roc_obj <- tryCatch({ roc(y_permuted, cv_probs, quiet = TRUE) }, error = function(e) NULL)
      if (!is.null(roc_obj)) permutation_results$rf_auroc[p] <- as.numeric(auc(roc_obj))
    }
  }
  
  return(permutation_results)
}

# =============================================================================
# PROFILE RANKING
# =============================================================================

rank_profiles <- function(X, y, models, config) {
  log_message("Ranking profiles by prediction confidence...")
  
  all_probs <- list()
  if (!is.null(models$rf)) all_probs$rf <- predict(models$rf$model, X, type = "prob")[, "1"]
  if (!is.null(models$svm)) {
    svm_pred <- predict(models$svm$model, as.matrix(X), probability = TRUE)
    all_probs$svm <- attr(svm_pred, "probabilities")[, "1"]
  }
  if (!is.null(models$xgboost)) all_probs$xgboost <- predict(models$xgboost$model, as.matrix(X))
  if (!is.null(models$mlp)) all_probs$mlp <- predict(models$mlp$model, as.matrix(X))[, 2]
  
  avg_prob <- rowMeans(do.call(cbind, all_probs), na.rm = TRUE)
  confidence <- abs(avg_prob - 0.5) * 2
  
  ranking <- data.frame(
    sample_index = 1:nrow(X),
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
    model_performance = results$summary,
    feature_importance = if (!is.null(results$feature_importance)) {
      head(results$feature_importance, 50)
    } else NULL,
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
  start_time <- Sys.time()
  log_message(paste(rep("=", 60), collapse = ""))
  log_message("Starting IntelliGenes-Style ML Classification Pipeline")
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
  
  ranking <- rank_profiles(X_selected, data$y, final_models, config)
  
  # Export results
  results <- list(
    summary = summary,
    feature_importance = cv_results$feature_importance,
    permutation = permutation_results,
    original_metrics = original_metrics,
    ranking = ranking,
    selected_features = selected_features
  )
  
  output_path <- file.path(config$output_dir, config$output_json)
  export_to_json(results, config, output_path)
  
  end_time <- Sys.time()
  log_message(sprintf("Pipeline completed in %.2f minutes", 
                      as.numeric(difftime(end_time, start_time, units = "mins"))))
  
  return(invisible(results))
}

# =============================================================================
# RUN
# =============================================================================

if (!interactive()) {
  run_pipeline(config)
}
