import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Copy, Check, Download, Code, FileCode2 } from "lucide-react";
import { toast } from "sonner";
import type { MLResults } from "@/types/ml-results";

interface ModelExportTabProps {
  data: MLResults;
}

export function ModelExportTab({ data }: ModelExportTabProps) {
  const [copiedCode, setCopiedCode] = useState<string | null>(null);

  const selectedFeatures = data.selected_features || [];
  const config = data.metadata.config;

  const copyToClipboard = (code: string, label: string) => {
    navigator.clipboard.writeText(code);
    setCopiedCode(label);
    toast.success(`${label} code copied to clipboard`);
    setTimeout(() => setCopiedCode(null), 2000);
  };

  const downloadCode = (code: string, filename: string) => {
    const blob = new Blob([code], { type: "text/plain" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
    toast.success(`Downloaded ${filename}`);
  };

  const rfCode = `# =============================================================================
# Random Forest Prediction Script
# Generated from IntelliGenes ML Classifier
# =============================================================================

library(randomForest)

# Selected features from training
SELECTED_FEATURES <- c(
  ${selectedFeatures.map((f) => `"${f}"`).join(",\n  ")}
)

# Load your trained model (save this after training)
# saveRDS(rf_model, "rf_model.rds")
# rf_model <- readRDS("rf_model.rds")

#' Predict new samples using trained Random Forest model
#' @param new_data Data frame with new samples (rows = samples, cols = features)
#' @param rf_model Trained randomForest model object
#' @return Data frame with predictions and probabilities
predict_new_samples_rf <- function(new_data, rf_model) {
  # Ensure data has required features
  missing_features <- setdiff(SELECTED_FEATURES, colnames(new_data))
  if (length(missing_features) > 0) {
    stop(paste("Missing features:", paste(missing_features, collapse = ", ")))
  }
  
  # Select and order features
  X <- new_data[, SELECTED_FEATURES, drop = FALSE]
  
  # Scale features (use same scaling as training)
  X <- as.data.frame(scale(X))
  
  # Get predictions
  predictions <- predict(rf_model, X)
  probabilities <- predict(rf_model, X, type = "prob")
  
  result <- data.frame(
    sample = rownames(new_data),
    predicted_class = predictions,
    prob_class_0 = probabilities[, "0"],
    prob_class_1 = probabilities[, "1"],
    confidence = abs(probabilities[, "1"] - 0.5) * 2
  )
  
  return(result)
}

# Example usage:
# new_samples <- read.delim("new_expression_matrix.txt", row.names = 1)
# new_samples <- as.data.frame(t(new_samples))  # Transpose if rows are features
# predictions <- predict_new_samples_rf(new_samples, rf_model)
# print(predictions)
`;

  const svmCode = `# =============================================================================
# SVM Prediction Script
# Generated from IntelliGenes ML Classifier
# =============================================================================

library(e1071)

# Selected features from training
SELECTED_FEATURES <- c(
  ${selectedFeatures.map((f) => `"${f}"`).join(",\n  ")}
)

# SVM configuration used
SVM_KERNEL <- "${config.svm_kernel || "radial"}"

#' Predict new samples using trained SVM model
#' @param new_data Data frame with new samples
#' @param svm_model Trained svm model object
#' @return Data frame with predictions and probabilities
predict_new_samples_svm <- function(new_data, svm_model) {
  missing_features <- setdiff(SELECTED_FEATURES, colnames(new_data))
  if (length(missing_features) > 0) {
    stop(paste("Missing features:", paste(missing_features, collapse = ", ")))
  }
  
  X <- new_data[, SELECTED_FEATURES, drop = FALSE]
  X <- as.data.frame(scale(X))
  
  predictions <- predict(svm_model, as.matrix(X))
  prob_pred <- predict(svm_model, as.matrix(X), probability = TRUE)
  probabilities <- attr(prob_pred, "probabilities")
  
  result <- data.frame(
    sample = rownames(new_data),
    predicted_class = predictions,
    prob_class_0 = probabilities[, "0"],
    prob_class_1 = probabilities[, "1"],
    confidence = abs(probabilities[, "1"] - 0.5) * 2
  )
  
  return(result)
}

# Example usage:
# svm_model <- readRDS("svm_model.rds")
# predictions <- predict_new_samples_svm(new_samples, svm_model)
`;

  const xgboostCode = `# =============================================================================
# XGBoost Prediction Script
# Generated from IntelliGenes ML Classifier
# =============================================================================

library(xgboost)

# Selected features from training
SELECTED_FEATURES <- c(
  ${selectedFeatures.map((f) => `"${f}"`).join(",\n  ")}
)

#' Predict new samples using trained XGBoost model
#' @param new_data Data frame with new samples
#' @param xgb_model Trained xgboost model object
#' @return Data frame with predictions and probabilities
predict_new_samples_xgboost <- function(new_data, xgb_model) {
  missing_features <- setdiff(SELECTED_FEATURES, colnames(new_data))
  if (length(missing_features) > 0) {
    stop(paste("Missing features:", paste(missing_features, collapse = ", ")))
  }
  
  X <- new_data[, SELECTED_FEATURES, drop = FALSE]
  X <- as.data.frame(scale(X))
  
  probabilities <- predict(xgb_model, as.matrix(X))
  predictions <- factor(ifelse(probabilities > 0.5, "1", "0"), levels = c("0", "1"))
  
  result <- data.frame(
    sample = rownames(new_data),
    predicted_class = predictions,
    prob_class_0 = 1 - probabilities,
    prob_class_1 = probabilities,
    confidence = abs(probabilities - 0.5) * 2
  )
  
  return(result)
}

# Example usage:
# xgb_model <- xgb.load("xgboost_model.model")
# predictions <- predict_new_samples_xgboost(new_samples, xgb_model)
`;

  const ensembleCode = `# =============================================================================
# Ensemble Prediction Script (All Models)
# Generated from IntelliGenes ML Classifier
# =============================================================================

library(randomForest)
library(e1071)
library(xgboost)
library(class)
library(nnet)

# Selected features from training
SELECTED_FEATURES <- c(
  ${selectedFeatures.map((f) => `"${f}"`).join(",\n  ")}
)

# KNN configuration
KNN_K <- ${config.knn_k || 5}

#' Ensemble prediction using all trained models
#' @param new_data Data frame with new samples
#' @param models List containing all trained model objects
#' @param method Voting method: "soft" or "hard"
#' @return Data frame with ensemble predictions
predict_ensemble <- function(new_data, models, method = "soft") {
  missing_features <- setdiff(SELECTED_FEATURES, colnames(new_data))
  if (length(missing_features) > 0) {
    stop(paste("Missing features:", paste(missing_features, collapse = ", ")))
  }
  
  X <- new_data[, SELECTED_FEATURES, drop = FALSE]
  X <- as.data.frame(scale(X))
  
  # Get probabilities from each model
  all_probs <- list()
  all_preds <- list()
  
  if (!is.null(models$rf)) {
    all_probs$rf <- predict(models$rf, X, type = "prob")[, "1"]
    all_preds$rf <- predict(models$rf, X)
  }
  
  if (!is.null(models$svm)) {
    svm_pred <- predict(models$svm, as.matrix(X), probability = TRUE)
    all_probs$svm <- attr(svm_pred, "probabilities")[, "1"]
    all_preds$svm <- svm_pred
  }
  
  if (!is.null(models$xgboost)) {
    all_probs$xgboost <- predict(models$xgboost, as.matrix(X))
    all_preds$xgboost <- factor(ifelse(all_probs$xgboost > 0.5, "1", "0"), levels = c("0", "1"))
  }
  
  if (!is.null(models$knn)) {
    knn_pred <- knn(train = models$knn$X_train, test = X, 
                    cl = models$knn$y_train, k = KNN_K, prob = TRUE)
    knn_prob <- attr(knn_pred, "prob")
    all_probs$knn <- ifelse(knn_pred == "1", knn_prob, 1 - knn_prob)
    all_preds$knn <- knn_pred
  }
  
  if (!is.null(models$mlp)) {
    mlp_prob <- predict(models$mlp, as.matrix(X))
    all_probs$mlp <- mlp_prob[, 2]
    all_preds$mlp <- factor(ifelse(mlp_prob[, 2] > 0.5, "1", "0"), levels = c("0", "1"))
  }
  
  if (method == "soft") {
    # Soft voting: average probabilities
    avg_prob <- rowMeans(do.call(cbind, all_probs), na.rm = TRUE)
    ensemble_pred <- factor(ifelse(avg_prob > 0.5, "1", "0"), levels = c("0", "1"))
    ensemble_prob <- avg_prob
  } else {
    # Hard voting: majority vote
    pred_matrix <- do.call(cbind, lapply(all_preds, as.character))
    ensemble_pred <- apply(pred_matrix, 1, function(row) {
      names(which.max(table(row)))
    })
    ensemble_pred <- factor(ensemble_pred, levels = c("0", "1"))
    ensemble_prob <- rowMeans(do.call(cbind, all_probs), na.rm = TRUE)
  }
  
  result <- data.frame(
    sample = rownames(new_data),
    predicted_class = ensemble_pred,
    ensemble_probability = ensemble_prob,
    confidence = abs(ensemble_prob - 0.5) * 2
  )
  
  # Add individual model predictions
  for (name in names(all_probs)) {
    result[[paste0(name, "_prob")]] <- all_probs[[name]]
  }
  
  return(result)
}

# Example usage:
# models <- list(
#   rf = readRDS("rf_model.rds"),
#   svm = readRDS("svm_model.rds"),
#   xgboost = xgb.load("xgboost_model.model"),
#   knn = readRDS("knn_model.rds"),  # Contains X_train, y_train
#   mlp = readRDS("mlp_model.rds")
# )
# predictions <- predict_ensemble(new_samples, models, method = "soft")
`;

  const codes = [
    { id: "rf", label: "Random Forest", code: rfCode, icon: <FileCode2 className="w-4 h-4" /> },
    { id: "svm", label: "SVM", code: svmCode, icon: <FileCode2 className="w-4 h-4" /> },
    { id: "xgboost", label: "XGBoost", code: xgboostCode, icon: <FileCode2 className="w-4 h-4" /> },
    { id: "ensemble", label: "Ensemble", code: ensembleCode, icon: <Code className="w-4 h-4" /> },
  ];

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Code className="w-5 h-5 text-primary" />
            Export Prediction Code
          </CardTitle>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground mb-6">
            Download R code snippets to make predictions with the trained models. 
            These scripts include the selected features and model configurations from your analysis.
          </p>

          <Tabs defaultValue="rf" className="space-y-4">
            <TabsList className="grid grid-cols-4 w-full">
              {codes.map((item) => (
                <TabsTrigger key={item.id} value={item.id} className="gap-2">
                  {item.icon}
                  <span className="hidden sm:inline">{item.label}</span>
                </TabsTrigger>
              ))}
            </TabsList>

            {codes.map((item) => (
              <TabsContent key={item.id} value={item.id}>
                <div className="relative">
                  <div className="absolute top-3 right-3 flex gap-2 z-10">
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => copyToClipboard(item.code, item.label)}
                    >
                      {copiedCode === item.label ? (
                        <Check className="w-4 h-4 mr-1" />
                      ) : (
                        <Copy className="w-4 h-4 mr-1" />
                      )}
                      Copy
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => downloadCode(item.code, `${item.id}_prediction.R`)}
                    >
                      <Download className="w-4 h-4 mr-1" />
                      Download
                    </Button>
                  </div>
                  <pre className="bg-muted/50 rounded-lg p-4 pt-14 overflow-auto text-sm font-mono max-h-[500px]">
                    {item.code}
                  </pre>
                </div>
              </TabsContent>
            ))}
          </Tabs>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Selected Features ({selectedFeatures.length})</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-wrap gap-2">
            {selectedFeatures.map((feature) => (
              <span
                key={feature}
                className="px-3 py-1 bg-primary/10 text-primary rounded-full text-sm font-mono"
              >
                {feature}
              </span>
            ))}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
