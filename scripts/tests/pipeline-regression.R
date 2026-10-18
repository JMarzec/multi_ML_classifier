# Run: Rscript scripts/tests/pipeline-regression.R
suppressPackageStartupMessages({library(caret); library(randomForest); library(e1071); library(xgboost); library(class); library(nnet); library(pROC); library(jsonlite); library(dplyr); library(tidyr); library(survival)})
load_functions <- function(path) {
  e <- new.env(parent = globalenv())
  for (expr in parse(path)) {
    if (is.call(expr) && identical(expr[[1]], as.name("<-")) &&
        (is.call(expr[[3]]) && identical(expr[[3]][[1]], as.name("function")))) eval(expr, e)
  }
  e$VAR_QUANTILE <- .25; e$MIN_VALID_FOLD_FRACTION <- 1
  e$survival_available <- TRUE; e$tsne_available <- FALSE; e$umap_available <- FALSE
  e
}
cfg <- list(seed=42, n_folds=3, n_repeats=2, train_ratio=NULL, feature_selection_method="none", max_features=3,
  scale_data=FALSE, model_scaling=list(rf=FALSE, svm=TRUE, xgboost=FALSE, knn=TRUE, mlp=TRUE, dr=TRUE),
  rf_ntree=30, rf_mtry=NULL, svm_kernel="radial", svm_cost=1, svm_gamma=NULL, knn_k=3,
  mlp_size=3, mlp_decay=.01, mlp_maxit=30, xgb_max_depth=2, xgb_eta=.3, xgb_nrounds=3,
  n_permutations=2, target_variable="status", top_percent=10, time_variable=NULL, event_variable=NULL,
  analysis_mode="full", output_json="results.json")
set.seed(14)
ids <- as.character(c(1,10,2,setdiff(3:24,10)))
y <- factor(rep(c("case A","control/B"), each=12))
X <- data.frame(g1=rnorm(24)+as.numeric(y), g2=rnorm(24), g3=rnorm(24), constant=rep(1,24), row.names=ids)
root <- if (length(commandArgs(TRUE))) commandArgs(TRUE)[1] else tempfile("pipeline-")
dir.create(root, recursive=TRUE, showWarnings=FALSE)
expr <- data.frame(feature=names(X), t(X), check.names=FALSE)
write.table(expr, file.path(root,"expr.tsv"), sep="\t", row.names=FALSE, quote=FALSE)
annot <- data.frame(sample=ids, status=as.character(y), time=seq_len(24) * 3, event=rep(c(1,0,1),8))
write.table(annot[order(as.numeric(ids)),], file.path(root,"annot.tsv"), sep="\t", row.names=FALSE, quote=FALSE)
cfg$expression_matrix_file <- file.path(root,"expr.tsv"); cfg$annotation_file <- file.path(root,"annot.tsv")
cv <- load_functions("scripts/multi_ML_classifier.R")
d <- cv$load_data(cfg)
stopifnot(identical(d$sample_ids, ids), identical(as.character(d$y), as.character(y)))
pp <- cv$fit_preprocessor(X[1:18,], y[1:18], cfg)
held <- X[19:24,]; held$g1 <- 1e9
transformed_held <- cv$apply_preprocessor(held,pp,TRUE)
stopifnot(identical(pp$center, vapply(X[1:18,pp$features,drop=FALSE],mean,numeric(1))),
  max(transformed_held$g1) > 1e8)
scaled <- cv$apply_preprocessor(X[1:18,], pp, TRUE)
stopifnot(max(abs(colMeans(scaled))) < 1e-8, max(abs(vapply(scaled, sd, numeric(1))-1)) < 1e-8)
splits <- cv$make_splits(y,cfg)
stopifnot(length(splits)==6L)
for (r in 1:2) stopifnot(identical(sort(unlist(lapply(splits[vapply(splits,function(s)s$repeat_id==r,logical(1))], `[[`, "test"))),1:24))
holdout_cfg <- cfg; holdout_cfg$train_ratio <- .7
holdouts <- cv$make_splits(y,holdout_cfg)
stopifnot(length(holdouts)==2L, length(holdouts[[1]]$train)==18L, length(holdouts[[1]]$test)==6L)
partial <- list(rf=list(list(accuracy=.8)))
stopifnot(is.null(cv$aggregate_results(partial,2L,list(rf=1L))$rf))
actual <- factor(c(rep("0",10), rep("1",2)), levels=c("0","1"))
pred <- factor(c(rep("0",8),rep("1",2),"0","1"), levels=c("0","1"))
m <- cv$calculate_metrics(actual,pred)
stopifnot(m$sensitivity==.5, m$specificity==.8, m$confusion_matrix$tp==1)
single <- cv$calculate_metrics(actual,factor(rep("0",12),levels=c("0","1")), rep(.5,12))
stopifnot(single$accuracy==10/12, single$f1_score==0, single$auroc==.5)
rev <- cv$calculate_metrics(actual,pred, c(rep(.9,10),rep(.1,2)))
stopifnot(rev$auroc==0)
res <- cv$run_cv_all_methods(d$X_raw,d$X_raw,d$y,cfg)
stopifnot(res$attempted_folds==6L, res$valid_folds$rf==6L, res$valid_folds$soft_vote==6L)
perm <- cv$run_permutation_test(d$X_raw,d$y,cfg)
stopifnot(length(perm$per_model$rf$auroc)==2L, all(is.finite(perm$per_model$rf$auroc)))
zero_cfg <- cfg; zero_cfg$n_permutations <- 0L
stopifnot(length(cv$run_permutation_test(d$X_raw,d$y,zero_cfg)$rf_auroc)==0L)
full <- load_functions("scripts/multi_ML_classifier_full_training.R")
f <- full$load_and_preprocess_data(cfg)
stopifnot(identical(f$sample_ids,ids), identical(as.character(f$y),as.character(y)), f$preprocessing_stats$constant_features_removed==1L,
  identical(dimnames(f$X_raw),dimnames(f$X_scaled)), nrow(f$unscaled_expr)==24L)
fr <- full$train_all_models(f$X_raw,f$X_scaled,f$y,cfg)
stopifnot(identical(as.character(fr$rf$predictions),as.character(predict(fr$rf$model,f$X_raw))))
fp <- full$run_permutation_test(f$X_raw,f$X_scaled,f$y,cfg)
stopifnot(length(fp$null_aurocs$rf)==2L, all(is.finite(fp$null_aurocs$rf)))
cat("PASS: alignment, preprocessing, k-fold coverage, metrics, arbitrary labels, both model/permutation flows\n")
# Full pipelines include JSON export and model serialization.
cfg$time_variable <- "time"; cfg$event_variable <- "event"
cfg$output_dir <- file.path(root,"cv"); cv$run_pipeline(cfg)
cfg$output_dir <- file.path(root,"full"); full$run_pipeline(cfg)
for (mode in c("cv","full")) {
  output <- jsonlite::fromJSON(file.path(root,mode,"results.json"))
  stopifnot(nrow(output$profile_ranking$all_rankings)==24L)
  models <- readRDS(file.path(root,mode,"trained_models.rds"))
  stopifnot(is.raw(models$xgboost_raw), inherits(xgb.load.raw(models$xgboost_raw),"xgb.Booster.handle"))
  risk <- output$survival_analysis$model_risk_scores$soft_vote
  stopifnot(is.finite(risk$cutoff))
}
cat("Fixture outputs:", root, "\n")
