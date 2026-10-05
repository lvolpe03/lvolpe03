cat("\014")
###################################################
# PECN 6180 - Classification Project - Luke Volpe #
###################################################

# Goal: predict (un)employment from other ACS variables
# Dataset source: IPUMS 2024 ACS

library(tidyverse)
library(rsample)
library(caret)
library(ggplot2)
library(forcats)
library(pROC)
library(ada)

# ========================================
# PART 1: DATA SETUP
# ========================================
# Load-in
acs24 <- read.csv("ACS2024.csv")

# Eliminate unneeded technical and detailed variables
acs24 <- acs24 |> 
  select(-c("YEAR", "SAMPLE", "SERIAL", "CBSERIAL", "HHWT", 
            "CLUSTER", "STRATA", "GQ", "PERNUM", "PERWT", 
            "RACED", "HISPAND", "EDUCD", "DEGFIELDD", "EMPSTATD",
            "MIGRATE1D", "VETSTATD", "MARRNO", "RACE", 
            "HCOVPRIV", "LABFORCE", "FTOTINC", "INCWAGE", "MIGRATE1",
            "COUNTYFIP", "STATEFIP", "DEGFIELD", "RACPACIS"))

# Narrow target var to employed/unemployed (only those in labor force)
acs24 <- acs24 |> filter(EMPSTAT == 1 | EMPSTAT == 2)

# Set cat variables as factors
acs24 <- acs24 |>
  mutate(across(c("HHTYPE", "REGION", "SEX",
          "MARST", "SPEAKENG", "YRSUSA2",
          "RACAMIND", "RACASIAN", "RACBLK", "RACWHT", "RACOTHER", 
          "HCOVANY", "SCHOOL", "EMPSTAT"), as.factor))

# Collapse certain factor levels for sparseness
acs24$EDUC <- factor(acs24$EDUC,
                  levels = c(00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 10, 11),
                  labels = c("Less_than_HS", "Less_than_HS", "Less_than_HS",
                             "Less_than_HS", "Less_than_HS", "Less_than_HS", "Finished_HS",
                             "College_Incomplete", "College_Incomplete", "College_Incomplete",
                             "4year_college", "Post_grad"),
                  ordered = TRUE)
acs24$CITIZEN <- factor(acs24$CITIZEN,
                        levels = c(0, 1, 2, 3),
                        labels = c("US_born", "Non_native_citizen", 
                                   "Non_native_citizen", "Non_citizen"))
acs24$HISPAN <- factor(acs24$HISPAN,
                       levels = c(0, 1, 2, 3, 4),
                       labels = c("Not_Hispanic", "Hispanic", "Hispanic", 
                                  "Hispanic", "Hispanic"))
acs24$VETSTAT <- factor(acs24$VETSTAT,
                       levels = c(0, 1, 2),
                       labels = c("Non_Vet", "Non_Vet", "Veteran"))

# Recode other factors
acs24$SEX <- fct_recode(acs24$SEX, "Male" = "1", "Female" = "2")
acs24$EMPSTAT <- fct_recode(acs24$EMPSTAT, "Employed" = "1", "Unemployed" = "2")
acs24$EMPSTAT <- factor(acs24$EMPSTAT, levels = c("Unemployed", "Employed"))
acs24$RACAMIND <- fct_recode(acs24$RACAMIND, "Not_native" = "1", "Native Amer" = "2")
acs24$RACASIAN <- fct_recode(acs24$RACASIAN, "Not_asian" = "1", "Asian" = "2")
acs24$RACBLK <- fct_recode(acs24$RACBLK, "Not_black" = "1", "Black" = "2")
acs24$RACWHT <- fct_recode(acs24$RACWHT, "Not_white" = "1", "White" = "2")
acs24$RACOTHER <- fct_recode(acs24$RACOTHER, "Not_other" = "1", "Other" = "2")
acs24$HCOVANY <- fct_recode(acs24$HCOVANY, "Not_insured" = "1", "Insured" = "2")
acs24$SCHOOL <- fct_recode(acs24$SCHOOL, "Non_student" = "1", "Student" = "2")
acs24$YRSUSA2 <- fct_recode(acs24$YRSUSA2, "N/A"  = "0", "0-5"  = "1", "6-10" = "2", 
                            "11-15" = "3", "16-20" = "4", "21+"  = "5")
acs24$SPEAKENG <- fct_recode(acs24$SPEAKENG, "no_english"   = "1", "only_english" = "3",
                             "very_well" = "4", "well" = "5", "not_well" = "6")
acs24$MARST <- fct_recode(acs24$MARST, "married_present" = "1", "married_absent" = "2",
                          "separated" = "3", "divorced" = "4", "widowed" = "5", "single" = "6")

# Engineered variables
# Unemp risk score, logged income, age^2, and 
acs24 <- acs24 |>
  mutate(
    risk_profile = as.integer(AGE < 25) +
      as.integer(EDUC %in% c("Less_than_HS", "Finished_HS")) +
      as.integer(HCOVANY == "Not_insured"),
    log_income = log(INCTOT - min(INCTOT) + 1),
    age2 = AGE^2,
    non_cit_lang = as.integer(CITIZEN == "Non_citizen" 
                               & SPEAKENG %in% c("no_english", "well", "not_well")))
# Final structure check
acs24 <- acs24 |> select(-INCTOT)
str(acs24)

# Shrink dataset to ~ 2,000 obs for computation purposes
set.seed(123)
acs24 <- acs24 %>%
  group_by(EMPSTAT) %>%
  slice_sample(prop = 0.001191) %>%
  ungroup()

# View summary stats and target class imbalance
summary(acs24)
table(acs24$EMPSTAT)

# Make a matrix version of dataset so AdaBoost can have numerical inputs
X <- model.matrix(EMPSTAT ~ . -1, data = acs24)
y <- acs24$EMPSTAT
acs_matrix <- data.frame(X, EMPSTAT = y)

# Train/test split
set.seed(123)
train_idx  <- createDataPartition(acs24$EMPSTAT, p = 0.8, list = FALSE)
train_data <- acs24[train_idx, ]
test_data  <- acs24[-train_idx, ]

train_data_num <- acs_matrix[train_idx, ]
test_data_num <- acs_matrix[-train_idx, ]
# ========================================
# PART 2: SHARED CV INFRASTRUCTURE
# ========================================
set.seed(123)
# 5-fold CV
cv_folds <- createFolds(train_data$EMPSTAT, k = 5, returnTrain = TRUE)

ctrl <- trainControl(method          = "cv",
                     index           = cv_folds,
                     classProbs      = TRUE,
                     summaryFunction = twoClassSummary,
                     savePredictions = "final")

# ========================================
# PART 3: FIT MODELS
# ========================================
### SINGLE TREE FAMILY ###
# Model 1: Deep Tree
fit_tree_deep <- train(EMPSTAT ~ .,
                       data      = train_data,
                       method    = "rpart",
                       tuneGrid  = data.frame(cp = 0.0001),
                       trControl = ctrl,
                       metric    = "ROC")
cat("Deep tree CV AUC:", round(fit_tree_deep$results$ROC, 3), "\n")

# Model 2: Pruned Tree
cp_grid <- data.frame(cp = seq(0.001, 0.10, by = 0.005))

fit_tree_pruned <- train(EMPSTAT ~ .,
                         data      = train_data,
                         method    = "rpart",
                         tuneGrid  = cp_grid,
                         trControl = ctrl,
                         metric    = "ROC")
cat("Pruned tree -- best cp:", fit_tree_pruned$bestTune$cp,
    " | CV AUC:", round(max(fit_tree_pruned$results$ROC), 3), "\n")


### ENSEMBLE FAMILY ###
# Model 3: Random Forest
rf_grid <- data.frame(mtry = c(3, 4, 5, 6, 7))
set.seed(123)
fit_rf <- train(EMPSTAT ~ .,
                data      = train_data,
                method    = "rf",
                tuneGrid  = rf_grid,
                trControl = ctrl,
                metric    = "ROC")
cat("Random Forest -- best mtry:", fit_rf$bestTune$mtry,
    " | CV AUC:", round(max(fit_rf$results$ROC), 3), "\n")

# Model 4: AdaBoost
ada_grid <- expand.grid(iter     = c(50, 75, 100),
                        maxdepth = c(1, 2),
                        nu       = c(0.01, 0.1, 1))
set.seed(123)
fit_ada <- train(EMPSTAT ~ .,
                 data      = train_data_num,
                 method    = "ada",
                 tuneGrid  = ada_grid,
                 trControl = ctrl,
                 metric    = "ROC")
cat("AdaBoost -- best iter:", fit_ada$bestTune$iter,
    " maxdepth:", fit_ada$bestTune$maxdepth,
    " nu:", fit_ada$bestTune$nu,
    " | CV AUC:", round(max(fit_ada$results$ROC), 3), "\n")


### LOGISTIC REGRESSION ###
set.seed(123)
fit_logit <- train(EMPSTAT ~ . ,
                   data = train_data,
                   method = "glm",
                   family = "binomial",
                   trControl = ctrl,
                   metric = "ROC")
cat("Logistic Regression CV AUC:", round(fit_logit$results$ROC, 3), "\n")


### KNN ###
set.seed(123)
fit_knn <- train(EMPSTAT ~ .,
                 data      = train_data_num,
                 method    = "knn",
                 preProcess = c("center", "scale"),
                 tuneGrid  = data.frame(k = c(5, 10, 15, 20, 25, 30)),
                 trControl = ctrl,
                 metric    = "ROC")
cat("KNN -- best k:", fit_knn$bestTune$k,
    " | CV AUC:", round(max(fit_knn$results$ROC), 3), "\n")

# ========================================
# PART 4: COMPARE MODELS
# ========================================
all_models <- resamples(list(
  "Deep Tree"    = fit_tree_deep,
  "Pruned Tree"  = fit_tree_pruned,
  "Rand. Forest" = fit_rf,
  "AdaBoost"     = fit_ada,
  "Logistic"     = fit_logit,
  "KNN"          = fit_knn))
summary(all_models)

# Model comparison table
cv_stats <- summary(all_models)$statistics
model_comp_table <- data.frame(
  Model       = rownames(cv_stats$ROC),
  CV_AUC      = round(cv_stats$ROC[, "Mean"], 3),
  Sensitivity = round(cv_stats$Sens[, "Mean"], 3),
  Specificity = round(cv_stats$Spec[, "Mean"], 3)
)
model_comp_table <- model_comp_table[order(-model_comp_table$CV_AUC), ]
rownames(model_comp_table) <- NULL
model_comp_table

# Mean CV AUC dot plot
auc_plot <- dotplot(all_models, metric = "ROC",
        main = "5-Fold CV AUC by Model")
auc_plot

# Best model by mean CV AUC
cv_auc <- summary(all_models)$statistics$ROC[, "Mean"]
best_model_name <- names(which.max(cv_auc))
cat("Best model by mean CV AUC:", best_model_name, "\n")
print(round(sort(cv_auc, decreasing = TRUE), 3))

# ========================================
# PART 5: EVALUATE BEST MODEL ON TEST DATA
# ========================================
# First find optimal threshold b/c my classes are so imbalanced
cv_preds <- fit_rf$pred
thresholds <- seq(0.05, 0.5, by = 0.05)
# Use F1 to assess
f1_scores <- sapply(thresholds, function(t) {
  pred_class <- ifelse(cv_preds$Unemployed >= t, "Unemployed", "Employed")
  pred_class <- factor(pred_class, levels = c("Unemployed", "Employed"))
  cm <- confusionMatrix(pred_class, cv_preds$obs, positive = "Unemployed")
  cm$byClass["F1"]
})
best_threshold <- thresholds[which.max(f1_scores)]
cat("Best threshold:", best_threshold, "\n")

# Run on test set 
pred_probs <- predict(fit_rf, test_data, type = "prob")

pred_class_tuned <- ifelse(pred_probs$Unemployed >= best_threshold,
                           "Unemployed", "Employed")
pred_class_tuned <- factor(pred_class_tuned, levels = c("Unemployed", "Employed"))

# Confusion matrix
confusionMatrix(pred_class_tuned, test_data$EMPSTAT,
                positive = "Unemployed", mode = "prec_recall")

# ROC curve
roc_obj <- roc(test_data$EMPSTAT, pred_probs$Unemployed,
               levels    = c("Employed", "Unemployed"),
               direction = "<")
par(pty = "s")
plot(roc_obj,
     main = "ROC Curve -- Random Forest",
     sub  = paste("AUC =", round(auc(roc_obj), 3)),
     col  = "purple2", lwd = 3)
abline(a = 1, b = -1, lty = 2, col = "gray60")
par(pty = "m")

cat("Test set AUC -- Random Forest: ", auc(roc_obj))

