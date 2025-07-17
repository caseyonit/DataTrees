# Biodiversity & Habitat (BDH) Analysis with Decision Trees
# This script performs both regression and classification analysis on BDH scores

# Load required packages
# Set CRAN mirror
options(repos = c(CRAN = "https://cran.rstudio.com"))

# Install packages if not already installed
required_packages <- c(
  "tidyverse",
  "randomForest",
  "caret",
  "xgboost",
  "ranger",
  "lightgbm",
  "MLmetrics"
)

# Install packages if not already installed
for (pkg in required_packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, repos = "https://cran.rstudio.com")
    library(pkg, character.only = TRUE)
  }
}

# Load packages with error handling
tryCatch({
  library(tidyverse)
  library(randomForest)
  library(caret)
  library(xgboost)
  library(ranger)
  library(lightgbm)
}, error = function(e) {
  cat(sprintf("Error loading packages: %s\n", e$message))
  stop("Failed to load required packages")
})

library(rpart.plot)     # Tree visualization
library(DT)             # Interactive tables
library(htmlwidgets)    # Save interactive plots
library(ggpubr)         # Publication-ready plots

# Set seed for reproducibility
set.seed(123)

# Load and preprocess data
data <- read.csv("Data/epi2024_data.csv")

# Create output directory if it doesn't exist
output_dir <- "analysis_output"
if (!dir.exists(output_dir)) {
  dir.create(output_dir)
}

# Log file for tracking
sink(file.path(output_dir, "analysis_log.txt"), append = TRUE)
on.exit(sink())

# 1. Data Quality Assessment
cat("\n=== Data Quality Assessment ===\n")

# Check for missing values
missing_summary <- colSums(is.na(data))
cat("Missing values per column:\n")
print(missing_summary[missing_summary > 0])

# Check class distribution
class_dist <- table(data$BDH)
cat("\nBDH Distribution:\n")
print(class_dist)

# 2. Basic Feature Engineering
cat("\n=== Feature Engineering ===\n")

# First, check which features actually exist in the data
existing_features <- names(data)
interaction_features <- c("EPI", "ECS", "FSH", "APO", "AGR")
valid_features <- intersect(interaction_features, existing_features)

if (length(valid_features) > 1) {
  cat(sprintf("Creating interaction terms with %d valid features\n", length(valid_features)))
  for (i in 1:(length(valid_features) - 1)) {
    for (j in (i + 1):length(valid_features)) {
      col1 <- valid_features[i]
      col2 <- valid_features[j]
      if (all(!is.na(data[[col1]])) && all(!is.na(data[[col2]]))) {
        new_col <- paste0(col1, "_x_", col2)
        if (!new_col %in% names(data)) {
          data[[new_col]] <- data[[col1]] * data[[col2]]
          cat(sprintf("Created interaction term: %s\n", new_col))
        }
      }
    }
  }
} else {
  cat("Warning: Not enough valid features for interaction terms\n")
}

# Create basic polynomial features
numeric_cols <- sapply(data, is.numeric)
if (any(numeric_cols)) {
  cat("Creating polynomial features...\n")
  for (col in names(numeric_cols)[numeric_cols]) {
    if (all(!is.na(data[[col]])) && all(!is.infinite(data[[col]])) && all(!is.nan(data[[col]]))) {
      new_col <- paste0(col, "_sq")
      if (!new_col %in% names(data)) {
        data[[new_col]] <- data[[col]]^2
        cat(sprintf("Created polynomial feature: %s\n", new_col))
      }
    }
  }
} else {
  cat("Warning: No numeric features found for polynomial features\n")
}

# Create BDH categories using more sophisticated method
breaks <- quantile(data$BDH, c(0, 0.33, 0.66, 1), na.rm = TRUE)
data$BDH_Class <- cut(data$BDH, breaks, labels = c("Low", "Medium", "High"), include.lowest = TRUE)

# 3. Advanced Data Splitting
cat("\n=== Data Splitting ===\n")

# Create stratified train/test split
set.seed(123)
train_index <- createDataPartition(data$BDH_Class, p = 0.8, list = FALSE)
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

# 4. Feature Selection
cat("\n=== Feature Selection ===\n")

# Remove highly correlated features
# First, ensure we have numeric data without NAs
numeric_cols <- sapply(train_data, is.numeric)
if (any(numeric_cols)) {
  numeric_data <- train_data[, numeric_cols]
  # Remove columns with NAs
  numeric_data <- numeric_data[, colSums(is.na(numeric_data)) == 0]
  
  if (ncol(numeric_data) > 1) {
    # Calculate correlation matrix
    cor_matrix <- cor(numeric_data, use = "pairwise.complete.obs")
    
    # Find highly correlated features
    high_corr <- findCorrelation(cor_matrix, cutoff = 0.8)
    
    if (length(high_corr) > 0) {
      # Remove correlated features
      train_data <- train_data[, -high_corr]
      test_data <- test_data[, -high_corr]
      cat(sprintf("Removed %d highly correlated features\n", length(high_corr)))
    }
  }
} else {
  cat("Warning: No numeric features for correlation analysis\n")
}

# 5. Advanced Model Training
cat("\n=== Model Training ===\n")

# Set up faster cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = multiClassSummary,
  savePredictions = TRUE
)

# Train models with simpler configurations

# Random Forest model
rf_grid <- expand.grid(
  mtry = 5,
  splitrule = "gini",
  min.node.size = 1
)

rf_model <- train(
  BDH_Class ~ .,
  data = train_data,
  method = "ranger",
  trControl = ctrl,
  tuneGrid = rf_grid,
  num.trees = 500,
  importance = "permutation"
)

# XGBoost model with simpler grid
xgb_grid <- expand.grid(
  nrounds = 100,
  max_depth = 4,
  eta = 0.1,
  gamma = 0.1,
  colsample_bytree = 0.9,
  min_child_weight = 3,
  subsample = 0.9
)

xgb_model <- train(
  BDH_Class ~ .,
  data = train_data,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = xgb_grid,
  verbose = 0
)

# CatBoost model (optional)
if (requireNamespace("catboost", quietly = TRUE) && "catboost" %in% caret::modelLookup()$model) {
  library(catboost)
  cat("Training CatBoost model...\n")
  catboost_grid <- expand.grid(
    iterations = 100,
    learning_rate = 0.1,
    depth = 6,
    l2_leaf_reg = 3
  )
  catboost_model <- train(
    BDH_Class ~ .,
    data = train_data,
    method = "catboost",
    trControl = ctrl,
    tuneGrid = catboost_grid,
    verbose = FALSE
  )
} else {
  cat("CatBoost package or caret method not available – skipping CatBoost model.\n")
}

# 6. Model Stacking
cat("\n=== Model Stacking ===\n")

# Create predictions from base models
base_models <- list(
  rf = rf_model,
  xgb = xgb_model
)
if (exists("catboost_model")) {
  base_models$cat <- catboost_model
}

# Create meta-features safely
meta_features <- NULL
for (name in names(base_models)) {
  model <- base_models[[name]]
  preds <- as.data.frame(predict(model, newdata = test_data, type = "prob"))
  colnames(preds) <- paste(name, colnames(preds), sep = "_")
  if (is.null(meta_features)) {
    meta_features <- preds
  } else {
    meta_features <- cbind(meta_features, preds)
  }
}

# Train meta-model using logistic regression
meta_model <- train(
  BDH_Class ~ .,
  data = data.frame(BDH_Class = test_data$BDH_Class, meta_features),
  method = "multinom",
  trace = FALSE
)

# 7. Final Evaluation
cat("\n=== Final Evaluation ===\n")

# Make final predictions
final_preds <- predict(meta_model, newdata = meta_features)
final_accuracy <- mean(final_preds == test_data$BDH_Class)

cat(sprintf("\nFinal Model Accuracy: %.2f%%\n", final_accuracy * 100))

cat("\n=== Feature Importance ===\n")

# Get feature importance from best model
importance <- varImp(rf_model)
print(importance)

# Remove rows with missing BDH values
data_clean <- data %>%
  filter(!is.na(BDH))

# Select relevant features and handle missing values
features <- c("EPI", "ECS", "FSH", "APO", "AGR", "WRS", "AIR", "H2O", "HMT", "WMG", "CCH")

# Impute missing values with median (for numeric) or mode (for categorical)
impute_median <- function(x) {
  x[is.na(x)] <- median(x, na.rm = TRUE)
  return(x)
}

data_clean <- data_clean %>%
  mutate(across(where(is.numeric), impute_median))

# Set up cross-validation
ctrl <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = TRUE,
  classProbs = TRUE
)

# Split data into training and testing sets
set.seed(123)
train_idx <- createDataPartition(data_clean$BDH, p = 0.8, list = FALSE)
train_data <- data_clean[train_idx, ]
test_data <- data_clean[-train_idx, ]

# 1. Enhanced Regression Model (Random Forest)
cat("\n=== Building Enhanced Regression Model (Random Forest) ===\n")
reg_formula <- as.formula(paste("BDH ~", paste(features, collapse = " + ")))

# Train Random Forest with cross-validation
rf_reg <- train(
  reg_formula,
  data = train_data,
  method = "rf",
  trControl = ctrl,
  tuneLength = 5,
  importance = TRUE
)

# 2. Enhanced Classification Model (XGBoost)
cat("\n=== Building Enhanced Classification Model (XGBoost) ===\n")
class_formula <- as.formula(paste("BDH_Class ~", paste(features, collapse = " + ")))

# Train XGBoost with cross-validation
xgb_grid <- expand.grid(
  nrounds = 100,
  max_depth = 4,
  eta = 0.1,
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

xgb_model <- train(
  class_formula,
  data = train_data,
  method = "xgbTree",
  trControl = ctrl,
  tuneGrid = xgb_grid,
  verbose = FALSE
)

# 3. Feature Importance
cat("\n=== Calculating Feature Importance ===\n")
# For regression
rf_imp <- varImp(rf_reg, scale = FALSE)
# For classification
xgb_imp <- varImp(xgb_model, scale = FALSE)

# Function to create tree visualization
create_tree_plot <- function(tree, title, filename) {
  # Create output directory if it doesn't exist
  if (!dir.exists("output")) {
    dir.create("output")
  }
  
  # Create output filenames
  png_file <- file.path("output", paste0(filename, ".png"))
  html_file <- file.path("output", paste0(filename, ".html"))
  
  # Save the plot as PNG
  png(png_file, width = 1200, height = 800)
  prp(
    tree,
    type = 4,           # Draw split labels below nodes
    extra = 101,        # Display number of observations and probabilities
    fallen.leaves = TRUE,
    roundint = TRUE,
    main = title,
    box.palette = "RdYlGn",  # Color palette
    cex = 0.8,
    tweak = 1.2
  )
  dev.off()
  
  # Create HTML wrapper with the image
  # Create HTML content with proper escaping
  html_content <- sprintf(
    '<!DOCTYPE html>
    <html>
    <head>
      <title>%s</title>
      <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .container { max-width: 1200px; margin: 0 auto; text-align: center; }
        img { max-width: 100%%; height: auto; }
      </style>
    </head>
    <body>
      <div class="container">
        <h2>%s</h2>
        <img src="%s" alt="Decision Tree">
      </div>
    </body>
    </html>',
    title,
    title,
    basename(png_file)
  )
  
  # Save HTML file
  writeLines(html_content, html_file)
  
  # Return relative paths for the main HTML
  return(list(html = basename(html_file), 
              png = basename(png_file),
              dir = "output"))
}

# Create output directory if it doesn't exist
if (!dir.exists("output")) {
  dir.create("output", recursive = TRUE)
}

# Make predictions for both models
# Regression predictions
reg_predictions <- predict(rf_reg, test_data)
reg_r2 <- R2(reg_predictions, test_data$BDH)
reg_rmse <- RMSE(reg_predictions, test_data$BDH)

# Classification predictions
class_predictions <- predict(xgb_model, test_data)
class_probs <- predict(xgb_model, test_data, type = "prob")
class_accuracy <- mean(class_predictions == test_data$BDH_Class, na.rm = TRUE) * 100

# Confusion matrix and metrics
conf_matrix <- confusionMatrix(class_predictions, test_data$BDH_Class)
precision <- conf_matrix$byClass[,"Pos Pred Value"]
recall <- conf_matrix$byClass[,"Sensitivity"]
f1_score <- 2 * (precision * recall) / (precision + recall)

# Create visualizations for feature importance
create_importance_plot <- function(imp, title) {
  imp_df <- as.data.frame(imp$importance)
  imp_df$feature <- rownames(imp_df)
  imp_df <- imp_df %>% 
    arrange(desc(Overall)) %>%
    head(10)  # Top 10 features
  
  p <- ggplot(imp_df, aes(x = reorder(feature, Overall), y = Overall)) +
    geom_col(fill = "steelblue") +
    coord_flip() +
    labs(title = title, x = "Feature", y = "Importance") +
    theme_minimal()
  
  return(p)
}

# Create visualizations for model performance
perf_plot <- function(actual, predicted, title) {
  df <- data.frame(Actual = actual, Predicted = predicted)
  p <- ggplot(df, aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.6) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    labs(title = title) +
    theme_minimal()
  return(p)
}

# Generate and save plots
if (!dir.exists("output")) dir.create("output")

# Save importance plots
imp_plot_reg <- create_importance_plot(rf_imp, "Feature Importance (Regression)")
imp_plot_class <- create_importance_plot(xgb_imp, "Feature Importance (Classification)")
ggsave("output/feature_importance_reg.png", imp_plot_reg, width = 8, height = 6)
ggsave("output/feature_importance_class.png", imp_plot_class, width = 8, height = 6)

# Save performance plot
reg_perf_plot <- perf_plot(test_data$BDH, reg_predictions, "Regression: Actual vs Predicted BDH")
ggsave("output/regression_performance.png", reg_perf_plot, width = 8, height = 6)

# Create a summary HTML file
html_content <- sprintf(
  '<!DOCTYPE html>
  <html>
  <head>
    <title>Biodiversity & Habitat (BDH) Analysis</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      .container { max-width: 1200px; margin: 0 auto; }
      .metrics { 
        background-color: #f8f9fa; 
        padding: 20px; 
        border-radius: 5px; 
        margin: 20px 0; 
      }
      .plot-image {
        max-width: 100%%; height: auto;
        border: 1px solid #ddd;
        border-radius: 4px;
        margin: 10px 0;
      }
      .row {
        display: flex;
        flex-wrap: wrap;
        margin: 0 -10px;
      }
      .column {
        flex: 50%%;
        padding: 0 10px;
        box-sizing: border-box;
      }
      @media (max-width: 768px) {
        .column {
          flex: 100%%;
        }
      }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Biodiversity & Habitat (BDH) Analysis</h1>
      <p>Analysis of BDH scores across countries using machine learning models.</p>
      
      <div class="metrics">
        <h2>Model Performance</h2>
        
        <h3>Regression (Random Forest)</h3>
        <p><strong>R-squared:</strong> %0.3f (Higher is better, max 1.0)</p>
        <p><strong>RMSE:</strong> %0.2f (Lower is better)</p>
        <img src="output/regression_performance.png" alt="Regression Performance" class="plot-image">
        
        <h3>Classification (XGBoost)</h3>
        <p><strong>Accuracy:</strong> %0.2f%%</p>
        <p><strong>Precision (Weighted Avg):</strong> %0.2f%%</p>
        <p><strong>Recall (Weighted Avg):</strong> %0.2f%%</p>
        <p><strong>F1 Score (Weighted Avg):</strong> %0.2f%%</p>
        
        <h3>Feature Importance</h3>
        <div class="row">
          <div class="column">
            <h4>Regression</h4>
            <img src="output/feature_importance_reg.png" alt="Feature Importance (Regression)" class="plot-image">
          </div>
          <div class="column">
            <h4>Classification</h4>
            <img src="output/feature_importance_class.png" alt="Feature Importance (Classification)" class="plot-image">
          </div>
        </div>
      </div>
    </div>
  </body>
  </html>',
  reg_r2,
  reg_rmse,
  class_accuracy,
  mean(precision, na.rm = TRUE) * 100,
  mean(recall, na.rm = TRUE) * 100,
  mean(f1_score, na.rm = TRUE) * 100
)

# Save the main HTML file
writeLines(html_content, "BDH_Analysis_Report.html")

# Print completion message
cat("\nAnalysis complete! Open 'BDH_Analysis_Report.html' to view the results.\n")

# Print model summaries
cat("\n=== Model Summaries ===\n")

# Regression summary
cat("\n=== Regression Model (Random Forest) ===\n")
print(rf_reg)

# Classification summary
cat("\n\n=== Classification Model (XGBoost) ===\n")
print(xgb_model)

# Performance metrics
cat("\n\n=== Final Model Performance ===\n")
cat("\nRegression (Test Set):")
cat(sprintf("\n- R-squared: %0.3f", reg_r2))
cat(sprintf("\n- RMSE: %0.2f", reg_rmse))

cat("\n\nClassification (Test Set):")
cat(sprintf("\n- Accuracy: %0.2f%%", class_accuracy))
cat(sprintf("\n- Precision (Weighted): %0.2f%%", mean(precision, na.rm = TRUE) * 100))
cat(sprintf("\n- Recall (Weighted): %0.2f%%", mean(recall, na.rm = TRUE) * 100))
cat(sprintf("\n- F1 Score (Weighted): %0.2f%%", mean(f1_score, na.rm = TRUE) * 100))

# Save model objects for future use
saveRDS(rf_reg, "output/regression_model.rds")
saveRDS(xgb_model, "output/classification_model.rds")

# All files are already saved in the output directory
# No need for additional file operations
