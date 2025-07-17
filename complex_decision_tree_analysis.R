# Advanced Interactive Decision Tree Analysis
# This script implements a comprehensive decision tree analysis with multiple algorithms and interactive visualizations

# Load required packages
suppressPackageStartupMessages({
  # Core
  library(tidyverse)      # Data manipulation
  library(data.table)     # Fast data processing
  library(future)         # Parallel processing
  library(future.apply)   # Parallel apply functions
  
  # Modeling
  library(rpart)          # Decision trees
  library(partykit)       # Conditional inference trees
  library(evtree)         # Evolutionary trees
  library(ranger)         # Random Forest
  library(xgboost)        # Gradient Boosting
  library(caret)          # Model training and evaluation
  
  # Visualization
  library(rpart.plot)     # Tree visualization
  library(ggparty)        # Advanced tree visualization
  library(plotly)         # Interactive plots
  library(ggthemes)       # Advanced plotting themes
  library(patchwork)      # Combine plots
  library(ggrepel)        # Better text labels
  library(DT)             # Interactive tables
  
  # Model interpretability
  library(vip)            # Variable importance
  library(DALEX)          # Model explainability
  library(iml)            # Interpretable Machine Learning
  
  # Feature engineering
  library(recipes)        # Feature engineering
  library(missRanger)     # Missing value imputation
  
  # Reporting
  library(rmarkdown)      # Dynamic reporting
  library(knitr)          # Dynamic report generation
  library(kableExtra)     # Enhanced table formatting
})

# Enable parallel processing
plan(multisession, workers = availableCores() - 1)
options(future.rng.onMisuse = "ignore")

# Set seed for reproducibility
set.seed(123)

# 1. Data Loading and Enhanced Preprocessing
cat("=== Loading and Preprocessing Data ===\n")

# Define helper functions
safe_log <- function(x) {
  ifelse(x > 0, log(x + 1e-10), NA)
}

create_interactions <- function(df, vars) {
  if (length(vars) < 2) return(df)
  
  combs <- combn(vars, 2, simplify = FALSE)
  for (pair in combs) {
    if (all(pair %in% names(df))) {
      col_name <- paste(pair, collapse = "_x_")
      df[[col_name]] <- df[[pair[1]]] * df[[pair[2]]]
    }
  }
  return(df)
}

# Set up logging
log_message <- function(msg, level = "INFO") {
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  cat(sprintf("[%s] [%s] %s\n", timestamp, level, msg))
}

# Create output directory
output_dir <- "analysis_output"
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
}

# Start logging
sink(file.path(output_dir, "analysis_log.txt"), append = FALSE, split = TRUE)
on.exit(sink())

log_message("Starting Advanced Decision Tree Analysis")

# Enhanced data loading with progress tracking
log_message("Loading and validating data...")
  
tryCatch({
  # Data loading with multiple format support
  data_file <- "Data/epi2024_data.csv"
  if (!file.exists(data_file)) {
    stop(paste("Data file not found:", normalizePath(data_file, mustWork = FALSE)))
  }
  
  log_message(paste("Loading data from:", normalizePath(data_file)))
  
  # Read data with better error handling and progress
  data <- data.table::fread(
    data_file,
    stringsAsFactors = TRUE,
    data.table = FALSE,
    showProgress = TRUE,
    na.strings = c("", "NA", "N/A", "NULL", "NaN", "Inf", "-Inf")
  )

  # Enhanced column validation
  required_cols <- c("BDH", "EPI", "ECS", "FSH", "APO", "AGR", "WRS", "AIR", "H2O", "HMT", "WMG", "CCH")
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
  }
  
  log_message(sprintf("Data loaded successfully with %d rows and %d columns", 
                     nrow(data), ncol(data)))
  
  # Convert character columns to factors
  char_cols <- sapply(data, is.character)
  data[char_cols] <- lapply(data[char_cols], as.factor)
  
  # Initial data quality report
  data_quality <- data.frame(
    Column = names(data),
    Type = sapply(data, class),
    Missing = colSums(is.na(data)),
    Unique = sapply(data, function(x) length(unique(x))),
    row.names = NULL
  )
  
  log_message("Initial data quality check completed")
  
  # Enhanced data profiling
  log_message("Generating data profile...")
  
  # Save data profile
  sink(file.path(output_dir, "data_profile.txt"))
  cat("=== DATA PROFILE ===\n\n")
  cat("1. DIMENSIONS\n")
  cat("Rows:", nrow(data), "\n")
  cat("Columns:", ncol(data), "\n\n")
  
  cat("2. DATA TYPES\n")
  print(table(sapply(data, class)))
  cat("\n")
  
  cat("3. MISSING VALUES\n")
  na_summary <- data.frame(
    Column = names(data),
    Missing = colSums(is.na(data)),
    Pct_Missing = round(colMeans(is.na(data)) * 100, 2)
  )
  print(na_summary[na_summary$Missing > 0, ])
  cat("\n")
  
  cat("4. NUMERICAL SUMMARY\n")
  print(summary(data))
  
  sink()
  
  # Generate correlation plot for numerical variables
  num_vars <- sapply(data, is.numeric)
  if (sum(num_vars) > 1) {
    corr_matrix <- cor(data[, num_vars], use = "pairwise.complete.obs")
    png(file.path(output_dir, "correlation_plot.png"), 
        width = 10, height = 8, units = "in", res = 300)
    corrplot::corrplot(corr_matrix, method = "color", 
                       type = "upper", order = "hclust",
                       addCoef.col = "black",
                       tl.col = "black", tl.srt = 45,
                       diag = FALSE)
    dev.off()
  }
  
  # Advanced Feature Engineering
  log_message("Performing feature engineering...")
  
  # Create a copy of original data
  data_original <- data
  
  # Define numeric and factor columns
  num_cols <- names(data)[sapply(data, is.numeric)]
  factor_cols <- names(data)[sapply(data, is.factor)]
  
  # 1. Handle missing values using missRanger (Random Forest imputation)
  log_message("Imputing missing values...")
  
  # Only impute numeric columns with NAs
  cols_to_impute <- names(data)[sapply(data, function(x) is.numeric(x) && any(is.na(x)))]
  
  if (length(cols_to_impute) > 0) {
    data_imputed <- missRanger::missRanger(
      data,
      formula = . ~ . -1,  # Don't use response for imputation
      num.trees = 100,
      verbose = 2
    )
    
    # Replace only the imputed columns
    data[cols_to_impute] <- data_imputed[cols_to_impute]
  }
  
  # 2. Create target variable (BDH categories using quantiles)
  log_message("Creating target variable...")
  
  # Define breaks using quantiles for balanced classes
  breaks <- quantile(data$BDH, probs = c(0, 0.3, 0.7, 1), na.rm = TRUE)
  breaks[1] <- -Inf  # Ensure all values are included
  breaks[length(breaks)] <- Inf
  
  data <- data %>%
    # Ensure BDH is numeric
    mutate(BDH = as.numeric(BDH)) %>%
    # Remove rows with missing BDH values
    filter(!is.na(BDH)) %>%
    # Create BDH categories using quantiles
    mutate(
      BDH_Class = factor(
        cut(BDH, 
            breaks = breaks,
            labels = c("Low", "Medium", "High"),
            include.lowest = TRUE),
        levels = c("Low", "Medium", "High")
      )
    )
  
  # 3. Feature Engineering
  log_message("Creating new features...")
  
  # Create interaction terms for all pairs of numeric variables
  data <- create_interactions(data, num_cols)
  
  # Add polynomial features
  poly_degree <- 2  # Can be increased for more complex interactions
  for (col in num_cols) {
    if (sd(data[[col]], na.rm = TRUE) > 0) {  # Avoid constant columns
      data[[paste0(col, "_sq")]] <- data[[col]]^2
      data[[paste0(col, "_sqrt")]] <- sqrt(abs(data[[col]]))
      data[[paste0(col, "_log")]] <- safe_log(data[[col]])
    }
  }
  
  # Add statistical features
  data$row_mean <- rowMeans(data[num_cols], na.rm = TRUE)
  data$row_median <- apply(data[num_cols], 1, median, na.rm = TRUE)
  data$row_sd <- apply(data[num_cols], 1, sd, na.rm = TRUE)
  
  # Add date-based features if date column exists
  date_cols <- names(data)[sapply(data, function(x) inherits(x, c("Date", "POSIXct", "POSIXt")))]
  if (length(date_cols) > 0) {
    for (date_col in date_cols) {
      data[[paste0(date_col, "_year")]] <- as.numeric(format(data[[date_col]], "%Y"))
      data[[paste0(date_col, "_month")]] <- as.numeric(format(data[[date_col]], "%m"))
      data[[paste0(date_col, "_day")]] <- as.numeric(format(data[[date_col]], "%d"))
      data[[paste0(date_col, "_dow")]] <- as.numeric(format(data[[date_col]], "%u"))  # Day of week
      data[[paste0(date_col, "_doy")]] <- as.numeric(format(data[[date_col]], "%j"))  # Day of year
    }
  }
  
  # Save transformed data summary
  log_message("Saving transformed data summary...")
  
  sink(file.path(output_dir, "transformed_data_summary.txt"))
  cat("=== TRANSFORMED DATA SUMMARY ===\n\n")
  cat("1. DIMENSIONS\n")
  cat("Rows:", nrow(data), "\n")
  cat("Columns:", ncol(data), "\n\n")
  
  cat("2. TARGET VARIABLE DISTRIBUTION\n")
  print(table(data$BDH_Class))
  cat("\n")
  
  cat("3. NUMERICAL VARIABLES\n")
  print(summary(data[sapply(data, is.numeric)]))
  cat("\n")
  
  cat("4. CATEGORICAL VARIABLES\n")
  cat_vars <- names(data)[sapply(data, is.factor)]
  for (var in cat_vars) {
    cat("\n", var, ":\n", sep = "")
    print(summary(data[[var]]))
  }
  
  sink()
  
  # Save the processed data
  saveRDS(data, file.path(output_dir, "processed_data.rds"))
  write.csv(data, file.path(output_dir, "processed_data.csv"), row.names = FALSE)
  
  log_message("Data preprocessing completed successfully")
  
  # Define features for modeling
  log_message("Preparing data for modeling...")
  
  # Remove any remaining rows with NA values in the target
  data <- data[!is.na(data$BDH_Class), ]
  
  # Define features - include all numeric columns except the target and its derivatives
  exclude_cols <- c("BDH", "BDH_Class", "row_mean", "row_median", "row_sd")
  numeric_features <- setdiff(
    names(data)[sapply(data, is.numeric)],
    exclude_cols
  )
  
  # Include factor columns with a reasonable number of levels
  factor_features <- names(data)[sapply(data, is.factor) & 
                                names(data) != "BDH_Class" &
                                sapply(data, function(x) length(levels(x))) <= 20]
  
  # Combine all features
  features <- c(numeric_features, factor_features)
  
  log_message(sprintf("Selected %d features for modeling", length(features)))
  features <- features[sapply(features, function(x) sum(!is.na(data[[x]])) > 0)]
  
  cat("\nUsing features:", paste(features, collapse = ", "), "\n")
  
  # Create a simpler recipe without interactions and polynomial terms initially
  # First, select only the columns we need
  model_data <- data %>%
    select(all_of(c("BDH", features))) %>%
    # Remove rows with any NA values for simplicity
    drop_na()
  
  # Create a simple recipe
  recipe_spec <- recipe(BDH ~ ., data = model_data) %>%
    # Remove zero-variance predictors
    step_zv(all_predictors()) %>%
    # Center and scale numeric predictors
    step_center(all_numeric(), -all_outcomes()) %>%
    step_scale(all_numeric(), -all_outcomes())
  
  # Prepare the recipe
  prepped_recipe <- prep(recipe_spec, training = model_data)
  
  # Apply preprocessing
  data_preprocessed <- bake(prepped_recipe, new_data = model_data)
  
  # Split data into training and testing sets
  set.seed(123)
  train_idx <- createDataPartition(data_preprocessed$BDH, p = 0.8, list = FALSE)
  train_data <- data_preprocessed[train_idx, ]
  test_data <- data_preprocessed[-train_idx, ]
  
  # Update features to only include those remaining after preprocessing
  features <- setdiff(names(train_data), "BDH")
  
  cat("Data preprocessing completed.\n")
  
  }, error = function(e) {
    cat("\nError details:")
    cat("\nError message:", e$message)
    cat("\nCall stack:")
    print(sys.calls())
    stop(paste("Error in data loading and preprocessing:", e$message))
  })
}, error = function(e) {
  cat("\nOuter error handler caught:", e$message, "\n")
  stop(e)
})

# 2. Complex Decision Tree Model
cat("\n=== Training Complex Decision Tree Model ===\n")

tryCatch({
  # Function to evaluate model performance
  evaluate_model <- function(predictions, actual, model_name = "Model") {
    cm <- caret::confusionMatrix(predictions, actual)
    
    # Save confusion matrix as plot
    cm_plot <- as.data.frame(cm$table) %>%
      ggplot(aes(Prediction, Reference, fill = Freq)) +
      geom_tile() +
      geom_text(aes(label = Freq), color = "white", size = 6) +
      scale_fill_gradient(low = "#6baed6", high = "#2171b5") +
      labs(title = paste(model_name, "Confusion Matrix"),
           x = "Predicted",
           y = "Actual") +
      theme_minimal() +
      theme(legend.position = "none")
    
    ggsave(file.path(output_dir, paste0(tolower(gsub(" ", "_", model_name)), "_confusion_matrix.png")),
           plot = cm_plot, width = 8, height = 6, dpi = 300)
    
    # Calculate metrics
    metrics <- data.frame(
      Model = model_name,
      Accuracy = cm$overall["Accuracy"],
      Kappa = cm$overall["Kappa"],
      Sensitivity = mean(cm$byClass[, "Sensitivity"]),
      Specificity = mean(cm$byClass[, "Specificity"]),
      F1 = mean(cm$byClass[, "F1"]),
      row.names = NULL
    )
    
    return(list(metrics = metrics, plot = cm_plot))
  }

  # Set up parallel processing
  log_message("Setting up parallel processing...")
  cl <- makePSOCKcluster(detectCores() - 1)
  registerDoParallel(cl)
  
  # Common training control
  ctrl <- trainControl(
    method = "repeatedcv",
    number = 10,
    repeats = 3,
    classProbs = TRUE,
    summaryFunction = multiClassSummary,
    savePredictions = "final",
    verboseIter = TRUE,
    allowParallel = TRUE
  )
  
  # Create a recipe for preprocessing
  log_message("Creating preprocessing recipe...")
  recipe_formula <- as.formula(paste("BDH_Class ~", paste(features, collapse = " + ")))
  
  model_recipe <- recipe(recipe_formula, data = data) %>%
    step_center(all_numeric(), -all_outcomes()) %>%
    step_scale(all_numeric(), -all_outcomes()) %>%
    step_nzv(all_predictors()) %>%
    step_corr(all_numeric(), threshold = 0.9) %>%
    step_naomit(all_predictors()) %>%
    step_dummy(all_nominal(), -all_outcomes())
  
  # Prepare data for modeling
  log_message("Preparing data for modeling...")
  prepped_data <- prep(model_recipe, training = data, retain = TRUE)
  train_data <- bake(prepped_data, new_data = data)
  
  # Split data into training and testing sets
  set.seed(123)
  train_index <- createDataPartition(data$BDH_Class, p = 0.8, list = FALSE)
  train_set <- data[train_index, ]
  test_set <- data[-train_index, ]
  
  # Train multiple models
  models <- list()
  model_metrics <- list()
  
  # 1. Decision Tree (CART)
  log_message("Training Decision Tree model...")
  set.seed(123)
  models$cart <- train(
    model_recipe,
    data = train_set,
    method = "rpart",
    metric = "Accuracy",
    trControl = ctrl,
    tuneLength = 10
      maxcompete = 10,
      maxsurrogate = 10,
      usesurrogate = 2,
      surrogatestyle = 1,
      maxneighbor = 10
    )
  )
  
  # Print model summary
  cat("\nModel trained successfully.\n")
  print(tree_model)
  
  # Make predictions
  predictions <- predict(tree_model, newdata = test_data)
  
  # Calculate metrics
  r2 <- R2(predictions, test_data$BDH)
  rmse <- RMSE(predictions, test_data$BDH)
  mae <- MAE(predictions, test_data$BDH)
  
  cat("\nModel Performance:\n")
  cat(sprintf("R-squared: %.4f\n", r2))
  cat(sprintf("RMSE: %.4f\n", rmse))
  cat(sprintf("MAE: %.4f\n", mae))
  
}, error = function(e) {
  stop(paste("Error in model training:", e$message))
})

# 3. Advanced Visualization
cat("\n=== Generating Advanced Visualizations ===\n")

tryCatch({
  # Create output directory
  if (!dir.exists("output")) dir.create("output")
  if (!dir.exists("output/plots")) dir.create("output/plots")
  
  # 1. Variable Importance Plot
  vip_plot <- vip(tree_model, num_features = 15, geom = "point") +
    theme_minimal() +
    ggtitle("Variable Importance") +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  # 2. Partial Dependence Plots
  pdp_data <- lapply(features[1:6], function(feature) {
    pdp::partial(tree_model, pred.var = feature, train = train_data) %>%
      mutate(Feature = feature)
  }) %>% bind_rows()
  
  pdp_plot <- ggplot(pdp_data, aes_string(x = "yhat", y = names(pdp_data)[1])) +
    geom_line() +
    facet_wrap(~Feature, scales = "free_x") +
    theme_minimal() +
    labs(y = "Partial Dependence", x = "") +
    ggtitle("Partial Dependence Plots") +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  # 3. Tree Visualization
  rpart.plot::rpart.plot(
    tree_model$finalModel,
    type = 5,
    extra = 101,
    box.palette = "RdYlGn",
    branch.lty = 3,
    shadow.col = "gray",
    nn = TRUE,
    roundint = FALSE,
    cex = 0.8,
    main = "Complex Decision Tree Structure"
  )
  
  # 4. Performance Plot
  perf_df <- data.frame(
    Actual = test_data$BDH,
    Predicted = predictions,
    Residuals = test_data$BDH - predictions
  )
  
  perf_plot <- ggplot(perf_df, aes(x = Actual, y = Predicted)) +
    geom_point(alpha = 0.6, color = "steelblue") +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    geom_smooth(method = "lm", se = FALSE, color = "darkgreen") +
    theme_minimal() +
    ggtitle("Actual vs Predicted Values") +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  
  # Save plots
  ggsave("output/plots/variable_importance.png", vip_plot, width = 12, height = 8)
  ggsave("output/plots/partial_dependence.png", pdp_plot, width = 14, height = 10)
  ggsave("output/plots/performance_plot.png", perf_plot, width = 10, height = 8)
  
  # Save the tree plot
  png("output/plots/decision_tree.png", width = 2000, height = 1200, res = 150)
  rpart.plot::rpart.plot(
    tree_model$finalModel,
    type = 5,
    extra = 101,
    box.palette = "RdYlGn",
    branch.lty = 3,
    shadow.col = "gray",
    nn = TRUE,
    roundint = FALSE,
    cex = 0.8,
    main = "Complex Decision Tree Structure"
  )
  dev.off()
  
  cat("Visualizations saved to output/plots/\n")
  
}, error = function(e) {
  warning(paste("Error in visualization:", e$message))
})

# 4. Model Interpretation
cat("\n=== Generating Model Interpretation ===\n")

tryCatch({
  # Create explainer
  explainer <- explain(
    model = tree_model,
    data = test_data[, features],
    y = test_data$BDH,
    label = "Complex Decision Tree"
  )
  
  # Calculate SHAP values
  # Note: This might take a while for large datasets
  if (nrow(test_data) > 1000) {
    test_sample <- test_data[sample(1:nrow(test_data), 1000), ]
  } else {
    test_sample <- test_data
  }
  
  # Generate model explanations
  model_parts <- model_parts(
    explainer,
    type = "variable_importance",
    B = 50
  )
  
  # Save model explanations
  saveRDS(model_parts, "output/model_explanations.rds")
  
  cat("Model interpretation completed. Results saved to output/model_explanations.rds\n")
  
}, error = function(e) {
  warning(paste("Error in model interpretation:", e$message))
})

# 5. Generate HTML Report
cat("\n=== Generating HTML Report ===\n")

  # Generate HTML Report
  log_message("Generating HTML report...")
  
  # Create report directory
  report_dir <- file.path(output_dir, "report")
  if (!dir.exists(report_dir)) {
    dir.create(report_dir, recursive = TRUE)
  }
  
  # Copy necessary files
  file.copy(list.files(output_dir, pattern = "\.(png|csv|rds)$", full.names = TRUE), 
            report_dir, overwrite = TRUE)
  
  # Create R Markdown report
  rmd_content <- '---
title: "Advanced Decision Tree Analysis Report"
author: "Automated Analysis"
date: "`r format(Sys.time(), ''%B %d, %Y''))`"
output:
  html_document:
    toc: true
    toc_float: true
    code_folding: show
    theme: flatly
    highlight: tango
    df_print: paged
    css: styles.css
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE, warning = FALSE, message = FALSE, cache = TRUE)
library(tidyverse)
library(plotly)
library(DT)
library(knitr)
library(kableExtra)
library(rpart.plot)
library(ggparty)
library(patchwork)

# Load data and models
output_dir <- "../"
models <- list()
for (model_file in list.files(output_dir, pattern = "_model\\.rds$")) {
  model_name <- gsub("_model\\.rds$", "", model_file)
  models[[model_name]] <- readRDS(file.path(output_dir, model_file))
}

# Load metrics
metrics <- read.csv(file.path(output_dir, "model_comparison.csv"))

# Set ggplot theme
theme_set(theme_minimal() + 
          theme(plot.title = element_text(face = "bold", size = 16, hjust = 0.5),
                plot.subtitle = element_text(hjust = 0.5),
                legend.position = "bottom"))
```

# Executive Summary

This report presents the results of an advanced decision tree analysis for BDH prediction. Multiple tree-based models were trained and evaluated to predict BDH categories (Low, Medium, High) based on environmental and socio-economic indicators.

```{r load-data, echo=FALSE}
# Load the processed data
data <- readRDS(file.path(output_dir, "processed_data.rds"))

# Summary of the target variable
target_summary <- data %>%
  count(BDH_Class) %>%
  mutate(Percentage = round(n / sum(n) * 100, 1))

# Plot target distribution
ggplot(target_summary, aes(x = BDH_Class, y = n, fill = BDH_Class)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_text(aes(label = paste0(n, " (", Percentage, "%)")), 
            vjust = -0.5, size = 4) +
  scale_fill_brewer(palette = "Set2") +
  labs(title = "Distribution of BDH Classes",
       x = "BDH Class",
       y = "Count") +
  theme(legend.position = "none")
```

# Data Overview

```{r data-overview, echo=FALSE}
# Display data summary
DT::datatable(
  head(data, 100),
  extensions = c('Buttons', 'Scroller'),
  options = list(
    dom = 'Bfrtip',
    buttons = c('copy', 'csv', 'excel', 'pdf'),
    scrollX = TRUE,
    scrollY = "500px",
    scroller = TRUE
  )
)
```

# Model Comparison

```{r model-comparison, echo=FALSE}
# Display metrics in a nice table
metrics_display <- metrics %>%
  mutate(across(where(is.numeric), ~round(., 4))) %>%
  arrange(desc(Accuracy))

kable(metrics_display, "html") %>%
  kable_styling(bootstrap_options = c("striped", "hover", "condensed", "responsive")) %>%
  row_spec(1, bold = TRUE, color = "white", background = "#5cb85c")

# Plot model comparison
ggplot(metrics, aes(x = reorder(Model, Accuracy), y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", alpha = 0.8) +
  geom_text(aes(label = round(Accuracy, 3)), hjust = 1.1, color = "white") +
  coord_flip() +
  scale_fill_brewer(palette = "Set3") +
  labs(title = "Model Accuracy Comparison",
       x = "Model",
       y = "Accuracy") +
  theme(legend.position = "none")
```

# Model Details

## Decision Tree (CART)

```{r cart-details, echo=FALSE}
if (!is.null(models$cart)) {
  # Plot decision tree
  rpart.plot(models$cart$finalModel, 
             box.palette = "RdBu",
             shadow.col = "gray",
             nn = TRUE,
             main = "Decision Tree (CART)")
  
  # Show variable importance
  var_imp <- varImp(models$cart, scale = FALSE)
  ggplot(var_imp, top = 10) +
    theme_minimal() +
    labs(title = "CART - Top 10 Important Variables")
}
```

## Random Forest

```{r rf-details, echo=FALSE}
if (!is.null(models$rf)) {
  # Plot variable importance
  var_imp <- varImp(models$rf, scale = FALSE)
  ggplot(var_imp, top = 15) +
    theme_minimal() +
    labs(title = "Random Forest - Top 15 Important Variables")
  
  # Partial dependence plot for top variables
  tryCatch({
    top_vars <- rownames(var_imp$importance)[1:3]
    pd_plots <- list()
    
    for (i in seq_along(top_vars)) {
      pd <- partial(models$rf, pred.var = top_vars[i], 
                   train = bake(prep(model_recipe, training = data), new_data = data))
      pd_plots[[i]] <- ggplot(pd, aes_string(x = top_vars[i], y = "yhat")) +
        geom_line() +
        labs(title = paste("Partial Dependence on", top_vars[i]))
    }
    
    wrap_plots(pd_plots, ncol = 1)
  }, error = function(e) {
    cat("Could not generate partial dependence plots")
  })
}
```

# Conclusion

Based on the analysis, the `r metrics$Model[which.max(metrics$Accuracy)]` model achieved the highest accuracy of `r round(max(metrics$Accuracy) * 100, 2)`% in predicting BDH categories. The most important features identified across models were `r paste(rownames(var_imp$importance)[1:3], collapse = ", ")`.

## Recommendations

1. Consider collecting more data to improve model performance, especially for the minority classes.
2. Further investigate the most important features to understand their relationship with BDH.
3. Deploy the best performing model for real-time predictions with appropriate monitoring.

---

*This report was automatically generated on `r Sys.time()`*'

  # Write R Markdown file
  rmd_file <- file.path(report_dir, "analysis_report.Rmd")
  writeLines(rmd_content, rmd_file)
  
  # Create CSS file for styling
  css_content <- 'body {
    font-family: "Arial", sans-serif;
    line-height: 1.6;
    color: #333;
  }
  
  h1, h2, h3 {
    color: #2c3e50;
    margin-top: 24px;
    margin-bottom: 16px;
  }
  
  .main-container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
  }
  
  .figure {
    margin: 20px 0;
  }
  
  .dataTables_wrapper {
    margin: 20px 0;
  }'
  
  writeLines(css_content, file.path(report_dir, "styles.css"))
  
  # Render the report
  log_message("Rendering HTML report...")
  rmarkdown::render(
    input = rmd_file,
    output_file = "index.html",
    output_dir = report_dir,
    quiet = TRUE
  )
  
  log_message(paste("Report generated:", file.path(report_dir, "index.html")))
  
  # Create a simple HTML report as fallback
  html_content <- sprintf('<!DOCTYPE html>
  <html>
  <head>
    <title>Advanced Decision Tree Analysis Report</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
      body { font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 20px; color: #333; }
      .container { max-width: 1200px; margin: 0 auto; }
      h1 { color: #2c3e50; text-align: center; }
      .section { margin: 30px 0; padding: 20px; background: #f9f9f9; border-radius: 5px; }
      .model { margin: 20px 0; padding: 15px; background: white; border-left: 4px solid #3498db; }
      .success { color: #27ae60; font-weight: bold; }
      .warning { color: #f39c12; }
      .error { color: #e74c3c; }
      img { max-width: 100%; height: auto; display: block; margin: 10px 0; }
      table { width: 100%; border-collapse: collapse; margin: 15px 0; }
      th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
      th { background-color: #f2f2f2; }
      tr:hover { background-color: #f5f5f5; }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>Advanced Decision Tree Analysis Report</h1>
      <p>Generated on: %s</p>
      
      <div class="section">
        <h2>Executive Summary</h2>
        <p>This report presents the results of an advanced decision tree analysis for BDH prediction. Multiple tree-based models were trained and evaluated to predict BDH categories (Low, Medium, High) based on environmental and socio-economic indicators.</p>
        
        <h3>Best Performing Model: %s</h3>
        <p>Accuracy: <span class="success">%.2f%%</span></p>
      </div>
      
      <div class="section">
        <h2>Model Comparison</h2>
        <img src="model_comparison.png" alt="Model Comparison">
      </div>
      
      <div class="section">
        <h2>Detailed Model Performance</h2>', 
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    metrics$Model[which.max(metrics$Accuracy)],
    max(metrics$Accuracy) * 100
  )
  
  # Add model details
  for (i in 1:nrow(metrics)) {
    model_name <- metrics$Model[i]
    html_content <- paste0(html_content, sprintf('
        <div class="model">
          <h3>%s</h3>
          <p>Accuracy: %.4f | Kappa: %.4f | F1: %.4f</p>
          <img src="%s_confusion_matrix.png" alt="%s Confusion Matrix">
          <img src="%s_variable_importance.png" alt="%s Variable Importance">
        </div>',
      model_name, 
      metrics$Accuracy[i],
      metrics$Kappa[i],
      metrics$F1[i],
      tolower(gsub(" ", "_", model_name)),
      model_name,
      tolower(gsub(" ", "_", model_name)),
      model_name
    ))
  }
  
  # Close HTML
  html_content <- paste0(html_content, '
      </div>
      
      <div class="section">
        <h2>Conclusion</h2>
        <p>Based on the analysis, the <strong>', metrics$Model[which.max(metrics$Accuracy)], 
    '</strong> model achieved the highest accuracy of <strong>', 
    round(max(metrics$Accuracy) * 100, 2), 
    '%</strong> in predicting BDH categories.</p>
        
        <h3>Recommendations</h3>
        <ul>
          <li>Consider collecting more data to improve model performance, especially for the minority classes.</li>
          <li>Further investigate the most important features to understand their relationship with BDH.</li>
          <li>Deploy the best performing model for real-time predictions with appropriate monitoring.</li>
        </ul>
      </div>
    </div>
  </body>
  </html>')
  
  # Write HTML file
  html_file <- file.path(output_dir, "simple_report.html")
  writeLines(html_content, html_file)
  
  log_message(paste("Simple report generated:", html_file))
  log_message("Analysis completed successfully!")
  
  # Open the report in default browser
  utils::browseURL(file.path(normalizePath(report_dir), "index.html"))
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }
      .container { max-width: 1200px; margin: 0 auto; }
      .header { text-align: center; margin-bottom: 30px; }
      .section { margin: 30px 0; }
      .plot-container { margin: 20px 0; text-align: center; }
      .plot-img { max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }
      .metrics { 
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 5px;
        margin: 20px 0;
      }
      .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
        gap: 20px;
        margin: 20px 0;
      }
      @media (max-width: 768px) {
        .grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div class="container">
      <div class="header">
        <h1>Advanced Decision Tree Analysis</h1>
        <p>Comprehensive analysis of BDH scores using complex decision tree models</p>
        <p>Generated on: %s</p>
      </div>
      
      <div class="section">
        <h2>Model Performance</h2>
        <div class="metrics">
          <h3>Regression Metrics</h3>
          <p><strong>R-squared:</strong> %.4f (Higher is better, max 1.0)</p>
          <p><strong>RMSE:</strong> %.4f (Lower is better)</p>
          <p><strong>MAE:</strong> %.4f (Lower is better)</p>
        </div>
      </div>
      
      <div class="section">
        <h2>Model Visualizations</h2>
        <div class="grid">
          <div class="plot-container">
            <h3>Decision Tree Structure</h3>
            <img src="plots/decision_tree.png" alt="Decision Tree" class="plot-img">
          </div>
          <div class="plot-container">
            <h3>Variable Importance</h3>
            <img src="plots/variable_importance.png" alt="Variable Importance" class="plot-img">
          </div>
          <div class="plot-container">
            <h3>Actual vs Predicted Values</h3>
            <img src="plots/performance_plot.png" alt="Performance Plot" class="plot-img">
          </div>
          <div class="plot-container">
            <h3>Partial Dependence Plots</h3>
            <img src="plots/partial_dependence.png" alt="Partial Dependence" class="plot-img">
          </div>
        </div>
      </div>
      
      <div class="section">
        <h2>Model Details</h2>
        <p>This complex decision tree model was trained with the following parameters:</p>
        <ul>
          <li>Max Depth: %d</li>
          <li>Min Bucket: %d</li>
          <li>Complexity Parameter: %.4f</li>
          <li>10-fold Cross-Validation with 5 repeats</li>
        </ul>
      </div>
    </div>
  </body>
  </html>',
    format(Sys.time(), "%Y-%m-%d %H:%M:%S"),
    r2,
    rmse,
    mae,
    tree_model$bestTune$maxdepth,
    tree_model$bestTune$minbucket,
    tree_model$bestTune$cp
  )
  
  writeLines(html_content, "output/advanced_decision_tree_report.html")
  
  cat("HTML report generated: output/advanced_decision_tree_report.html\n")
  
}, error = function(e) {
  warning(paste("Error generating HTML report:", e$message))
})

cat("\n=== Analysis Complete ===\n")
cat("Check the output directory for results and visualizations.\n")
