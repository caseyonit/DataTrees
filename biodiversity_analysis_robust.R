# Biodiversity & Habitat (BDH) Analysis with Machine Learning
# Robust version with error handling and improved reliability

# Function to safely load packages
safe_require <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cran.rstudio.com/")
    library(pkg, character.only = TRUE)
  } else {
    library(pkg, character.only = TRUE)
  }
}

# Load required packages with error handling
required_pkgs <- c(
  "tidyverse",    # Data manipulation
  "randomForest", # Random Forest
  "caret",        # Model training and evaluation
  "xgboost",      # Gradient Boosting
  "ggpubr",       # Publication-ready plots
  "DT"            # Interactive tables
)

suppressMessages({
  for (pkg in required_pkgs) {
    tryCatch({
      safe_require(pkg)
    }, error = function(e) {
      stop(paste("Failed to load package:", pkg, "\nError:", e$message))
    })
  }
})

# Set seed for reproducibility
set.seed(123)

# Function to safely read data
safe_read_data <- function(file_path) {
  if (!file.exists(file_path)) {
    stop(paste("Data file not found at:", file_path))
  }
  
  tryCatch({
    read.csv(file_path, stringsAsFactors = FALSE)
  }, error = function(e) {
    stop(paste("Error reading data file:", e$message))
  })
}

# Function to preprocess data
preprocess_data <- function(data) {
  tryCatch({
    # Create BDH categories for classification
    data <- data %>%
      mutate(
        BDH_Class = case_when(
          BDH < 40 ~ "Low",
          BDH >= 40 & BDH <= 60 ~ "Medium",
          BDH > 60 ~ "High",
          TRUE ~ NA_character_
        ),
        BDH_Class = factor(BDH_Class, levels = c("Low", "Medium", "High"))
      )
    
    # Remove rows with missing BDH values
    data <- data %>%
      filter(!is.na(BDH))
    
    # Select relevant features
    features <- c("EPI", "ECS", "FSH", "APO", "AGR", "WRS", "AIR", "H2O", "HMT", "WMG", "CCH")
    
    # Impute missing values with median
    impute_median <- function(x) {
      x[is.na(x)] <- median(x, na.rm = TRUE)
      x
    }
    
    data <- data %>%
      mutate(across(all_of(features), impute_median))
    
    return(list(data = data, features = features))
    
  }, error = function(e) {
    stop(paste("Error in data preprocessing:", e$message))
  })
}

# Function to train regression model
train_regression <- function(train_data, features) {
  cat("\n=== Training Regression Model (Random Forest) ===\n")
  
  reg_formula <- as.formula(paste("BDH ~", paste(features, collapse = " + ")))
  
  # Create separate control for regression (no class probabilities needed)
  reg_ctrl <- trainControl(
    method = "cv",
    number = 10,
    savePredictions = TRUE,
    classProbs = FALSE  # Disable class probabilities for regression
  )
  
  tryCatch({
    model <- train(
      reg_formula,
      data = train_data,
      method = "rf",
      trControl = reg_ctrl,
      tuneLength = 3,
      importance = TRUE
    )
    return(model)
    
  }, error = function(e) {
    stop(paste("Error in regression model training:", e$message))
  })
}

# Function to train classification model
train_classification <- function(train_data, features, ctrl) {
  cat("\n=== Training Classification Model (XGBoost) ===\n")
  
  class_formula <- as.formula(paste("BDH_Class ~", paste(features, collapse = " + ")))
  
  xgb_grid <- expand.grid(
    nrounds = 100,
    max_depth = 4,
    eta = 0.1,
    gamma = 0,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    subsample = 0.8
  )
  
  tryCatch({
    model <- train(
      class_formula,
      data = train_data,
      method = "xgbTree",
      trControl = ctrl,
      tuneGrid = xgb_grid,
      verbose = FALSE
    )
    return(model)
    
  }, error = function(e) {
    stop(paste("Error in classification model training:", e$message))
  })
}

# Function to evaluate model
evaluate_model <- function(model, test_data, model_type = "regression") {
  tryCatch({
    if (model_type == "regression") {
      # For regression, just get predictions and calculate metrics
      predictions <- predict(model, newdata = test_data)
      r2 <- R2(predictions, test_data$BDH)
      rmse <- RMSE(predictions, test_data$BDH)
      return(list(predictions = predictions, r2 = r2, rmse = rmse))
      
    } else {
      # For classification, get both class predictions and probabilities
      predictions <- predict(model, newdata = test_data)
      probs <- predict(model, newdata = test_data, type = "prob")
      accuracy <- mean(predictions == test_data$BDH_Class, na.rm = TRUE)
      
      # Only calculate confusion matrix and related metrics if we have predictions
      if (length(predictions) > 0) {
        conf_matrix <- confusionMatrix(predictions, test_data$BDH_Class)
        
        precision <- conf_matrix$byClass[, "Pos Pred Value"]
        recall <- conf_matrix$byClass[, "Sensitivity"]
        f1 <- 2 * (precision * recall) / (precision + recall)
        
        return(list(
          predictions = predictions,
          probs = probs,
          accuracy = accuracy,
          precision = precision,
          recall = recall,
          f1 = f1,
          conf_matrix = conf_matrix
        ))
      } else {
        return(list(
          predictions = predictions,
          probs = probs,
          accuracy = accuracy
        ))
      }
    }
  }, error = function(e) {
    warning(paste("Warning in model evaluation:", e$message))
    return(NULL)
  })
}

# Function to create importance plot
create_importance_plot <- function(importance_data, title) {
  tryCatch({
    imp_df <- as.data.frame(importance_data$importance)
    imp_df$feature <- rownames(imp_df)
    imp_df <- imp_df %>% 
      arrange(desc(Overall)) %>%
      head(10)
    
    p <- ggplot(imp_df, aes(x = reorder(feature, Overall), y = Overall)) +
      geom_col(fill = "steelblue") +
      coord_flip() +
      labs(title = title, x = "Feature", y = "Importance") +
      theme_minimal()
    
    return(p)
  }, error = function(e) {
    warning(paste("Failed to create importance plot:", e$message))
    return(NULL)
  })
}

# Function to create performance plot
create_performance_plot <- function(actual, predicted, title) {
  tryCatch({
    df <- data.frame(Actual = actual, Predicted = predicted)
    p <- ggplot(df, aes(x = Actual, y = Predicted)) +
      geom_point(alpha = 0.6) +
      geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
      labs(title = title) +
      theme_minimal()
    
    return(p)
  }, error = function(e) {
    warning(paste("Failed to create performance plot:", e$message))
    return(NULL)
  })
}

# Function to generate HTML report
generate_html_report <- function(reg_results, class_results, output_dir) {
  tryCatch({
    # Format numbers for display
    format_num <- function(x, digits = 3) {
      if (is.numeric(x)) {
        return(round(x, digits))
      }
      return(x)
    }
    
    # Create HTML content
    html_content <- paste0('<!DOCTYPE html>
    <html>
    <head>
      <title>BDH Analysis Report</title>
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
          max-width: 100%;
          height: auto;
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
          flex: 50%;
          padding: 0 10px;
          box-sizing: border-box;
        }
        @media (max-width: 768px) {
          .column { flex: 100%; }
        }
      </style>
    </head>
    <body>
      <div class="container">
        <h1>Biodiversity & Habitat (BDH) Analysis</h1>
        <p>Analysis completed on: ', format(Sys.time(), "%Y-%m-%d %H:%M:%S"), '</p>
        
        <div class="metrics">
          <h2>Regression Model (Random Forest)</h2>
          <p><strong>R-squared:</strong> ', format_num(reg_results$r2), ' (Higher is better, max 1.0)</p>
          <p><strong>RMSE:</strong> ', format_num(reg_results$rmse), ' (Lower is better)</p>',
          if (!is.null(reg_results$perf_plot)) {
            paste0('\n          <img src="', file.path("plots", basename(reg_results$perf_plot)), '" alt="Regression Performance" class="plot-image">')
          },
          '
          <h2>Classification Model (XGBoost)</h2>
          <p><strong>Accuracy:</strong> ', format_num(class_results$accuracy * 100, 1), '%</p>
          <p><strong>Precision (Weighted):</strong> ', format_num(mean(class_results$precision, na.rm = TRUE) * 100, 1), '%</p>
          <p><strong>Recall (Weighted):</strong> ', format_num(mean(class_results$recall, na.rm = TRUE) * 100, 1), '%</p>
          <p><strong>F1 Score (Weighted):</strong> ', format_num(mean(class_results$f1, na.rm = TRUE) * 100, 1), '%</p>',
          '
          <h3>Feature Importance</h3>
          <div class="row">',
          if (!is.null(reg_results$imp_plot)) {
            paste0('\n            <div class="column">
              <h4>Regression</h4>
              <img src="', file.path("plots", basename(reg_results$imp_plot)), '" alt="Feature Importance (Regression)" class="plot-image">
            </div>')
          },
          if (!is.null(class_results$imp_plot)) {
            paste0('\n            <div class="column">
              <h4>Classification</h4>
              <img src="', file.path("plots", basename(class_results$imp_plot)), '" alt="Feature Importance (Classification)" class="plot-image">
            </div>')
          },
          '\n          </div>
        </div>
      </div>
    </body>
    </html>')
    
    # Ensure the output directory exists
    if (!dir.exists(output_dir)) {
      dir.create(output_dir, recursive = TRUE)
    }
    
    # Write HTML file
    output_file <- file.path(output_dir, "biodiversity_analysis_report.html")
    writeLines(html_content, output_file)
    
    return(output_file)
    
  }, error = function(e) {
    stop(paste("Error generating HTML report:", e$message))
  })
}

# Main execution
main <- function() {
  tryCatch({
    cat("=== Starting BDH Analysis ===\n")
    
    # Create output directories
    output_dir <- "output"
    plots_dir <- file.path(output_dir, "plots")
    dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)
    dir.create(plots_dir, showWarnings = FALSE, recursive = TRUE)
    
    # Load and preprocess data
    cat("\n1. Loading and preprocessing data...\n")
    data <- safe_read_data("Data/epi2024_data.csv")
    processed <- preprocess_data(data)
    data_clean <- processed$data
    features <- processed$features
    
    # Set up cross-validation
    ctrl <- trainControl(
      method = "cv",
      number = 10,
      savePredictions = TRUE,
      classProbs = TRUE
    )
    
    # Split data
    set.seed(123)
    train_idx <- createDataPartition(data_clean$BDH, p = 0.8, list = FALSE)
    train_data <- data_clean[train_idx, ]
    test_data <- data_clean[-train_idx, ]
    
    # Train models
    rf_model <- train_regression(train_data, features)
    xgb_model <- train_classification(train_data, features, ctrl)
    
    # Evaluate models
    reg_results <- evaluate_model(rf_model, test_data, "regression")
    class_results <- evaluate_model(xgb_model, test_data, "classification")
    
    # Create visualizations
    cat("\n3. Generating visualizations...\n")
    
    # Create plots directory if it doesn't exist
    if (!dir.exists(plots_dir)) {
      dir.create(plots_dir, recursive = TRUE)
    }
    
    # Save importance plots
    reg_imp_plot <- file.path(plots_dir, "feature_importance_regression.png")
    class_imp_plot <- file.path(plots_dir, "feature_importance_classification.png")
    perf_plot <- file.path(plots_dir, "regression_performance.png")
    
    # Create and save importance plots
    imp_plot_reg <- create_importance_plot(varImp(rf_model), "Feature Importance (Regression)")
    if (!is.null(imp_plot_reg)) {
      ggsave(reg_imp_plot, imp_plot_reg, width = 8, height = 6)
      reg_results$imp_plot <- reg_imp_plot
    }
    
    imp_plot_class <- create_importance_plot(varImp(xgb_model), "Feature Importance (Classification)")
    if (!is.null(imp_plot_class)) {
      ggsave(class_imp_plot, imp_plot_class, width = 8, height = 6)
      class_results$imp_plot <- class_imp_plot
    }
    
    # Create and save performance plot
    perf_plot_obj <- create_performance_plot(
      test_data$BDH, 
      reg_results$predictions, 
      "Regression: Actual vs Predicted BDH"
    )
    
    if (!is.null(perf_plot_obj)) {
      ggsave(perf_plot, perf_plot_obj, width = 8, height = 6)
      reg_results$perf_plot <- perf_plot
    }
    
    # Generate HTML report
    cat("\n4. Generating HTML report...\n")
    report_file <- generate_html_report(reg_results, class_results, output_dir)
    
    # Save models
    saveRDS(rf_model, file.path(output_dir, "random_forest_model.rds"))
    saveRDS(xgb_model, file.path(output_dir, "xgboost_model.rds"))
    
    # Print completion message
    cat("\n=== Analysis Complete ===\n")
    cat("\nRegression (Test Set):")
    cat(sprintf("\n- R-squared: %0.3f", reg_results$r2))
    cat(sprintf("\n- RMSE: %0.2f", reg_results$rmse))
    
    cat("\n\nClassification (Test Set):")
    cat(sprintf("\n- Accuracy: %0.1f%%", class_results$accuracy * 100))
    cat(sprintf("\n- Precision (Weighted): %0.1f%%", mean(class_results$precision, na.rm = TRUE) * 100))
    cat(sprintf("\n- Recall (Weighted): %0.1f%%", mean(class_results$recall, na.rm = TRUE) * 100))
    cat(sprintf("\n- F1 Score (Weighted): %0.1f%%", mean(class_results$f1, na.rm = TRUE) * 100))
    
    cat("\n\nReport generated:", normalizePath(report_file), "\n")
    
  }, error = function(e) {
    cat("\n!!! ERROR: ", e$message, "\n")
    traceback()
    return(invisible(NULL))
  })
}

# Run the analysis
main()
