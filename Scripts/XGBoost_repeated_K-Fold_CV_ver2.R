#-------------------------------------------------------------------------------
#               XGBoost with Repeated K-Fold Cross-Validation                  #
#                                     Ver2                                     #
#-------------------------------------------------------------------------------

# Clear the workspace and close all graphics devices
rm(list = ls())
if (!is.null(dev.list())) dev.off()
gc()  # Garbage collection to free up memory

#-------------------------------------------------------------------------------
# Load necessary libraries
library(readxl)       # For reading Excel files
library(dplyr)        # For data manipulation
library(tidyr)
library(ggplot2)      # For plotting
library(viridis)      # For color scales
library(xgboost)      # For XGBoost model
library(caret)        # For data splitting and cross-validation
library(Metrics)      # For calculating performance metrics
library(ModelMetrics) # For R-squared calculation
library(doFuture)     # For parallel processing with future backend
library(doRNG)        # For reproducible parallel random number generation
library(foreach)      # For parallel loops
library(progressr)    # For progress bar
library(lubridate)    # For date manipulation

#-------------------------------------------------------------------------------
# Set up future plan for parallel processing
library(future)
cores <- availableCores() - 2  # Use two less than the number of available cores
plan(multisession, workers = cores)
registerDoFuture()

#-------------------------------------------------------------------------------
# Set up progress handler
handlers(global = TRUE)
handlers("txtprogressbar")  # You can choose different handlers

#-------------------------------------------------------------------------------
# Load the data
data <- read_excel("/Volumes/T7/data_for_machine_learning.xlsx")

# Preview data structure
str(data)

#-------------------------------------------------------------------------------
# Define predictors and target variable

predictors <- c("log(rrs443)", "log(rrs490)", "log(rrs510)", "log(rrs560)" , "DOY")
target_variable <- "log(chl)"

# Select relevant columns
data <- data %>% select(all_of(c(predictors, target_variable)))

# Check for missing values
#if (sum(is.na(data)) > 0) {
# Remove rows with missing values
#  data <- data %>% drop_na()
#}

#-------------------------------------------------------------------------------
# Split data into training and test sets (70/30 split)
set.seed(42)  # For reproducibility
train_index <- createDataPartition(data[[target_variable]], p = 0.70, list = FALSE)
train_data <- data[train_index, ]
test_data  <- data[-train_index, ]

# Separate features and target for train and test sets
train_x <- train_data %>% select(-all_of(target_variable))
train_y <- train_data[[target_variable]]
test_x  <- test_data %>% select(-all_of(target_variable))
test_y  <- test_data[[target_variable]]

# Convert data to matrices
train_matrix <- as.matrix(train_x)
test_matrix  <- as.matrix(test_x)

#-------------------------------------------------------------------------------
# Hyper-parameter tuning using Repeated K-Fold Cross-Validation

# Define the parameter grid
param_grid <- expand.grid(
  max_depth         = c(3, 4, 5, 6),
  eta               = c(0.01, 0.1),
  gamma             = c(0.1, 0.2, 0.3, 0.4, 0.6, 0.8, 1),
  colsample_bytree  = c(0.7, 0.8, 0.9),
  min_child_weight  = c(1, 3, 5, 7, 8, 9),
  subsample         = c(0.4, 0.5, 0.8, 0.9),
  lambda            = c(0),           # L2 regularization term
  alpha             = c(0)            # L1 regularization term
)

# Set number of rounds and early stopping
nrounds               <- 500
early_stopping_rounds <- 10
k_folds               <- 10   # Number of folds for cross-validation
n_repeats             <- 5    # Number of repeats for repeated k-fold CV

# Initialize variables to store the best results
best_rmse    <- Inf
best_params  <- list()
best_nrounds <- 0

# Convert param_grid to a list of parameter combinations
param_list <- split(param_grid, seq(nrow(param_grid)))
total_tasks <- length(param_list) * n_repeats  # Total number of tasks including repeats

#-------------------------------------------------------------------------------
# Perform cross-validation for each combination of hyperparameters
# Use doRNG to ensure reproducible results in parallel

with_progress({
  p <- progressor(steps = total_tasks)
  
  results <- foreach(i = seq_along(param_list), .combine = rbind) %dorng% {
    # Access the data within the worker
    train_matrix_worker <- train_matrix
    train_y_worker      <- train_y
    
    # Create DMatrix within the worker
    dtrain_worker <- xgb.DMatrix(data = train_matrix_worker, label = train_y_worker)
    
    params <- as.list(param_list[[i]])
    params$booster     <- "gbtree"
    params$objective   <- "reg:squarederror"
    params$eval_metric <- "rmse"
    params$nthread     <- 1  # Use one thread per worker to avoid over-subscription
    
    # Initialize vector to store RMSE for each repeat
    rmse_repeats <- c()
    nrounds_repeats <- c()
    
    for (repeat_idx in 1:n_repeats) {
      # Perform k-fold cross-validation
      cv <- xgb.cv(
        params                = params,
        data                  = dtrain_worker,
        nrounds               = nrounds,
        nfold                 = k_folds,
        early_stopping_rounds = early_stopping_rounds,
        verbose               = 0,
        maximize              = FALSE,
        showsd                = TRUE,
        stratified            = TRUE
      )
      
      # Store the best RMSE and number of rounds
      rmse_repeats <- c(rmse_repeats, min(cv$evaluation_log$test_rmse_mean))
      nrounds_repeats <- c(nrounds_repeats, cv$best_iteration)
      
      # Call garbage collector to free up memory
      rm(cv)
      gc()
      
      # Update progress bar after each repeat
      p(message = sprintf("Param combo %d, repeat %d", i, repeat_idx))
    }
    
    # Calculate the average RMSE and average number of rounds over repeats
    mean_rmse      <- mean(rmse_repeats)
    mean_nrounds   <- round(mean(nrounds_repeats))
    
    # Return the results as a data frame row
    result <- c(mean_rmse, mean_nrounds, unlist(params))
    names(result) <- c("mean_rmse", "mean_nrounds", names(params))
    return(result)
  }
})

# Convert results to a data frame
results_df <- as.data.frame(results)
results_df$mean_rmse    <- as.numeric(as.character(results_df$mean_rmse))
results_df$mean_nrounds <- as.numeric(as.character(results_df$mean_nrounds))

# Find the best hyperparameters based on average RMSE
best_row <- results_df[which.min(results_df$mean_rmse), ]
best_params <- as.list(best_row[3:ncol(best_row)])
best_nrounds <- best_row$mean_nrounds

# Print best parameters
cat("Best average RMSE from repeated cross-validation:", best_row$mean_rmse, "\n")
cat("Best Parameters:\n")
print(best_params)
cat("Best average number of rounds (nrounds):", best_nrounds, "\n")

#-------------------------------------------------------------------------------
# Train the final model using the best parameters
set.seed(42)  # For reproducibility

# Remove parameters that are not needed in xgb.train
best_params$nthread <- cores  # Use multiple cores for final training
best_params$mean_rmse <- NULL
best_params$mean_nrounds <- NULL

# Prepare DMatrix for final model
dtrain <- xgb.DMatrix(data = train_matrix, label = train_y)
dtest  <- xgb.DMatrix(data = test_matrix, label = test_y)

final_model <- xgb.train(
  params                = best_params,
  data                  = dtrain,
  nrounds               = best_nrounds,
  watchlist             = list(train = dtrain, eval = dtest),
  early_stopping_rounds = early_stopping_rounds,
  verbose               = 0
)

#-------------------------------------------------------------------------------
# Evaluate the model

# Training set predictions and metrics
train_predictions <- predict(final_model, newdata = dtrain)
train_rmse        <- rmse(train_y, train_predictions)
train_mae <- mae(10^(train_y), 10^(train_predictions))  # MAE on no-log scale
train_r2          <- R2(train_predictions, train_y)
cat("Training RMSE:", train_rmse, "\n")
cat("Training MAE:", train_mae, "\n")
cat("Training R-squared:", train_r2, "\n")

# Test set predictions and metrics
test_predictions <- predict(final_model, newdata = dtest)
test_rmse        <- rmse(test_y, test_predictions)
test_mae <- mae(10^(test_y), 10^(test_predictions))  # MAE on no-log scale
test_r2          <- R2(test_predictions, test_y)
cat("Test RMSE:", test_rmse, "\n")
cat("Test MAE:", test_mae, "\n")
cat("Test R-squared:", test_r2, "\n")

#-------------------------------------------------------------------------------
# Clean up future plan
plan(sequential)

#-------------------------------------------------------------------------------
# Plotting Section # 

# Calculate residuals and standard deviation for Training set
train_residuals <- train_y - train_predictions
train_residual_sd <- sd(train_residuals)

# Prepare data frame for plotting
train_pred_vs_actual <- data.frame(
  Actual = train_y,
  Predicted = train_predictions
)

# 1. Plot Predicted vs. Actual for Training Set with your customized code
ggplot(train_pred_vs_actual, aes(x = Actual, y = Predicted)) +
  geom_point(color = "#2C7BB6", alpha = 0.6) +  # Scatter points
geom_smooth(method = "lm", color = "#00008B", linetype = "solid", se = FALSE, linewidth = 1) +  # Linear fit line
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#006400", linewidth = 1) +  # 1:1 line
  geom_abline(slope = 1, intercept = train_residual_sd, linetype = "dotted", color = "#696969", linewidth = 1) +  # +1 SD line
  geom_abline(slope = 1, intercept = -train_residual_sd, linetype = "dotted", color = "#696969", linewidth = 1) +  # -1 SD line
  labs(
    title = "Predicted vs. Actual (Training Set)",
    x = "Actual Shannon Diversity (Training)",
    y = "Predicted Shannon Diversity (Training)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5),
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)
  ) +
  annotate("text", x = min(train_pred_vs_actual$Actual), y = max(train_pred_vs_actual$Predicted), 
           label = paste0("R² = ", round(train_r2, 2), 
                          "\nRMSE = ", round(train_rmse, 2), 
                          "\nMAE = ", round(train_mae, 2),
                          "\nN = ", nrow(train_pred_vs_actual)),
           hjust = 0, vjust = 1, size = 5, color = "black") +
  coord_fixed(ratio = 1, 
              xlim = range(train_pred_vs_actual$Actual, train_pred_vs_actual$Predicted),
              ylim = range(train_pred_vs_actual$Actual, train_pred_vs_actual$Predicted))

# Calculate residuals and standard deviation for Test set
test_residuals <- test_y - test_predictions
test_residual_sd <- sd(test_residuals)

# Prepare data frame for plotting
test_pred_vs_actual <- data.frame(
  Actual = test_y,
  Predicted = test_predictions
)

# 2. Plot Predicted vs. Actual for Test Set with your customized code
ggplot(test_pred_vs_actual, aes(x = Actual, y = Predicted)) +
  geom_point(color = "#D7191C", alpha = 0.6) +  # Scatter points
  geom_smooth(method = "lm", color = "#8B0000", linetype = "solid", se = FALSE, linewidth = 1) +  # Linear fit line
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "#006400", linewidth = 1) +  # 1:1 line
  geom_abline(slope = 1, intercept = test_residual_sd, linetype = "dotted", color = "#696969", linewidth = 1) +  # +1 SD line
  geom_abline(slope = 1, intercept = -test_residual_sd, linetype = "dotted", color = "#696969", linewidth = 1) +  # -1 SD line
  labs(
    title = "Predicted vs. Actual (Test Set)",
    x = "Actual Shannon Diversity (Test)",
    y = "Predicted Shannon Diversity (Test)"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5),
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)
  ) +
  annotate("text", x = min(test_pred_vs_actual$Actual), y = max(test_pred_vs_actual$Predicted), 
           label = paste0("R² = ", round(test_r2, 2), 
                          "\nRMSE = ", round(test_rmse, 2), 
                          "\nMAE = ", round(test_mae, 2),
                          "\nN = ", nrow(test_pred_vs_actual)),
           hjust = 0, vjust = 1, size = 5, color = "black") +
  coord_fixed(ratio = 1, 
              xlim = range(test_pred_vs_actual$Actual, test_pred_vs_actual$Predicted),
              ylim = range(test_pred_vs_actual$Actual, test_pred_vs_actual$Predicted))

# 3. Feature Importance Plot using ggplot2 and a Discrete Palette
importance_matrix <- xgb.importance(feature_names = colnames(train_matrix), model = final_model)
importance_df <- importance_matrix %>%
  arrange(desc(Gain)) %>%
  mutate(Feature = factor(Feature, levels = rev(Feature)))  # For ordered plotting

ggplot(importance_df, aes(x = Feature, y = Gain, fill = Feature)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_viridis_d(option = "D") +  # Discrete palette
  labs(
    title = "Feature Importance",
    x = "Features",
    y = "Gain"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    legend.position = "none",
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 1),
    plot.title = element_text(hjust = 0.5)
  )

# 4. Density Plot of Residuals for Training and Testing Sets with Vertical Line at Zero
residuals_data <- data.frame(
  Residuals = c(train_residuals, test_residuals),
  Dataset = factor(c(rep("Training", length(train_residuals)), rep("Testing", length(test_residuals))))
)

ggplot(residuals_data, aes(x = Residuals, fill = Dataset)) +
  geom_density(alpha = 0.6) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black", linewidth = 1) +  # Vertical dashed line at zero
  scale_fill_viridis_d() +
  labs(
    title = "Density Plot of Residuals",
    x = "Residuals",
    y = "Density"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    plot.title = element_text(hjust = 0.5),
    panel.border = element_rect(colour = "black", fill = NA, linewidth = 1)
  )

