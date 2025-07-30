# Scenario 4: Peak Performance: Competitive Classification with Gradient Boosting

## Learning Objectives
*   Understand the core concept of "boosting" in ensemble learning.
*   Implement a high-performance gradient boosting model using the XGBoost library.
*   Learn about key hyperparameters in boosting models (e.g., `n_estimators`, `learning_rate`, `max_depth`).
*   Evaluate the model using the ROC AUC score, a robust metric for imbalanced classes.
*   Plot an ROC curve to visualize the classifier's performance.

## Target Audience
Intermediate to Advanced

## Tools/Libraries
`pandas`, `scikit-learn`, `xgboost`, `matplotlib`

## Core Concepts & Methodology
*   **Boosting:** An ensemble technique where models are built sequentially. Each new model attempts to correct the errors made by the previous ones. Instead of independent trees (like in Random Forest), boosting builds a chain of models that learn from their predecessors.
*   **Gradient Boosting:** A specific type of boosting that uses gradient descent to minimize the errors of the sequential models. It's a powerful and flexible algorithm.
*   **XGBoost (Extreme Gradient Boosting):** A highly optimized and efficient implementation of gradient boosting. It includes features like regularization to prevent overfitting and is known for its speed and performance, often winning machine learning competitions.
*   **ROC Curve and AUC:** The Receiver Operating Characteristic (ROC) curve is a graph showing a classifier's performance across all classification thresholds. It plots the True Positive Rate (TPR) against the False Positive Rate (FPR). The Area Under the Curve (AUC) is a single number summarizing this performance. An AUC of 1.0 is a perfect classifier, while an AUC of 0.5 is no better than random guessing. It's particularly useful when the classes are imbalanced.
*   **Why this method?** Gradient Boosting models like XGBoost and LightGBM are often the top-performing models for tabular data (like our dataset). Learning them is essential for anyone looking to achieve state-of-the-art results.

## Discussion Points & Challenges
*   Compare the ROC AUC of XGBoost with the accuracy scores of the previous models. Why might AUC be a more telling metric for this problem?
*   The `learning_rate` is a critical hyperparameter. What happens to the model's performance if you set it to a very high value (e.g., 1.0) or a very low value (e.g., 0.001)?
*   XGBoost has a feature for "early stopping," which automatically stops the training process when performance on a validation set stops improving. Research how to implement `early_stopping_rounds` in the `.fit()` method. Why is this useful?
*   **Challenge:** Install the `lightgbm` library (`pip install lightgbm`) and repeat this scenario using `lgb.LGBMClassifier`. Compare its speed and performance to XGBoost.
