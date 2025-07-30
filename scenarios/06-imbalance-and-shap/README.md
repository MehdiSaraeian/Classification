# Scenario 6: Real-World ML: Handling Imbalance and Explaining Predictions

## Learning Objectives
*   Recognize the challenges of working with imbalanced datasets.
*   Apply an over-sampling technique (SMOTE) to balance the class distribution.
*   Understand the concept of class weights as an alternative to resampling.
*   Use the SHAP library to explain the predictions of a black-box model (like XGBoost).
*   Generate and interpret SHAP force plots and summary plots.

## Target Audience
Advanced

## Tools/Libraries
`pandas`, `scikit-learn`, `xgboost`, `imblearn`, `shap`, `matplotlib`

## Core Concepts & Methodology
*   **Class Imbalance:** Occurs when one class in the dataset is much more frequent than another. This can cause models to become biased towards the majority class. Our dataset is moderately imbalanced (65.5% Benign, 34.5% Malignant).
*   **SMOTE (Synthetic Minority Over-sampling Technique):** An algorithm to address imbalance by creating "synthetic" new samples of the minority class. Instead of just duplicating existing samples, it generates new ones based on the feature space proximity of existing minority points.
*   **Class Weights:** A technique where you penalize the model more for making mistakes on the minority class during training. This can be done by setting the `class_weight` parameter in many scikit-learn models or `scale_pos_weight` in XGBoost.
*   **Model Interpretability (XAI - Explainable AI):** The process of understanding and trusting the results of a machine learning model. For complex "black box" models like XGBoost or Neural Networks, it's hard to know *why* they made a specific prediction.
*   **SHAP (SHapley Additive exPlanations):** A game theory approach to explaining the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values. For a single prediction, SHAP values show how much each feature contributed to pushing the prediction away from the base value (the average prediction over the dataset).
*   **Why this method?** In the real world, datasets are rarely perfect. They are often imbalanced. Furthermore, in high-stakes domains like medicine, simply having an accurate model is not enough; you must be able to explain *why* it makes the decisions it does. These techniques are crucial for deploying ML responsibly.

## Discussion Points & Challenges
*   Compare the classification report from the model trained on SMOTE data to one trained on the original data. Did SMOTE improve the recall for the minority class (malignant)?
*   An alternative to SMOTE is using class weights. For XGBoost, you can calculate `scale_pos_weight = count(negative class) / count(positive class)` and pass it to the classifier. Try this method and compare its results to the SMOTE approach.
*   Look at the SHAP summary plot. How does it differ from the feature importance plot from Random Forest? (Hint: SHAP shows not just the importance but also the *direction* of the effect).
*   Pick a few instances from the test set that the model got wrong. Use SHAP force plots to try and understand *why* the model made a mistake.
