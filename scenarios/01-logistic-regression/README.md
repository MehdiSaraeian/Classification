# Scenario 1: Your First Classifier: Predicting Outcomes with Logistic Regression

## Learning Objectives
*   Understand the fundamental concept of binary classification.
*   Learn how to load and prepare data using pandas, including adding column headers and handling missing values.
*   Implement a Logistic Regression model using scikit-learn.
*   Grasp the intuition behind the sigmoid function and a linear decision boundary.
*   Evaluate a classifier using accuracy, a confusion matrix, precision, recall, and the F1-score.

## Target Audience
Beginner

## Tools/Libraries
`pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`

## Core Concepts & Methodology
*   **Logistic Regression:** Despite its name, Logistic Regression is a model for classification, not regression. It models the probability that an input belongs to a particular category.
*   **Sigmoid Function:** It uses the sigmoid function to map any real-valued number into a value between 0 and 1. This output can be interpreted as the probability of belonging to the positive class.
*   **Decision Boundary:** The model learns a linear boundary to separate the two classes (benign vs. malignant). If the probability output is > 0.5, the instance is classified as one class; otherwise, it's classified as the other.
*   **Why this method?** Logistic Regression is a great first algorithm to learn. It's powerful, efficient to train, and its results are highly interpretable, making it a common baseline model in many classification tasks.

## Discussion Points & Challenges
*   Why is accuracy not always the best metric for medical diagnosis? (Hint: Think about the cost of a false negative vs. a false positive).
*   How does changing the `test_size` in `train_test_split` affect the model's performance and reliability?
*   What does the `stratify=y` parameter do, and why is it important for this dataset?
*   Explore the coefficients of the trained model (`model.coef_`). What do they tell you about the importance of each feature?
