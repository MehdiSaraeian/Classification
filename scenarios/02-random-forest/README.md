# Scenario 2: Beyond Lines: Making Decisions with Trees and Forests

## Learning Objectives
*   Understand the structure of a Decision Tree classifier.
*   Visualize a decision tree and understand its rules.
*   Recognize the problem of overfitting in high-capacity models.
*   Learn how an ensemble method (Random Forest) combines multiple models to improve robustness and accuracy.
*   Extract and visualize feature importances from a trained Random Forest model.

## Target Audience
Beginner to Intermediate

## Tools/Libraries
`pandas`, `scikit-learn`, `matplotlib`, `seaborn`

## Core Concepts & Methodology
*   **Decision Tree:** A non-linear model that learns a hierarchy of if/else questions to make a prediction. It's highly interpretable and mimics human decision-making.
*   **Gini Impurity / Information Gain:** These are the metrics used by the tree to decide the "best" split at each node to separate the classes as cleanly as possible.
*   **Overfitting:** Decision Trees can grow very deep and learn the training data perfectly, but fail to generalize to new, unseen data. We'll see this by comparing training and testing accuracy.
*   **Random Forest:** An ensemble method that builds many Decision Trees on different subsets of the data and features (`bagging`). To make a prediction, it averages the votes from all the individual trees. This process reduces overfitting and usually leads to a much better model.
*   **Why this method?** Decision Trees are easy to understand and visualize. Random Forest is a natural next step that introduces the powerful concept of ensembling and is one of the most widely used and effective machine learning algorithms.

## Discussion Points & Challenges
*   Compare the accuracy of the single Decision Tree to the Random Forest. Why is the Random Forest likely more accurate?
*   Train a Decision Tree with `max_depth=3` and another with no `max_depth`. Compare their training and testing accuracies. What does this show about overfitting?
*   What does the `n_estimators` parameter in `RandomForestClassifier` control? Try changing it to a very small number (e.g., 5) and a very large number (e.g., 500) and see how it affects performance.
*   How could the feature importance information be used by doctors or researchers?
