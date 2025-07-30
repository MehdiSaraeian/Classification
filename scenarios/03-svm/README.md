# Scenario 3: Finding the Perfect Divide: Classification with Support Vector Machines

## Learning Objectives
*   Understand the intuition behind Support Vector Machines (SVMs) and maximizing the margin.
*   Learn the role of support vectors in defining the decision boundary.
*   Apply the "kernel trick" to solve non-linearly separable problems.
*   Compare the performance of different kernels (e.g., `linear`, `rbf`).
*   Use cross-validation to get a more robust estimate of model performance.

## Target Audience
Intermediate

## Tools/Libraries
`pandas`, `scikit-learn`, `numpy`

## Core Concepts & Methodology
*   **Support Vector Machine (SVM):** A powerful classifier that works by finding the optimal hyperplane (a line, in 2D) that best separates the classes in the feature space.
*   **Margin:** The "street" between the classes. SVM tries to find the hyperplane that maximizes the width of this margin, making the model more robust to new data.
*   **Support Vectors:** The data points that lie closest to the hyperplane and are most difficult to classify. They are the critical elements that "support" the hyperplane.
*   **Kernel Trick:** A mathematical function that takes the original, non-linearly separable data and transforms it into a higher-dimensional space where a linear separator can be found. Common kernels are `linear`, `poly`, and `rbf` (Radial Basis Function).
*   **Cross-Validation:** A technique to evaluate a model by splitting the data into several "folds". The model is trained and tested multiple times, with each fold getting a turn as the test set. The results are then averaged, providing a more reliable performance estimate than a single train-test split.
*   **Why this method?** SVMs are very effective in high-dimensional spaces and are memory efficient because they only use a subset of training points (the support vectors). Understanding kernels is a key step towards mastering advanced ML techniques.

## Discussion Points & Challenges
*   Compare the performance of the linear and RBF kernels. Why do you think one performed better than the other?
*   The `SVC` model has two very important hyperparameters: `C` (the regularization parameter) and `gamma` (the kernel coefficient for 'rbf'). Research what these parameters do.
*   **Challenge:** Use `GridSearchCV` from scikit-learn to find the best combination of `C` and `gamma` for the RBF SVM. This is a crucial skill for optimizing model performance.
*   Why is cross-validation a better approach for evaluating a model than a single train-test split, especially with smaller datasets?
