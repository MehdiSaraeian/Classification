# ML/AI Classification Workshop Scenarios

This document provides a series of hands-on workshop scenarios for teaching ML/AI classification, progressing from beginner to advanced concepts. All scenarios use the Wisconsin Breast Cancer dataset as a practical foundation.

---

## Scenario 1: Introduction to Logistic Regression for Binary Classification

*   **Scenario Title:** Your First Classifier: Predicting Outcomes with Logistic Regression
*   **Learning Objectives:**
    *   Understand the fundamental concept of binary classification.
    *   Learn how to load and prepare data using pandas, including adding column headers and handling missing values.
    *   Implement a Logistic Regression model using scikit-learn.
    *   Grasp the intuition behind the sigmoid function and a linear decision boundary.
    *   Evaluate a classifier using accuracy, a confusion matrix, precision, recall, and the F1-score.
*   **Target Audience:** Beginner
*   **Tools/Libraries:** `pandas`, `numpy`, `scikit-learn`, `seaborn`, `matplotlib`
*   **Dataset Preparation:**
    *   The original dataset file (`breast-cancer-wisconsin.data`) is a raw CSV without a header row. We must first load it and assign the correct column names based on `breast-cancer-wisconsin.names`.
    *   The 'Bare Nuclei' column contains missing values denoted by `'?'`. A simple approach is to replace these with a placeholder, but a better method is to impute them. For this scenario, we will replace `'?'` with `NaN` and then fill the missing values with the median of the column. Using the median is more robust to outliers than using the mean.
    *   The `id` column is a unique identifier for each patient and provides no predictive value, so it will be dropped.
    *   The `class` column will be our target (`y`), and all other columns (after cleaning) will be our features (`X`).

*   **Core Concepts & Methodology:**
    *   **Logistic Regression:** Despite its name, Logistic Regression is a model for classification, not regression. It models the probability that an input belongs to a particular category.
    *   **Sigmoid Function:** It uses the sigmoid function to map any real-valued number into a value between 0 and 1. This output can be interpreted as the probability of belonging to the positive class.
    *   **Decision Boundary:** The model learns a linear boundary to separate the two classes (benign vs. malignant). If the probability output is > 0.5, the instance is classified as one class; otherwise, it's classified as the other.
    *   **Why this method?** Logistic Regression is a great first algorithm to learn. It's powerful, efficient to train, and its results are highly interpretable, making it a common baseline model in many classification tasks.

*   **Hands-on Steps:**

    ```python
    # 1. Import necessary libraries
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
    import seaborn as sns
    import matplotlib.pyplot as plt

    # 2. Load the dataset and assign column names
    column_names = [
        'id', 'clump_thickness', 'unif_cell_size', 'unif_cell_shape',
        'marg_adhesion', 'single_epith_cell_size', 'bare_nuclei',
        'bland_chrom', 'norm_nucleoli', 'mitoses', 'class'
    ]
    df = pd.read_csv('breast-cancer-wisconsin.data', names=column_names)

    # 3. Initial Data Inspection
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    df.info()

    # 4. Preprocess the data
    # Replace '?' with NaN (Not a Number)
    df['bare_nuclei'] = df['bare_nuclei'].replace('?', np.nan)
    # Convert the column to a numeric type
    df['bare_nuclei'] = pd.to_numeric(df['bare_nuclei'])
    # Impute missing values with the median of the column
    median_bare_nuclei = df['bare_nuclei'].median()
    df['bare_nuclei'].fillna(median_bare_nuclei, inplace=True)

    # Drop the 'id' column as it's not a feature
    df.drop('id', axis=1, inplace=True)

    # The 'class' column uses 2 for benign and 4 for malignant.
    # It's good practice to map this to 0 and 1.
    df['class'] = df['class'].map({2: 0, 4: 1})

    print("\nDataset after preprocessing:")
    print(df.head())
    print(f"\nMissing values in 'bare_nuclei' after imputation: {df['bare_nuclei'].isnull().sum()}")

    # 5. Define Features (X) and Target (y)
    X = df.drop('class', axis=1)
    y = df['class']

    # 6. Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 7. Feature Scaling
    # It's important to scale features for Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 8. Initialize and train the Logistic Regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train_scaled, y_train)

    # 9. Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # 10. Evaluate the model
    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign (0)', 'Malignant (1)'], yticklabels=['Benign (0)', 'Malignant (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Classification Report (Precision, Recall, F1-score)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Benign (0)', 'Malignant (1)']))
    ```

*   **Discussion Points & Challenges:**
    *   Why is accuracy not always the best metric for medical diagnosis? (Hint: Think about the cost of a false negative vs. a false positive).
    *   How does changing the `test_size` in `train_test_split` affect the model's performance and reliability?
    *   What does the `stratify=y` parameter do, and why is it important for this dataset?
    *   Explore the coefficients of the trained model (`model.coef_`). What do they tell you about the importance of each feature?

---

## Scenario 2: Decision Trees and Ensemble Methods (Random Forest)

*   **Scenario Title:** Beyond Lines: Making Decisions with Trees and Forests
*   **Learning Objectives:**
    *   Understand the structure of a Decision Tree classifier.
    *   Visualize a decision tree and understand its rules.
    *   Recognize the problem of overfitting in high-capacity models.
    *   Learn how an ensemble method (Random Forest) combines multiple models to improve robustness and accuracy.
    *   Extract and visualize feature importances from a trained Random Forest model.
*   **Target Audience:** Beginner to Intermediate
*   **Tools/Libraries:** `pandas`, `scikit-learn`, `matplotlib`, `seaborn`
*   **Dataset Preparation:** Same as Scenario 1. We will use the same cleaned and preprocessed dataset.

*   **Core Concepts & Methodology:**
    *   **Decision Tree:** A non-linear model that learns a hierarchy of if/else questions to make a prediction. It's highly interpretable and mimics human decision-making.
    *   **Gini Impurity / Information Gain:** These are the metrics used by the tree to decide the "best" split at each node to separate the classes as cleanly as possible.
    *   **Overfitting:** Decision Trees can grow very deep and learn the training data perfectly, but fail to generalize to new, unseen data. We'll see this by comparing training and testing accuracy.
    *   **Random Forest:** An ensemble method that builds many Decision Trees on different subsets of the data and features (`bagging`). To make a prediction, it averages the votes from all the individual trees. This process reduces overfitting and usually leads to a much better model.
    *   **Why this method?** Decision Trees are easy to understand and visualize. Random Forest is a natural next step that introduces the powerful concept of ensembling and is one of the most widely used and effective machine learning algorithms.

*   **Hands-on Steps:**

    ```python
    # Prerequisite: Assumes 'df' is the cleaned DataFrame from Scenario 1
    # (data loaded, missing values imputed, 'id' dropped, 'class' mapped to 0/1)
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report
    import matplotlib.pyplot as plt

    # --- Data Prep (repeat from Scenario 1 for a self-contained script) ---
    column_names = [
        'id', 'clump_thickness', 'unif_cell_size', 'unif_cell_shape',
        'marg_adhesion', 'single_epith_cell_size', 'bare_nuclei',
        'bland_chrom', 'norm_nucleoli', 'mitoses', 'class'
    ]
    df = pd.read_csv('breast-cancer-wisconsin.data', names=column_names)
    df['bare_nuclei'] = df['bare_nuclei'].replace('?', np.nan)
    df['bare_nuclei'] = pd.to_numeric(df['bare_nuclei'])
    df['bare_nuclei'].fillna(df['bare_nuclei'].median(), inplace=True)
    df.drop('id', axis=1, inplace=True)
    df['class'] = df['class'].map({2: 0, 4: 1})
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # --- End Data Prep ---

    # 1. Train a single Decision Tree
    # No feature scaling is needed for tree-based models
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)

    # 2. Evaluate the single Decision Tree
    y_pred_dt = dt_model.predict(X_test)
    print("--- Single Decision Tree ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_dt):.4f}")
    print(classification_report(y_test, y_pred_dt))

    # 3. Visualize the Decision Tree (first few levels)
    plt.figure(figsize=(20, 10))
    plot_tree(dt_model, feature_names=X.columns, class_names=['Benign', 'Malignant'], max_depth=3, filled=True, fontsize=10)
    plt.title("Decision Tree Visualization (Top Levels)")
    plt.show()

    # 4. Train a Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)

    # 5. Evaluate the Random Forest model
    y_pred_rf = rf_model.predict(X_test)
    print("\n--- Random Forest ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(classification_report(y_test, y_pred_rf))

    # 6. Analyze Feature Importance
    importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    importances.plot(kind='bar')
    plt.title('Feature Importances from Random Forest')
    plt.ylabel('Importance')
    plt.show()
    print("\nTop 5 most important features:")
    print(importances.head())
    ```

*   **Discussion Points & Challenges:**
    *   Compare the accuracy of the single Decision Tree to the Random Forest. Why is the Random Forest likely more accurate?
    *   Train a Decision Tree with `max_depth=3` and another with no `max_depth`. Compare their training and testing accuracies. What does this show about overfitting?
    *   What does the `n_estimators` parameter in `RandomForestClassifier` control? Try changing it to a very small number (e.g., 5) and a very large number (e.g., 500) and see how it affects performance.
    *   How could the feature importance information be used by doctors or researchers?

---

## Scenario 3: Support Vector Machines (SVM) with Different Kernels

*   **Scenario Title:** Finding the Perfect Divide: Classification with Support Vector Machines
*   **Learning Objectives:**
    *   Understand the intuition behind Support Vector Machines (SVMs) and maximizing the margin.
    *   Learn the role of support vectors in defining the decision boundary.
    *   Apply the "kernel trick" to solve non-linearly separable problems.
    *   Compare the performance of different kernels (e.g., `linear`, `rbf`).
    *   Use cross-validation to get a more robust estimate of model performance.
*   **Target Audience:** Intermediate
*   **Tools/Libraries:** `pandas`, `scikit-learn`, `numpy`
*   **Dataset Preparation:** Same as Scenario 1. Feature scaling is crucial for SVMs.

*   **Core Concepts & Methodology:**
    *   **Support Vector Machine (SVM):** A powerful classifier that works by finding the optimal hyperplane (a line, in 2D) that best separates the classes in the feature space.
    *   **Margin:** The "street" between the classes. SVM tries to find the hyperplane that maximizes the width of this margin, making the model more robust to new data.
    *   **Support Vectors:** The data points that lie closest to the hyperplane and are most difficult to classify. They are the critical elements that "support" the hyperplane.
    *   **Kernel Trick:** A mathematical function that takes the original, non-linearly separable data and transforms it into a higher-dimensional space where a linear separator can be found. Common kernels are `linear`, `poly`, and `rbf` (Radial Basis Function).
    *   **Cross-Validation:** A technique to evaluate a model by splitting the data into several "folds". The model is trained and tested multiple times, with each fold getting a turn as the test set. The results are then averaged, providing a more reliable performance estimate than a single train-test split.
    *   **Why this method?** SVMs are very effective in high-dimensional spaces and are memory efficient because they only use a subset of training points (the support vectors). Understanding kernels is a key step towards mastering advanced ML techniques.

*   **Hands-on Steps:**

    ```python
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import classification_report

    # --- Data Prep (repeat from Scenario 1 for a self-contained script) ---
    column_names = [
        'id', 'clump_thickness', 'unif_cell_size', 'unif_cell_shape',
        'marg_adhesion', 'single_epith_cell_size', 'bare_nuclei',
        'bland_chrom', 'norm_nucleoli', 'mitoses', 'class'
    ]
    df = pd.read_csv('breast-cancer-wisconsin.data', names=column_names)
    df['bare_nuclei'] = df['bare_nuclei'].replace('?', np.nan)
    df['bare_nuclei'] = pd.to_numeric(df['bare_nuclei'])
    df['bare_nuclei'].fillna(df['bare_nuclei'].median(), inplace=True)
    df.drop('id', axis=1, inplace=True)
    df['class'] = df['class'].map({2: 0, 4: 1})
    X = df.drop('class', axis=1)
    y = df['class']
    # For SVM, we need to scale the data before splitting or cross-validation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # --- End Data Prep ---

    # 1. Train an SVM with a Linear Kernel
    # We will use cross-validation instead of a single train/test split
    print("--- SVM with Linear Kernel ---")
    linear_svm = SVC(kernel='linear', random_state=42)

    # Perform 5-fold cross-validation
    cv_scores_linear = cross_val_score(linear_svm, X_scaled, y, cv=5)

    print(f"Cross-validation scores: {cv_scores_linear}")
    print(f"Average CV score: {cv_scores_linear.mean():.4f}")
    print(f"Standard deviation of CV scores: {cv_scores_linear.std():.4f}")

    # 2. Train an SVM with an RBF (Radial Basis Function) Kernel
    print("\n--- SVM with RBF Kernel ---")
    rbf_svm = SVC(kernel='rbf', random_state=42)

    # Perform 5-fold cross-validation
    cv_scores_rbf = cross_val_score(rbf_svm, X_scaled, y, cv=5)

    print(f"Cross-validation scores: {cv_scores_rbf}")
    print(f"Average CV score: {cv_scores_rbf.mean():.4f}")
    print(f"Standard deviation of CV scores: {cv_scores_rbf.std():.4f}")

    # 3. Full training and evaluation on a test set for a final report
    # Let's see the detailed report for the best model (usually RBF for this task)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

    rbf_svm_final = SVC(kernel='rbf', random_state=42)
    rbf_svm_final.fit(X_train, y_train)
    y_pred = rbf_svm_final.predict(X_test)

    print("\n--- Final Evaluation of RBF SVM on Test Set ---")
    print(classification_report(y_test, y_pred))
    ```

*   **Discussion Points & Challenges:**
    *   Compare the performance of the linear and RBF kernels. Why do you think one performed better than the other?
    *   The `SVC` model has two very important hyperparameters: `C` (the regularization parameter) and `gamma` (the kernel coefficient for 'rbf'). Research what these parameters do.
    *   **Challenge:** Use `GridSearchCV` from scikit-learn to find the best combination of `C` and `gamma` for the RBF SVM. This is a crucial skill for optimizing model performance.
    *   Why is cross-validation a better approach for evaluating a model than a single train-test split, especially with smaller datasets?

---

## Scenario 4: Gradient Boosting Machines (XGBoost/LightGBM)

*   **Scenario Title:** Peak Performance: Competitive Classification with Gradient Boosting
*   **Learning Objectives:**
    *   Understand the core concept of "boosting" in ensemble learning.
    *   Implement a high-performance gradient boosting model using the XGBoost library.
    *   Learn about key hyperparameters in boosting models (e.g., `n_estimators`, `learning_rate`, `max_depth`).
    *   Evaluate the model using the ROC AUC score, a robust metric for imbalanced classes.
    *   Plot an ROC curve to visualize the classifier's performance.
*   **Target Audience:** Intermediate to Advanced
*   **Tools/Libraries:** `pandas`, `scikit-learn`, `xgboost`, `matplotlib`
*   **Dataset Preparation:** Same as Scenario 1. No feature scaling is needed for XGBoost.

*   **Core Concepts & Methodology:**
    *   **Boosting:** An ensemble technique where models are built sequentially. Each new model attempts to correct the errors made by the previous ones. Instead of independent trees (like in Random Forest), boosting builds a chain of models that learn from their predecessors.
    *   **Gradient Boosting:** A specific type of boosting that uses gradient descent to minimize the errors of the sequential models. It's a powerful and flexible algorithm.
    *   **XGBoost (Extreme Gradient Boosting):** A highly optimized and efficient implementation of gradient boosting. It includes features like regularization to prevent overfitting and is known for its speed and performance, often winning machine learning competitions.
    *   **ROC Curve and AUC:** The Receiver Operating Characteristic (ROC) curve is a graph showing a classifier's performance across all classification thresholds. It plots the True Positive Rate (TPR) against the False Positive Rate (FPR). The Area Under the Curve (AUC) is a single number summarizing this performance. An AUC of 1.0 is a perfect classifier, while an AUC of 0.5 is no better than random guessing. It's particularly useful when the classes are imbalanced.
    *   **Why this method?** Gradient Boosting models like XGBoost and LightGBM are often the top-performing models for tabular data (like our dataset). Learning them is essential for anyone looking to achieve state-of-the-art results.

*   **Hands-on Steps:**

    ```python
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, roc_curve, classification_report
    import matplotlib.pyplot as plt

    # --- Data Prep (repeat from Scenario 1 for a self-contained script) ---
    column_names = [
        'id', 'clump_thickness', 'unif_cell_size', 'unif_cell_shape',
        'marg_adhesion', 'single_epith_cell_size', 'bare_nuclei',
        'bland_chrom', 'norm_nucleoli', 'mitoses', 'class'
    ]
    df = pd.read_csv('breast-cancer-wisconsin.data', names=column_names)
    df['bare_nuclei'] = df['bare_nuclei'].replace('?', np.nan)
    df['bare_nuclei'] = pd.to_numeric(df['bare_nuclei'])
    df['bare_nuclei'].fillna(df['bare_nuclei'].median(), inplace=True)
    df.drop('id', axis=1, inplace=True)
    df['class'] = df['class'].map({2: 0, 4: 1})
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # --- End Data Prep ---

    # 1. Initialize and train the XGBoost Classifier
    # Some common hyperparameters are included
    xgb_model = xgb.XGBClassifier(
        objective='binary:logistic', # for binary classification
        n_estimators=100,            # number of trees
        learning_rate=0.1,           # how much to shrink the contribution of each tree
        max_depth=3,                 # maximum depth of each tree
        use_label_encoder=False,     # suppresses a deprecation warning
        eval_metric='logloss',       # evaluation metric for the training process
        random_state=42
    )
    xgb_model.fit(X_train, y_train)

    # 2. Make predictions
    y_pred = xgb_model.predict(X_test)
    y_pred_proba = xgb_model.predict_proba(X_test)[:, 1] # get probabilities for the positive class

    # 3. Evaluate the model
    print("--- XGBoost Classifier ---")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Calculate ROC AUC Score
    auc = roc_auc_score(y_test, y_pred_proba)
    print(f"\nROC AUC Score: {auc:.4f}")

    # 4. Plot the ROC Curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Random guess line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    ```

*   **Discussion Points & Challenges:**
    *   Compare the ROC AUC of XGBoost with the accuracy scores of the previous models. Why might AUC be a more telling metric for this problem?
    *   The `learning_rate` is a critical hyperparameter. What happens to the model's performance if you set it to a very high value (e.g., 1.0) or a very low value (e.g., 0.001)?
    *   XGBoost has a feature for "early stopping," which automatically stops the training process when performance on a validation set stops improving. Research how to implement `early_stopping_rounds` in the `.fit()` method. Why is this useful?
    *   **Challenge:** Install the `lightgbm` library (`pip install lightgbm`) and repeat this scenario using `lgb.LGBMClassifier`. Compare its speed and performance to XGBoost.

---

## Scenario 5: Introduction to Neural Networks for Classification

*   **Scenario Title:** The Deep Dive: A Gentle Introduction to Neural Networks
*   **Learning Objectives:**
    *   Understand the basic architecture of a neural network (layers, neurons, activation functions).
    *   Build, compile, and train a simple sequential neural network using TensorFlow/Keras.
    *   Understand the roles of a loss function and an optimizer.
    *   Visualize the model's training and validation performance over epochs to identify learning trends.
    *   Use early stopping to prevent overfitting.
*   **Target Audience:** Advanced
*   **Tools/Libraries:** `pandas`, `scikit-learn`, `tensorflow`
*   **Dataset Preparation:** Same as Scenario 1. Feature scaling is essential for neural networks.

*   **Core Concepts & Methodology:**
    *   **Neural Network:** A model inspired by the human brain, composed of interconnected nodes called "neurons" organized in "layers."
    *   **Layers:** A typical network has an input layer (for the features), one or more "hidden" layers where computations happen, and an output layer that produces the final prediction.
    *   **Activation Functions:** Functions like `ReLU` (Rectified Linear Unit) are applied in hidden layers to introduce non-linearity, allowing the network to learn complex patterns. The output layer uses a `sigmoid` function for binary classification to produce a probability.
    *   **Backpropagation and Optimizer:** The network learns by making a prediction, calculating the error (using a `loss function` like `binary_crossentropy`), and then propagating the error backward through the network to adjust its internal weights. The `optimizer` (e.g., `Adam`) is the algorithm that handles these weight adjustments.
    *   **Epochs:** One full pass through the entire training dataset.
    *   **Early Stopping:** A regularization technique where you monitor the model's performance on a validation set and stop training when the performance stops improving, preventing overfitting.
    *   **Why this method?** While potentially overkill for this simple dataset, understanding neural networks is fundamental to modern AI and deep learning. This scenario provides a gentle, practical introduction to the core concepts and workflow.

*   **Hands-on Steps:**

    ```python
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import classification_report
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    import matplotlib.pyplot as plt

    # --- Data Prep (repeat from Scenario 1 for a self-contained script) ---
    column_names = [
        'id', 'clump_thickness', 'unif_cell_size', 'unif_cell_shape',
        'marg_adhesion', 'single_epith_cell_size', 'bare_nuclei',
        'bland_chrom', 'norm_nucleoli', 'mitoses', 'class'
    ]
    df = pd.read_csv('breast-cancer-wisconsin.data', names=column_names)
    df['bare_nuclei'] = df['bare_nuclei'].replace('?', np.nan)
    df['bare_nuclei'] = pd.to_numeric(df['bare_nuclei'])
    df['bare_nuclei'].fillna(df['bare_nuclei'].median(), inplace=True)
    df.drop('id', axis=1, inplace=True)
    df['class'] = df['class'].map({2: 0, 4: 1})
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # --- End Data Prep ---

    # 1. Build the Neural Network model
    model = Sequential([
        # Input layer and first hidden layer
        Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        # Dropout layer for regularization
        Dropout(0.2),
        # Second hidden layer
        Dense(16, activation='relu'),
        # Output layer - 1 neuron with sigmoid for binary classification
        Dense(1, activation='sigmoid')
    ])

    # 2. Compile the model
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Print a summary of the model architecture
    model.summary()

    # 3. Define an Early Stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss', # metric to monitor
        patience=10,        # number of epochs with no improvement to wait before stopping
        restore_best_weights=True # restores model weights from the epoch with the best value
    )

    # 4. Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2, # use part of training data for validation
        epochs=100,           # train for a large number of epochs; early stopping will handle the rest
        batch_size=32,
        callbacks=[early_stopping],
        verbose=1
    )

    # 5. Plot training & validation loss and accuracy
    history_df = pd.DataFrame(history.history)
    history_df.loc[:, ['loss', 'val_loss']].plot(title="Loss Curve")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()
    history_df.loc[:, ['accuracy', 'val_accuracy']].plot(title="Accuracy Curve")
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    # 6. Evaluate the model on the test set
    print("\n--- Neural Network Evaluation on Test Set ---")
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {accuracy:.4f}")

    # Get a full classification report
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype("int32") # Convert probabilities to binary classes
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    ```

*   **Discussion Points & Challenges:**
    *   What is the purpose of the `Dropout` layer? Try removing it or increasing the dropout rate (e.g., to 0.5) and see how it affects the training curves and final performance.
    *   Explain the difference between `loss` and `val_loss` in the training output. What does it mean if `val_loss` starts increasing while `loss` continues to decrease?
    *   The structure of the network (number of layers, number of neurons) is a key hyperparameter. Try making the network "wider" (more neurons per layer) or "deeper" (more layers). How does this impact performance and training time?
    *   What is the difference between a `batch` and an `epoch`?

---

## Scenario 6: Addressing Data Imbalance and Model Interpretability

*   **Scenario Title:** Real-World ML: Handling Imbalance and Explaining Predictions
*   **Learning Objectives:**
    *   Recognize the challenges of working with imbalanced datasets.
    *   Apply an over-sampling technique (SMOTE) to balance the class distribution.
    *   Understand the concept of class weights as an alternative to resampling.
    *   Use the SHAP library to explain the predictions of a black-box model (like XGBoost).
    *   Generate and interpret SHAP force plots and summary plots.
*   **Target Audience:** Advanced
*   **Tools/Libraries:** `pandas`, `scikit-learn`, `xgboost`, `imblearn`, `shap`, `matplotlib`
*   **Dataset Preparation:** Same as Scenario 1.

*   **Core Concepts & Methodology:**
    *   **Class Imbalance:** Occurs when one class in the dataset is much more frequent than another. This can cause models to become biased towards the majority class. Our dataset is moderately imbalanced (65.5% Benign, 34.5% Malignant).
    *   **SMOTE (Synthetic Minority Over-sampling Technique):** An algorithm to address imbalance by creating "synthetic" new samples of the minority class. Instead of just duplicating existing samples, it generates new ones based on the feature space proximity of existing minority points.
    *   **Class Weights:** A technique where you penalize the model more for making mistakes on the minority class during training. This can be done by setting the `class_weight` parameter in many scikit-learn models or `scale_pos_weight` in XGBoost.
    *   **Model Interpretability (XAI - Explainable AI):** The process of understanding and trusting the results of a machine learning model. For complex "black box" models like XGBoost or Neural Networks, it's hard to know *why* they made a specific prediction.
    *   **SHAP (SHapley Additive exPlanations):** A game theory approach to explaining the output of any machine learning model. It connects optimal credit allocation with local explanations using the classic Shapley values. For a single prediction, SHAP values show how much each feature contributed to pushing the prediction away from the base value (the average prediction over the dataset).
    *   **Why this method?** In the real world, datasets are rarely perfect. They are often imbalanced. Furthermore, in high-stakes domains like medicine, simply having an accurate model is not enough; you must be able to explain *why* it makes the decisions it does. These techniques are crucial for deploying ML responsibly.

*   **Hands-on Steps:**

    ```python
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    # Make sure to install imblearn and shap: pip install imbalanced-learn shap
    from imblearn.over_sampling import SMOTE
    import shap
    import matplotlib.pyplot as plt

    # --- Data Prep (repeat from Scenario 1 for a self-contained script) ---
    column_names = [
        'id', 'clump_thickness', 'unif_cell_size', 'unif_cell_shape',
        'marg_adhesion', 'single_epith_cell_size', 'bare_nuclei',
        'bland_chrom', 'norm_nucleoli', 'mitoses', 'class'
    ]
    df = pd.read_csv('breast-cancer-wisconsin.data', names=column_names)
    df['bare_nuclei'] = df['bare_nuclei'].replace('?', np.nan)
    df['bare_nuclei'] = pd.to_numeric(df['bare_nuclei'])
    df['bare_nuclei'].fillna(df['bare_nuclei'].median(), inplace=True)
    df.drop('id', axis=1, inplace=True)
    df['class'] = df['class'].map({2: 0, 4: 1})
    X = df.drop('class', axis=1)
    y = df['class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    # --- End Data Prep ---

    # --- Part 1: Handling Class Imbalance with SMOTE ---
    print("--- Handling Class Imbalance with SMOTE ---")
    print("Class distribution before SMOTE:")
    print(y_train.value_counts())

    # Apply SMOTE only to the training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print("\nClass distribution after SMOTE:")
    print(y_train_smote.value_counts())

    # Train a model on the balanced data
    xgb_smote = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_smote.fit(X_train_smote, y_train_smote)
    y_pred_smote = xgb_smote.predict(X_test)

    print("\nClassification Report (Model trained on SMOTE data):")
    print(classification_report(y_test, y_pred_smote))

    # --- Part 2: Model Interpretability with SHAP ---
    # We will use a model trained on the original (non-SMOTE) data for this
    print("\n--- Model Interpretability with SHAP ---")
    xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)

    # 1. Create a SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)

    # 2. Calculate SHAP values for the test set
    shap_values = explainer.shap_values(X_test)

    # 3. Visualize a single prediction with a force plot
    # This shows features contributing to push the prediction from the base value
    print("\nGenerating SHAP force plot for the first test instance...")
    shap.initjs() # required for force plots in notebooks
    display(shap.force_plot(explainer.expected_value, shap_values[0,:], X_test.iloc[0,:]))

    # 4. Create a summary plot to see global feature importance
    print("\nGenerating SHAP summary plot...")
    shap.summary_plot(shap_values, X_test, show=False)
    plt.title("SHAP Summary Plot")
    plt.show()

    # 5. Create a dependence plot for a single feature
    # Shows how a single feature's value affects the SHAP value (and thus the prediction)
    print("\nGenerating SHAP dependence plot for 'unif_cell_size'...")
    shap.dependence_plot("unif_cell_size", shap_values, X_test, interaction_index="unif_cell_shape")
    plt.show()
    ```

*   **Discussion Points & Challenges:**
    *   Compare the classification report from the model trained on SMOTE data to one trained on the original data. Did SMOTE improve the recall for the minority class (malignant)?
    *   An alternative to SMOTE is using class weights. For XGBoost, you can calculate `scale_pos_weight = count(negative class) / count(positive class)` and pass it to the classifier. Try this method and compare its results to the SMOTE approach.
    *   Look at the SHAP summary plot. How does it differ from the feature importance plot from Random Forest? (Hint: SHAP shows not just the importance but also the *direction* of the effect).
    *   Pick a few instances from the test set that the model got wrong. Use SHAP force plots to try and understand *why* the model made a mistake.
```
