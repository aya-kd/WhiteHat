# Import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, precision_score, recall_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

# Fix random seed for reproducibility
np.random.seed(7)

# Read data
data = pd.read_csv('opcodes_vulnerable.csv')

# Ensure 'opcodes' and 'vulnerable' columns are present
if 'opcodes' not in data.columns or 'vulnerable' not in data.columns:
    raise ValueError("The DataFrame does not contain required columns 'opcodes' and 'vulnerable'.")

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Vectorization using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
sequences = vectorizer.fit_transform(data.opcodes).toarray()

# Encode the labels
labels = np.array(data.vulnerable)
y = labels

# Standardize the data
scaler = StandardScaler()
sequences = scaler.fit_transform(sequences)

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print(class_weights_dict)

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_model = RandomForestClassifier(class_weight=class_weights_dict, random_state=42)

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, scoring='f1', n_jobs=-1, verbose=2)
grid_search.fit(sequences, y)

best_model = grid_search.best_estimator_

print("Best hyperparameters:", grid_search.best_params_)

# Save the best model obtained from GridSearchCV
joblib.dump(best_model, "best_random_forest_1.joblib")
print("Best model saved to best_random_forest_model.joblib")

# K-Fold Cross Validation with the best model
kf = KFold(n_splits=5, shuffle=True, random_state=42)

fold = 1
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []

for train_index, test_index in kf.split(sequences):
    x_train, x_test = sequences[train_index], sequences[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Fit the best model
    best_model.fit(x_train, y_train)

    # Predict the labels for the test set
    y_pred = best_model.predict(x_test)

    # Evaluation
    fold_accuracy = accuracy_score(y_test, y_pred)
    fold_precision = precision_score(y_test, y_pred)
    fold_recall = recall_score(y_test, y_pred)
    fold_f1 = f1_score(y_test, y_pred)

    accuracy_scores.append(fold_accuracy)
    precision_scores.append(fold_precision)
    recall_scores.append(fold_recall)
    f1_scores.append(fold_f1)

    print(f"Fold {fold}:")
    print(f"Accuracy: {fold_accuracy:.4f}")
    print(f"Precision: {fold_precision:.4f}")
    print(f"Recall: {fold_recall:.4f}")
    print(f"F1 Score: {fold_f1:.4f}")
    print()

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Fold {fold}')
    plt.show()

    fold += 1

# Print average metrics
print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Average Precision: {np.mean(precision_scores):.4f}")
print(f"Average Recall: {np.mean(recall_scores):.4f}")
print(f"Average F1 Score: {np.mean(f1_scores):.4f}")
