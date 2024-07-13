import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight  # Add this import
import matplotlib.pyplot as plt

# Function to truncate sequences to the first 5000 words
def truncate_sequences(sequences, max_words=5000):
    return [' '.join(seq.split()[:max_words]) for seq in sequences]

# Read data from CSV
data = pd.read_csv('opcodes_vulnerable.csv')

# Ensure 'opcodes' and 'vulnerable' columns are present
if 'opcodes' not in data.columns or 'vulnerable' not in data.columns:
    raise ValueError("The DataFrame does not contain required columns 'opcodes' and 'vulnerable'.")

# Separate opcodes and labels
opcodes = data['opcodes']
labels = data['vulnerable']

# Truncate the opcodes to the first 5000 words
opcodes = truncate_sequences(opcodes)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(opcodes, labels, test_size=0.2, random_state=42)

# Calculate class weights
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("Class weights:", class_weights_dict)

# Define a pipeline with feature extraction and SVM classifier
pipeline = Pipeline([
    ('vectorizer', TfidfVectorizer(binary=True)),
    ('svm', SVC(class_weight=class_weights_dict))
])

# Define parameter grid for GridSearchCV
param_grid = {
    'vectorizer__ngram_range': [(1, 2), (1, 3)],
    'vectorizer__max_df': [0.8, 0.9, 1.0],
    'vectorizer__min_df': [0.01, 0.02, 0.05],
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 'auto'],
    'svm__kernel': ['rbf', 'linear']
}

# Perform GridSearchCV with progress printing
grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, scoring='accuracy', verbose=2)
grid_search.fit(X_train, y_train)

# Get the accuracy for each iteration
results_df = pd.DataFrame(grid_search.cv_results_)

# Plot accuracy
plt.figure(figsize=(10, 6))
plt.plot(results_df['rank_test_score'], results_df['mean_test_score'], marker='o', linestyle='-')
plt.title('Accuracy vs. Iteration')
plt.xlabel('Iteration')
plt.ylabel('Mean Test Accuracy')
plt.grid(True)
plt.show()

# Print best parameters found by GridSearchCV
print("Best parameters found:")
print(grid_search.best_params_)

# Use best estimator from GridSearchCV
best_pipeline = grid_search.best_estimator_

# Predict on test set
y_pred = best_pipeline.predict(X_test)

# Calculate metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

# Display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Vulnerable', 'Vulnerable'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix for Test Set')
plt.show()

# Print metrics
print(f'Accuracy: {acc:.4f}')
print(f'Precision: {prec:.4f}')
print(f'Recall: {rec:.4f}')
print(f'F1 Score: {f1:.4f}')
