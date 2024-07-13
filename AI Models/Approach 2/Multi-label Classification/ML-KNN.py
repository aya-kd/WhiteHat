import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix
from skmultilearn.adapt import MLkNN
from sklearn.metrics import ConfusionMatrixDisplay

# Fix random seed for reproducibility
np.random.seed(7)

# Read data
data = pd.read_csv('opcodes_vulnerabilities_filtered.csv')

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Split the vulnerabilities column into lists of vulnerabilities
data['vulnerabilities'] = data['vulnerabilities'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Truncate the opcodes to only use the first 5000
data['opcodes'] = data['opcodes'].apply(lambda x: ' '.join(x.split()[:5000]))

# Vectorize the sequences of opcodes
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), stop_words='english')
X = vectorizer.fit_transform(data.opcodes)

# Binarize the labels using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(data['vulnerabilities'])

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the MLKNN model
mlknn = MLkNN()

# Define parameters for grid search
parameters = {'k': [1, 3, 5, 7, 9, 11, 13, 15]}

# Define cross-validation
cv = KFold(n_splits=10, shuffle=True, random_state=42)

# Perform grid search with cross-validation
grid_search = GridSearchCV(mlknn, parameters, cv=cv, scoring='f1_micro', verbose=1, n_jobs=-1)
grid_search.fit(x_train, y_train)

# Print the best parameters and best score
print("Best Parameters:", grid_search.best_params_)
print("Best F1-score:", grid_search.best_score_)

# Get the best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the best model on the test set
y_pred = best_model.predict(x_test)
y_pred_dense = y_pred.toarray()
y_test_dense = y_test

# Calculate evaluation metrics
accuracy = accuracy_score(y_test_dense, y_pred_dense)
precision = precision_score(y_test_dense, y_pred_dense, average='micro')
recall = recall_score(y_test_dense, y_pred_dense, average='micro')
f1 = f1_score(y_test_dense, y_pred_dense, average='micro')

print('\nTest Metrics:')
print('Test Accuracy: {:.2f}%'.format(accuracy * 100))
print('Test Precision: {:.4f}'.format(precision))
print('Test Recall: {:.4f}'.format(recall))
print('Test F1 Score: {:.4f}'.format(f1))

# Compute and display confusion matrix for each label
cms = multilabel_confusion_matrix(y_test_dense, y_pred_dense)
for i, cm in enumerate(cms):
    print(f'\nConfusion Matrix for Label {i}:')
    print(cm)

# Assuming that the MultiLabelBinarizer was fit using the vulnerability names
vulnerability_labels = mlb.classes_

# Plot confusion matrices with actual vulnerability labels
fig, axes = plt.subplots(nrows=len(cms)//3 + 1, ncols=3, figsize=(15, 10))
axes = axes.flatten()
for i, cm in enumerate(cms):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    disp.plot(ax=axes[i], cmap='Blues', values_format='d')
    axes[i].set_title(f'{vulnerability_labels[i]}')
plt.tight_layout()
plt.show()
