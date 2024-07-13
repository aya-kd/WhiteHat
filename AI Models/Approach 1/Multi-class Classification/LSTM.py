import tensorflow as tf
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Custom F1 Score Metric
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)
        
    def result(self):
        precision = self.precision.result()
        recall = self.recall.result()
        return 2 * ((precision * recall) / (precision + recall + tf.keras.backend.epsilon()))
    
    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

# Load the trained model
model = tf.keras.models.load_model('./Desktop/Model/model2_balanced.keras', custom_objects={'F1Score': F1Score})

# Read data
data = pd.read_csv('./Desktop/Model/WhiteHat/Dataset.csv')

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(data.opcodes)
sequences = tokenizer.texts_to_sequences(data.opcodes)

# Sequence padding
max_length = max([len(s) for s in sequences])
sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_length)

# Encode the labels
encoder = OneHotEncoder()
labels = np.array(data.vulnerability).reshape(-1, 1)
y = encoder.fit_transform(labels).toarray()

# Split the data into train, validation, and test sets
x_train, x_temp, y_train, y_temp = train_test_split(sequences, y, test_size=0.2, shuffle=True, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

# Evaluation
scores = model.evaluate(x_test, y_test)
print('Test Loss: {:.4f}'.format(scores[0]))
print('Test Accuracy: {:.2f}%'.format(scores[1] * 100))
print('Test Precision: {:.4f}'.format(scores[2]))
print('Test Recall: {:.4f}'.format(scores[3]))
print('Test F1 Score: {:.4f}'.format(scores[4]))

# Predict the labels for the test set
y_pred = model.predict(x_test)

# Convert one-hot encoded predictions and true labels to class labels
y_test_classes = np.argmax(y_test, axis=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Compute the confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=encoder.categories_[0])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# Plotting training & validation accuracy and loss values
history = model.history.history

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['accuracy'])
plt.plot(history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
