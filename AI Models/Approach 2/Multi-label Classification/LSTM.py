import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import tensorflow as tf

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

# Fix random seed for reproducibility
tf.random.set_seed(7) 

# Read data
data = pd.read_csv('opcodes_vulnerabilities_filtered.csv')

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Split the vulnerabilities column into lists of vulnerabilities
data['vulnerabilities'] = data['vulnerabilities'].apply(lambda x: x.split() if isinstance(x, str) else [])

# Tokenize the sequences of opcodes
tokenizer = Tokenizer() 
tokenizer.fit_on_texts(data.opcodes)
sequences = tokenizer.texts_to_sequences(data.opcodes)

# Pad sequences
max_length = 5000
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')

# Binarize the labels using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
binary_labels = mlb.fit_transform(data['vulnerabilities'])

# Split the data into train, validation, and test sets
x_train_val, x_test, y_train_val, y_test = train_test_split(padded_sequences, binary_labels, test_size=0.2, shuffle=True, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, shuffle=True, random_state=42)  # 0.25 * 0.8 = 0.2

# Flatten the list of lists
all_vulnerabilities = [vuln for sublist in data['vulnerabilities'] for vuln in sublist]

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(all_vulnerabilities), y=all_vulnerabilities)
class_weights_dict = dict(enumerate(class_weights))

# Define the model
model = Sequential()

# Embedding layer
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))

# Adding multiple LSTM layers with dropout
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))

# Dense layer
num_classes = binary_labels.shape[1]  # number of unique classes
model.add(Dense(units=128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=num_classes, activation='sigmoid'))  # Sigmoid activation for multi-label classification

# Compile the model
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy',  # Binary cross-entropy for multi-label classification
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), F1Score()]
)

# Print model summary
model.summary()

# Define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath='./Desktop/Model/Multi2.keras', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
]

# Fit the model on training data with class weights
history = model.fit(
    x_train, 
    y_train, 
    epochs=50, 
    batch_size=64, 
    validation_data=(x_val, y_val), 
    class_weight=class_weights_dict, 
    callbacks=callbacks
)

# Evaluation
scores = model.evaluate(x_test, y_test)
print('Test Loss: {:.4f}'.format(scores[0]))
print('Test Accuracy: {:.2f}%'.format(scores[1] * 100))
print('Test Precision: {:.4f}'.format(scores[2]))
print('Test Recall: {:.4f}'.format(scores[3]))
print('Test F1 Score: {:.4f}'.format(scores[4]))

# Predict the labels for the test set
y_pred = model.predict(x_test)

# Convert probabilities to binary predictions (thresholding at 0.5)
y_pred_binary = (y_pred > 0.5).astype(int)
y_test_binary = y_test  # Assuming y_test is already in binary format

# Compute the confusion matrix for each label
cms = multilabel_confusion_matrix(y_test_binary, y_pred_binary)

# Display the confusion matrix for each label
for i, cm in enumerate(cms):
    print(f'Confusion Matrix for Label {i}:')
    print(cm)

# Plotting training & validation accuracy and loss values
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.show()
