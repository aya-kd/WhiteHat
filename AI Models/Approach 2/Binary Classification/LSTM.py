# Import libraries
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Custom F1 Score Metric
class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super(F1Score, self).__init__(name=name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()
        
    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.cast(tf.greater_equal(y_pred, 0.5), tf.float32)
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
data = pd.read_csv('opcodes_vulnerable.csv')

# Ensure 'opcodes' and 'vulnerable' columns are present
if 'opcodes' not in data.columns or 'vulnerable' not in data.columns:
    raise ValueError("The DataFrame does not contain required columns 'opcodes' and 'vulnerable'.")

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)


# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.opcodes)
sequences = tokenizer.texts_to_sequences(data.opcodes)

#############################################
############## Sequence padding #############
#############################################
max_length = 5000
sequences = pad_sequences(sequences, maxlen=max_length)

# Encode the labels
# Encode the labels (without one-hot encoding)
labels = np.array(data.vulnerable)
# Reshape to make it compatible with binary classification
y = labels.reshape(-1, 1)

# Split the data into train, validation, and test sets
x_train, x_temp, y_train, y_temp = train_test_split(sequences, y, test_size=0.2, shuffle=True, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

# Calculate class weights
class_weights = compute_class_weight(class_weight = "balanced", classes= np.unique(data.vulnerable), y= data.vulnerable)
print(class_weights)
class_weights_dict = dict(enumerate(class_weights))


# Define the model
model = tf.keras.Sequential()

# Embedding layer
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))

# Adding multiple LSTM layers with dropout
model.add(LSTM(64, return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(64))

# Dense layer
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), F1Score()]
)



# Define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath='./Desktop/Model/model2_binary.keras', save_best_only=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.0001)
]



# Fit the model on training data with class weights
history = model.fit(
    x_train, 
    y_train, 
    epochs=30, 
    batch_size=64, 
    validation_data=(x_val, y_val),
    class_weight=class_weights_dict,
    callbacks=callbacks
)


# Print model summary
model.summary()


# Evaluation
scores = model.evaluate(x_test, y_test)
print('Test Loss: {:.4f}'.format(scores[0]))
print('Test Accuracy: {:.2f}%'.format(scores[1] * 100))
print('Test Precision: {:.4f}'.format(scores[2]))
print('Test Recall: {:.4f}'.format(scores[3]))
print('Test F1 Score: {:.4f}'.format(scores[4]))

# Predict the labels for the test set
y_pred = model.predict(x_test)
y_pred_classes = (y_pred > 0.5).astype(int)

# Compute the confusion matrix
cm = confusion_matrix(y_test, y_pred_classes)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

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
