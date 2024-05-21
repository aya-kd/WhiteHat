# Import libraries
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fix random seed for reproducibility
tf.random.set_seed(7)

# Read data
data = pd.read_csv('./Desktop/Model/WhiteHat/Dataset.csv')

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.opcodes)
sequences = tokenizer.texts_to_sequences(data.opcodes)

# Sequence padding
max_length = max([len(s) for s in sequences])
sequences = pad_sequences(sequences, maxlen=max_length)

# Encode the labels
encoder = OneHotEncoder()
labels = np.array(data.vulnerability).reshape(-1, 1)
y = encoder.fit_transform(labels).toarray()

# Split the data into train, validation, and test sets
x_train, x_temp, y_train, y_temp = train_test_split(sequences, y, test_size=0.3, shuffle=True, random_state=42)
x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, shuffle=True, random_state=42)

# Calculate class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(data.vulnerability), y=data.vulnerability)
class_weights_dict = dict(enumerate(class_weights))

# Define the model
model = tf.keras.Sequential()

# Embedding layer -> input
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length))

# LSTM layer -> model
model.add(LSTM(64))

# Dense layer -> output
num_classes = y.shape[1]  # number of unique classes
model.add(Dense(units=num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Print model summary
model.summary()

# Define callbacks
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
    tf.keras.callbacks.ModelCheckpoint(filepath='./Desktop/Model/model2_balanced.keras', save_best_only=True)
]

# Load the latest model if exists
try:
    model = tf.keras.models.load_model('./Desktop/Model/model2_balanced.keras')
    print("Loaded model from checkpoint.")
except:
    print("No checkpoint found, starting training from scratch.")

# Fit the model on training data with class weights
history = model.fit(x_train, y_train, epochs=39, batch_size=64, validation_data=(x_val, y_val), class_weight=class_weights_dict, callbacks=callbacks)

# Evaluation
scores = model.evaluate(x_test, y_test)
print('Test Loss: {:.4f}'.format(scores[0]))
print('Test Accuracy: {:.2f}%'.format(scores[1] * 100))
print('Test Precision: {:.4f}'.format(scores[2]))
print('Test Recall: {:.4f}'.format(scores[3]))

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
