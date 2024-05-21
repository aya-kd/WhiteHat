################################################### Added the dropout ####################################################

import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd


# fix random seed for reproducibility
tf.random.set_seed(7)

# read data
data = pd.read_csv('./Desktop/Model/WhiteHat/Dataset.csv')

# shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.opcodes)
sequences = tokenizer.texts_to_sequences(data.opcodes)

# Sequence padding
max_length = max([len(s) for s in sequences])
sequences = pad_sequences(sequences, maxlen=max_length)

# encode the labels
encoder = OneHotEncoder()
labels = np.array(data.vulnerability).reshape(-1, 1)
y = encoder.fit_transform(labels).toarray()

# split the data
x_train, x_test, y_train, y_test = train_test_split(sequences, y, test_size=0.2, shuffle=True, random_state=42)

# define the model
model = tf.keras.Sequential()

# Embedding layer -> input
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 150
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))

# LSTM layer -> model
model.add(LSTM(80, dropout=0.2, recurrent_dropout=0.2))

# Dense layer -> output
num_classes = y.shape[1]  # number of unique classess
model.add(Dense(units=num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'precision', 'recall', 'f1_score'])
#print(model.summary())

# Fit the model on training data
model.fit(x_train, y_train, epochs=30, batch_size=64)

# Evaluation
scores = model.evaluate(x_test, y_test)
print('Test Loss: {:.4f}'.format(scores[0]))
print('Test Accuracy: {:.2f}%'.format(scores[1] * 100))
print('Test Precision: {:.4f}'.format(scores[2]))
print('Test Recall: {:.4f}'.format(scores[3]))

model.save('./Desktop/Model/model_dropout.h5')

