# read data
data = pd.read_csv('./Dataset/CSV/Total/Dataset.csv')

# shauffle data
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

# Embedding layer
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 150
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim))

# LSTM layer
model.add(LSTM(units=64))

# Dense layer
num_classes = y.shape[1]  # number of unique classess
model.add(Dense(units=num_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit the model on training data
model.fit(x_train, y_train, epochs=10, batch_size=32)