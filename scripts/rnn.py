import pandas as pd
import tensorflow as tf
import json
import h5py
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import model_from_json
from tensorflow.keras.callbacks import EarlyStopping

# Charger le dataset
dataset_path = "/Users/gemmafelton/Desktop/Interface_web/corpus/nouveau_detection.csv"
df = pd.read_csv(dataset_path, delimiter=',')

# Séparer les données en features (X) et labels (y)
X = df['tweet']
y = df['class']

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Size of training set:", len(X_train))
print("Size of test set:", len(X_test))

# Tokenizer les données
max_words = 10000  # Nombre maximal de mots dans le vocabulaire
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

# Convertir les textes en séquences d'entiers
X_train_sequences = tokenizer.texts_to_sequences(X_train)
X_test_sequences = tokenizer.texts_to_sequences(X_test)

# Remplir les séquences pour qu'elles aient toutes la même longueur
max_sequence_length = 40
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length, padding='post')
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length, padding='post')

# Construire le modèle RNN
embedding_dim = 64
lstm_units = 64

model = Sequential()
model.add(Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_sequence_length))
model.add(LSTM(units=lstm_units))
model.add(Dense(3, activation='softmax'))  # 3 classes pour notre corpus

# Compiler le modèle
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Define early stoppin’g
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model with early stopping
model.fit(X_train_padded, y_train, validation_data=(X_test_padded, y_test), epochs=12, batch_size=30, callbacks=[early_stopping])

# Evaluate the model
accuracy = model.evaluate(X_test_padded, y_test)[1]
print('Accuracy:', accuracy)

# Confusion Matrix
y_pred = model.predict(X_test_padded)
y_pred_classes = y_pred.argmax(axis=1)
conf_matrix = confusion_matrix(y_test, y_pred_classes)

print("Confusion Matrix:")
print(conf_matrix)

# Classification Report
class_labels = {0: 'hate speech', 1: 'offensive language', 2: 'neither'}
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=class_labels.values()))

# Save the model architecture in JSON
model_json = model.to_json()
with open("/Users/gemmafelton/Desktop/models/rnn_model.json", "w") as json_file:
    json_file.write(model_json)

# Save the model weights
model.save_weights("/Users/gemmafelton/Desktop/models/rnn_model_weights.h5")

# Save the tokenizer configuration
tokenizer_config_path = "/Users/gemmafelton/Desktop/models/tokenizer_config.json"
with open(tokenizer_config_path, 'w') as tokenizer_config_file:
    tokenizer_config_file.write(json.dumps(tokenizer.get_config()))

# Save the tokenizer vocabulary
tokenizer_vocab_path = "/Users/gemmafelton/Desktop/models/tokenizer_vocab.json"
with open(tokenizer_vocab_path, 'w') as tokenizer_vocab_file:
    tokenizer_vocab_file.write(json.dumps(tokenizer.word_index))