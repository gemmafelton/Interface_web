import pandas as pd
import tensorflow as tf
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import model_from_json


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
max_sequence_length = 256
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

# Entraîner le modèle
model.fit(X_train_padded, y_train, validation_data=(X_test_padded, y_test), epochs=14, batch_size=60)

# Évaluer le modèle
accuracy = model.evaluate(X_test_padded, y_test)[1]
print('Accuracy:', accuracy)

# Sauvegarder l'architecture du modèle en JSON
model_json = model.to_json()
with open("/Users/gemmafelton/Desktop/Interface_web/scripts/models/rnn_model.json", "w") as json_file:
    json_file.write(model_json)

# Sauvegarder les poids du modèle
model.save_weights("/Users/gemmafelton/Desktop/Interface_web/scripts/models/rnn_model_weights.h5")

# Charger le modèle sauvegardé
loaded_model = tf.keras.models.model_from_json(model_json)
loaded_model.load_weights("/Users/gemmafelton/Desktop/Interface_web/scripts/models/rnn_model_weights.h5")

# Faire des prédictions sur un nouvel exemple
input_tweet = input("Enter the tweet that you'd like to analyse: ")
input_sequence = tokenizer.texts_to_sequences([input_tweet])
input_padded = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post')

output = loaded_model.predict(input_padded)
print(output)

# Assuming 'output' is a 2D array with shape (num_samples, num_classes)
probabilities = tf.nn.softmax(output)
probabilities_list = probabilities.numpy()

class_labels = {0: 'hate speech', 1: 'offensive language', 2: 'neither'}

print("Probabilities for each class:")
for i, prob in enumerate(probabilities_list[0]):
    print(f"Probability that this tweet is {class_labels[i]}: {prob * 100:.2f}%")