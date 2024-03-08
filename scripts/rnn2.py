import tensorflow as tf
import json
import pandas as pd
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Charger les données prétraitées
dataset = '/Users/gemmafelton/Desktop/Interface_web/corpus/detection_dh.csv'
df = pd.read_csv(dataset)

# Construire les étiquettes
df['label'] = df.apply(lambda row: 1 if row['hate_speech_count'] > 0 or row['offensive_language_count'] > 0 else 0, axis=1)

# Diviser les données en ensembles d'entraînement et de test
train_data, test_data, train_labels, test_labels = train_test_split(
    df['tweet'], df['label'], test_size=0.2, random_state=42
)

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(train_data)

train_sequences = tokenizer.texts_to_sequences(train_data)
test_sequences = tokenizer.texts_to_sequences(test_data)

max_sequence_length = max(len(sequence) for sequence in train_sequences)

train_sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(train_sequences, maxlen=max_sequence_length)
test_sequences_padded = tf.keras.preprocessing.sequence.pad_sequences(test_sequences, maxlen=max_sequence_length)

# Création du modèle RNN LSTM
model = tf.keras.Sequential([
    Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=max_sequence_length),
    Bidirectional(LSTM(64)),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compilation du modèle
model.compile(optimizer=Adam(learning_rate=2e-5),
              loss=BinaryCrossentropy(),
              metrics=[BinaryAccuracy(name='accuracy')])

# Entraînement du modèle
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

model.fit(train_sequences_padded, train_labels,
          epochs=5,
          batch_size=32,
          validation_split=0.2,
          callbacks=[early_stopping])

# Évaluation du modèle sur l'ensemble de test
predictions = model.predict(test_sequences_padded)
predicted_classes = [1 if pred > 0.5 else 0 for pred in predictions]

# Affichage du rapport de classification avec scikit-learn
classification_rep = classification_report(test_labels, predicted_classes)
print('\nClassification Report:\n', classification_rep)

# Enregistrer les résultats dans un fichier json
results_data = {
    "actual_labels": test_labels.tolist(),
    "predicted_labels": predictions.tolist(),
    "classification_report": classification_rep,
}

with open('/Users/gemmafelton/Desktop/Interface_web/scripts/models/rnn2_classification_results.json', 'w') as json_file:
    json.dump(results_data, json_file)
