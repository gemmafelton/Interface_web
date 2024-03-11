import pandas as pd
import tensorflow as tf
import json
import h5py
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the model
model_path = "/Users/gemmafelton/Desktop/models/rnn_model.json"
weights_path = "/Users/gemmafelton/Desktop/models/rnn_model_weights.h5"
loaded_model = tf.keras.models.model_from_json(open(model_path, "r").read())
loaded_model.load_weights(weights_path)

# Load the tokenizer configuration
tokenizer_config_path = "/Users/gemmafelton/Desktop/models/tokenizer_config.json"
with open(tokenizer_config_path, 'r') as tokenizer_config_file:
    tokenizer_config = json.load(tokenizer_config_file)

# Load the tokenizer vocabulary
tokenizer_vocab_path = "/Users/gemmafelton/Desktop/models/tokenizer_vocab.json"
with open(tokenizer_vocab_path, 'r') as tokenizer_vocab_file:
    tokenizer_vocab = json.load(tokenizer_vocab_file)

# Reconstruct the tokenizer
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.__dict__.update(tokenizer_config)
tokenizer.word_index = tokenizer_vocab


# Define max_sequence_length (replace with the actual value used during training)
max_sequence_length = 40

# Faire des prédictions sur un nouvel exemple
input_tweet = input("Enter the tweet that you'd like to analyse: ")
input_sequence = tokenizer.texts_to_sequences([input_tweet])
input_padded = pad_sequences(input_sequence, maxlen=max_sequence_length, padding='post')

output = loaded_model.predict(input_padded)
print(output)

# Assuming 'output' is a 2D array with shape (num_samples, num_classes)
probabilities_list = output

class_labels = {0: 'hate speech', 1: 'offensive language', 2: 'neither'}

print("Probabilities for each class:")
for i, prob in enumerate(probabilities_list[0]):
    print(f"Probability that this tweet is {class_labels[i]}: {prob * 100:.2f}%")