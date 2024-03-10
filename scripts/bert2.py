import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
import numpy as np

dataset_path = "/Users/gemmafelton/Desktop/Interface_web/corpus/nouveau_detection.csv"
df = pd.read_csv(dataset_path, delimiter=',')

X = df['tweet']
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(list(X_train.values), truncation=True, padding=True, max_length=128, return_tensors='tf')
test_encodings = tokenizer(list(X_test.values), truncation=True, padding=True, max_length=128, return_tensors='tf')

model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

train_input_ids = train_encodings['input_ids'].numpy()
train_attention_masks = train_encodings['attention_mask'].numpy()

test_input_ids = test_encodings['input_ids']
test_attention_masks = test_encodings['attention_mask']

test_input_ids = test_input_ids.numpy()
test_attention_masks = test_attention_masks.numpy()

model.fit([train_input_ids, train_attention_masks], y_train, validation_data=([test_input_ids, test_attention_masks], y_test), epochs=14, batch_size=100)

_, accuracy = model.evaluate([test_input_ids, test_attention_masks], y_test)
print('Accuracy:', accuracy)

model.save_pretrained("/Users/gemmafelton/Desktop/Interface_web/scripts/models/bert_model")

# Load the model and tokenizer from the saved directory
loaded_model = TFBertForSequenceClassification.from_pretrained("/Users/gemmafelton/Desktop/Interface_web/scripts/models/bert_model")
loaded_tokenizer = BertTokenizer.from_pretrained("/Users/gemmafelton/Desktop/Interface_web/scripts/models/bert_model")

input_tweet = input("Enter the tweet that you'd like to analyze: ")

input_encoding = loaded_tokenizer(input_tweet, truncation=True, padding=True, max_length=128, return_tensors='tf')

input_ids = input_encoding['input_ids'].numpy()
attention_masks = input_encoding['attention_mask'].numpy()

# Make predictions
output = loaded_model.predict([input_ids, attention_masks])

# Assuming 'output' is a 2D array with shape (num_samples, num_classes)
probabilities = tf.nn.softmax(output[0])
probabilities_list = probabilities.numpy()[0]

class_labels = {0: 'hate speech', 1: 'offensive language', 2: 'neither'}

print("Probabilities for each class:")
for i, prob in enumerate(probabilities_list):
    print(f"Probability that this tweet is {class_labels[i]}: {prob * 100:.2f}%")
