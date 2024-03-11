# Import necessary libraries
import pandas as pd
import tensorflow as tf
import numpy as np
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset from a CSV file
dataset_path = "/Users/gemmafelton/Desktop/Interface_web/corpus/nouveau_detection.csv"
df = pd.read_csv(dataset_path, delimiter=',')

# Split the dataset into training and testing sets
X = df['tweet']
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize the training and testing sets
train_encodings = tokenizer(list(X_train.values), truncation=True, padding=True, max_length=40, return_tensors='tf')
test_encodings = tokenizer(list(X_test.values), truncation=True, padding=True, max_length=40, return_tensors='tf')

# Build and compile the BERT model
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Extract input_ids and attention_masks for the training set
train_input_ids = train_encodings['input_ids'].numpy()
train_attention_masks = train_encodings['attention_mask'].numpy()

# Extract input_ids and attention_masks for the testing set
test_input_ids = test_encodings['input_ids']
test_attention_masks = test_encodings['attention_mask']
test_input_ids = test_input_ids.numpy()
test_attention_masks = test_attention_masks.numpy()

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the BERT model with early stopping
model.fit(
    [train_input_ids, train_attention_masks],
    y_train,
    validation_data=([test_input_ids, test_attention_masks], y_test),
    epochs=12,
    batch_size=30,
    callbacks=[early_stopping]
)

# Evaluate the BERT model on the test set
_, accuracy = model.evaluate([test_input_ids, test_attention_masks], y_test)
print('Accuracy:', accuracy)

# Make predictions on the test set
predictions = model.predict([test_input_ids, test_attention_masks])
predicted_labels = np.argmax(predictions[0], axis=1)

# Evaluate classification report
print('Classification Report:')
print(classification_report(y_test, predicted_labels))

# Save the trained BERT model and tokenizer in the same directory
model.save_pretrained("/Users/gemmafelton/Desktop/models/bert_model")
tokenizer.save_pretrained("/Users/gemmafelton/Desktop/models/bert_model")