from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Load the trained BERT model and tokenizer from the saved directory
loaded_model = TFBertForSequenceClassification.from_pretrained("/Users/gemmafelton/Desktop/models/bert_model")
loaded_tokenizer = BertTokenizer.from_pretrained("/Users/gemmafelton/Desktop/models/bert_model")

# Input tweet for analysis
input_tweet = input("Enter the tweet that you'd like to analyze:")

# Tokenize the input tweet using the loaded tokenizer
input_encoding = loaded_tokenizer.encode_plus(input_tweet, truncation=True, padding=True, max_length=40, return_tensors='tf')

# Extract input_ids and attention_masks for the input tweet
input_ids = input_encoding['input_ids'].numpy()
attention_masks = input_encoding['attention_mask'].numpy()

# Make predictions on the input tweet
output = loaded_model.predict([input_ids, attention_masks])
probabilities = tf.nn.softmax(output[0])
probabilities_list = probabilities.numpy()[0]

# Class labels mapping
class_labels = {0: 'hate speech', 1: 'offensive language', 2: 'neither'}

# Print probabilities for each class
print("Probabilities for each class:")
for i, prob in enumerate(probabilities_list):
    print(f"Probability that this tweet is {class_labels[i]}: {prob * 100:.2f}%")