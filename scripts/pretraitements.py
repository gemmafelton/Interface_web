# Import necessary libraries
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Load the CSV file
csv_path = '/Users/gemmafelton/Desktop/Interface_web/corpus/nouveau_corpus.csv'
df = pd.read_csv(csv_path)

# Replace null values with an empty string
df['tweet'] = df['tweet'].fillna('')

# Data preprocessing function
def preprocess_text(text):
    # Remove mentions and links
    text = re.sub(r'@[A-Za-z0-9_]+', '', str(text))
    text = re.sub(r'http\S+', '', text)
    # Remove punctuation and special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Convert to lowercase
    text = text.lower()
    return text

# Apply the preprocessing function to the text column
df['tweet'] = df['tweet'].apply(preprocess_text)

# Save the modified DataFrame to a new CSV file
new_csv_path = '/Users/gemmafelton/Desktop/Interface_web/corpus/nouveau_detection.csv'
df.to_csv(new_csv_path, index=False)

# Display the data after preprocessing
print(df.head())
