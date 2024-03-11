# Import necessary libraries
import pandas as pd

# Load your dataset from a CSV file
dataset_path = "/Users/gemmafelton/Desktop/Interface_web/corpus/corpus_original.csv"
df = pd.read_csv(dataset_path, delimiter=',')

# Choose the percentage of rows you want to keep
percentage_to_keep = 20  # For example, to keep 20% of the rows

# Select a random sample
new_df = df.sample(frac=percentage_to_keep / 100, random_state=42)

# Save the new dataset to a CSV file
new_path = "/Users/gemmafelton/Desktop/Interface_web/corpus/nouveau_corpus.csv"
new_df.to_csv(new_path, index=False)