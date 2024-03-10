import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Charger le fichier CSV
csv_path = '/Users/gemmafelton/Desktop/Interface_web/corpus/nouveau_corpus.csv'
df = pd.read_csv(csv_path)

# Remplacer les valeurs nulles par une chaîne vide
df['tweet'] = df['tweet'].fillna('')

# Prétraitement des données
def preprocess_text(text):
    # Supprimer les mentions et les liens
    text = re.sub(r'@[A-Za-z0-9_]+', '', str(text))
    text = re.sub(r'http\S+', '', text)
    # Supprimer la ponctuation et les caractères spéciaux
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Mettre en minuscules
    text = text.lower()
    return text

# Appliquer la fonction de prétraitement au texte
df['tweet'] = df['tweet'].apply(preprocess_text)

# Sauvegarder le DataFrame modifié dans un nouveau fichier CSV
new_csv_path = '/Users/gemmafelton/Desktop/Interface_web/corpus/nouveau_detection.csv'
df.to_csv(new_csv_path, index=False)

# Afficher les données après le prétraitement
print(df.head())