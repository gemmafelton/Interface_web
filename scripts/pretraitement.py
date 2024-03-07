import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Charger le fichier CSV
#csv_path = '/Users/tannina/Desktop/Interface_web/corpus/detection_dh.csv'
csv_path = '/Users/gemmafelton/Desktop/Interface_web/corpus/corpus_original.csv'
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
#new_csv_path = '/Users/tannina/Desktop/Interface_web/corpus/detection_dh.csv'
new_csv_path = '/Users/gemmafelton/Desktop/Interface_web/corpus/detection_dh.csv'
df.to_csv(new_csv_path, index=False)

# Afficher les données après le prétraitement
print(df.head())

# Diviser les données en ensembles d'entraînement et de test
train_data, test_data, train_labels, test_labels = train_test_split(
    df['tweet'], df['class'], test_size=0.2, random_state=42
)


# Mélanger les ensembles d'entraînement
train_data, train_labels = shuffle(train_data, train_labels, random_state=42)

# Afficher la taille des ensembles d'entraînement et de test
print(f"Taille de l'ensemble d'entraînement : {len(train_data)}")
print(f"Taille de l'ensemble de test : {len(test_data)}")