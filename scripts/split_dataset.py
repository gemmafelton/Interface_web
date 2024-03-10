import pandas as pd

# Charger votre dataset
dataset_path = "/Users/gemmafelton/Desktop/Interface_web/corpus/corpus_original.csv"
df = pd.read_csv(dataset_path, delimiter=',')

# Choisissez le pourcentage de lignes que vous souhaitez conserver
pourcentage_a_garder = 20  # Par exemple, pour conserver 50% des lignes

# Sélectionnez un échantillon aléatoire
nouveau_df = df.sample(frac=pourcentage_a_garder / 100, random_state=42)

# Enregistrez le nouveau dataset dans un fichier CSV
nouveau_chemin = "/Users/gemmafelton/Desktop/Interface_web/corpus/nouveau_corpus.csv"
nouveau_df.to_csv(nouveau_chemin, index=False)