# Projet de fin d'année

## Les objectifs du projet

Le défi majeur posé par le discours haineux en ligne, propageant l'intolérance, la discrimination et la violence verbale tout en exploitant la liberté d'expression, constitue une préoccupation croissante à l'ère de l'essor des médias sociaux. Notre projet se fixe pour objectif le développement d'un modèle de détection efficace du discours haineux sur Twitter, visant ainsi à créer un environnement en ligne plus sécurisé et inclusif. 
S'inspirant de la thèse de Patricia Chiril [Chiril (2021)] et d'autres travaux significatifs dans ce domaine [Chiril et al. (2019), Rottger et al. (2021), Davidson et al. (2017)], notre projet contribue à l'avancement des connaissances. Dans un paysage numérique en constante évolution, l'impératif de développer des outils automatisés de détection devient essentiel pour sauvegarder la qualité du dialogue en ligne et protéger les utilisateurs contre les effets néfastes du discours haineux.


## Les données utilisées

L'auteur d'origine, [Tom Davidson](https://huggingface.co/datasets/tdavidson/hate_speech_offensive), a utilisé Twitter API afin d'extraire les tweets anglais en utilisant plusieurs termes spécifiques rélévant du discours haineux ou du langage offensive. Les tweets ont ensuite été annotés manuellement par plusieurs annotateurs.

Le [dataset](https://github.com/gemmafelton/Interface_web/blob/main/corpus) est sous licence MIT. Les données sont téléchargéables sous forme csv et contiennent 22,660 lignes sous forme de tweets. Les colonnes qui se trouvent dans le tableau sont : count (*nombre total d'annotations*), hate_speech_count (*nombre d'annotations classifiant un tweet comme haineux*), offensive_language_count (*nombre d'annotations classifiant un tweet comme du langage offensive*), neither_count (*nombre d'annotations classifiant un tweet comme non haineux non offensive*).

Nous avons fait les prétraitements suivants sur le corpus : 
* Supprimer les mentions
* Supprimer les liens
* Supprimer la ponctuation et les caractères spéciaux
* Mettre en minuscules

Nous avons ensuite réduite la taille du corpus à 20%, soit 5,308 lignes, afin de pouvoir faire faciliter l'entrainement du modèle.

Puis nous avons divisé le corpus en entraînement et test : 
* Taille de l'ensemble d'entraînement : 3965
* Taille de l'ensemble de test : 992


## La méthodologie
*La méthodologie (comment vous vous êtes répartis le travail, comment vous avez identifié les problèmes et les avez résolus, différentes étapes du projet…)*

La concrétisation de ce projet a exigé un investissement significatif en termes de temps et d'essais, s'avérant être une démarche longue et complexe. La répartition des tâches au sein de l'équipe s'est effectuée en trois parties : Gemma s'est concentrée sur la création et les essais des modèles, Tannina a pris en charge le développement et les essais de l'API ainsi que de l'interface, tandis que la rédaction du document technique a été une collaboration entre nous deux. Cette approche a permis une spécialisation efficace dans chaque domaine, maximisant ainsi notre productivité et la qualité des résultats dans l'ensemble du projet.

La première partie du projet, dédiée aux modèles, a été marqué par d'importants défis, nous confrontant à la réalité de la puissance de nos ordinateurs. Initialement confiantes dans leurs capacités, le projet a révélé la nécessité de faire face à des limitations. Avec un corpus substantiel, nos machines ont montré leurs limites, nous obligeant à prendre d'autres mesures non prévues après une période prolongée de tentatives. La réduction conséquente de la taille du corpus, à seulement 20% de sa taille originale, a finalement débloqué le processus d'entraînement des modèles, nous permettant ainsi d'obtenir des résultats et d'entamer les phases cruciales d'entraînement et de modifications des paramètres.


# I. Reseau de neurones
## Les implémentations

*L’implémentation ou les implémentations (modélisation le cas échéant, modules et/ou API utilisés, différents langages le cas échéant)*


max length=40 : la longueur moyenne d'un tweet est d'environ 40 caractères
modification de batch size et epochs -> batch size = 32 ; epochs = 16
ajout de early stopping=3 donc ça s'arrete si le loss ne s'améliore pas pour 3 epochs consecutives 


## Les résultats

*Les résultats (fichiers output, visualisations…) et une discussion sur ces résultats (ce que vous auriez aimé faire et ce que vous avez pu faire par exemple)*




# II. Web Interfaces
## Les implémentations

*L’implémentation ou les implémentations (modélisation le cas échéant, modules et/ou API utilisés, différents langages le cas échéant)*

## Les résultats

*Les résultats (fichiers output, visualisations…) et une discussion sur ces résultats (ce que vous auriez aimé faire et ce que vous avez pu faire par exemple)*


# Discussion