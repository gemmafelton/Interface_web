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

Nous avons opté pour FastAPI comme infrastructure principale pour notre système de détection de tweets haineux en raison de plusieurs facteurs clés. En premier lieu, la réactivité et la facilité d'implémentation de FastAPI en font un choix idéal pour un projet nécessitant une analyse en temps réel de données provenant de sources dynamiques telles que Twitter. Grâce à sa gestion efficace des requêtes asynchrones basée sur ASGI, FastAPI nous permet de traiter rapidement les flux de données entrants, facilitant ainsi une détection proactive de contenu potentiellement offensant ou haineux.

En ce qui concerne l'interface utilisateur, nous avons cherché à nous rapprocher visuellement de l'interface de Twitter lors de la conception de notre interface utilisateur.
Notre approche a consisté à intégrer les technologies telles que le HTML, le JavaScript et le CSS pour créer une interface conviviale, qui emprunte visuellement à celle de Twitter. Le JavaScript a été utilisé pour dynamiser l'expérience en permettant des fonctionnalités interactives comme la mise à jour en temps réel des résultats de détection et l'accès à la barre de navigation, offrant ainsi aux utilisateurs une immersion fluide dans l'application.

Nous avons également apporté une attention minutieuse à l'aspect visuel en personnalisant l'apparence à l'aide de fichiers CSS. Inspirés des tendances de design contemporaines, nous avons intégré des éléments familiers à ceux de Twitter pour créer une interface à la fois moderne et intuitive. Cette approche vise à garantir une expérience utilisateur cohérente et plaisante, tout en favorisant la familiarité avec notre application.

En unissant la robustesse de FastAPI à une interface utilisateur inspirée de Twitter, notre objectif était de créer une expérience d'interaction positive avec notre application de détection de contenu offensant sur les réseaux sociaux. Nous avons ainsi cherché à offrir aux utilisateurs une expérience sécurisée et agréable tout en leur fournissant des fonctionnalités avancées de détection et de suivi des tweets haineux.
## Les résultats

*Les résultats (fichiers output, visualisations…) et une discussion sur ces résultats (ce que vous auriez aimé faire et ce que vous avez pu faire par exemple)*






# Problèmes... ?

Nous avons rencontré plusieurs problèmes tout au long du processus, et malheureusement, certains d'entre eux n'ont pas encore été résolus. L'un des principaux défis a été lié à l'API, où nous avons constaté des difficultés avec le chargement du modèle ainsi que des problèmes de cohérence entre l'API et le modèle pré-entraîné. Ces difficultés ont entraîné des résultats imprévisibles lors des prédictions, compromettant ainsi l'efficacité de notre système de détection.

De plus, des problèmes persistants ont été observés avec l'interface utilisateur. Selon la console, peu importe le mot saisi, il était systématiquement étiqueté comme offensif, ce qui suggère un dysfonctionnement dans le processus de classification. En outre, des erreurs telles que "undefined is labeled as undefined with a confidence of undefined" ont été rencontrées, indiquant un manque de clarté dans les prédictions et une incapacité à fournir des informations précises sur les résultats de détection.

Il convient également de noter que notre interface API a été construite sur deux modèles distincts, mais cette approche n'a pas résolu nos problèmes. En effet, malgré l'utilisation de deux modèles différents, nous avons constaté que les prédictions demeuraient souvent constantes, quelle que soit l'entrée. En résumé, ces défis techniques ont entravé notre capacité à fournir des résultats fiables et précis, soulignant la nécessité de poursuivre nos efforts pour résoudre ces problèmes et améliorer la performance globale de notre système.

Il est également important de noter que certains de ces problèmes pourraient être liés à des erreurs dans le code JavaScript de l'interface utilisateur. Étant donné que les messages d'erreur mentionnent des termes tels que "undefined" et "confiance undefined", il est probable que ces problèmes soient liés à des variables non définies ou à des erreurs de manipulation des données dans le script JavaScript. Il est donc essentiel de passer en revue et de déboguer le code JavaScript pour identifier et corriger ces erreurs potentielles, ce qui pourrait contribuer à résoudre certains des problèmes rencontrés avec l'interface utilisateur.

En résumé, ces défis techniques ont entravé notre capacité à fournir des résultats fiables et précis, soulignant la nécessité de poursuivre nos efforts pour résoudre ces problèmes et améliorer la performance globale de notre système.

# Discussion
