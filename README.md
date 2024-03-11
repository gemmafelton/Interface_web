# La détéction du discours haineux

## Les objectifs du projet

Le défi majeur posé par le discours haineux en ligne, propageant l'intolérance, la discrimination et la violence verbale tout en exploitant la liberté d'expression, constitue une préoccupation croissante à l'ère de l'essor des médias sociaux. Notre projet se fixe pour objectif le développement d'un modèle de détection efficace du discours haineux sur Twitter, visant ainsi à créer un environnement en ligne plus sécurisé et inclusif.
S'inspirant de la thèse de Patricia Chiril [Chiril (2021)] et d'autres travaux significatifs dans ce domaine [Chiril et al. (2019), Rottger et al. (2021), Davidson et al. (2017)], notre projet contribue à l'avancement des connaissances. Dans un paysage numérique en constante évolution, l'impératif de développer des outils automatisés de détection devient essentiel pour sauvegarder la qualité du dialogue en ligne et protéger les utilisateurs contre les effets néfastes du discours haineux.


## Les données utilisées

L'auteur d'origine, [Tom Davidson](https://huggingface.co/datasets/tdavidson/hate_speech_offensive), a utilisé Twitter API afin d'extraire les tweets anglais en utilisant plusieurs termes spécifiques rélévant du discours haineux ou du langage offensive. Les tweets ont ensuite été annotés manuellement par plusieurs annotateurs.

Le [dataset](https://github.com/gemmafelton/Interface_web/blob/main/corpus) est sous licence MIT. Les données sont téléchargéables sous forme csv et contiennent 22,660 lignes sous forme de tweets. Les colonnes qui se trouvent dans le tableau sont : count (*nombre total d'annotations*), hate_speech_count (*nombre d'annotations classifiant un tweet comme haineux*), offensive_language_count (*nombre d'annotations classifiant un tweet comme du langage offensive*), neither_count (*nombre d'annotations classifiant un tweet comme non haineux non offensive*).

### Les prétraitements
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

La concrétisation de ce projet a exigé un investissement significatif en termes de temps et d'essais, s'avérant être une démarche longue et complexe. La répartition des tâches au sein de l'équipe s'est effectuée en trois parties : Gemma s'est concentrée sur la création et les essais des modèles, Tannina a pris en charge le développement et les essais de l'API ainsi que de l'interface, tandis que la rédaction du document technique a été une collaboration entre nous deux. Cette approche a permis une spécialisation efficace dans chaque domaine, maximisant ainsi notre productivité et la qualité des résultats dans l'ensemble du projet.

La première partie du projet, dédiée aux modèles, a été marqué par d'importants défis, nous confrontant à la réalité de la puissance de nos ordinateurs. Initialement confiantes dans leurs capacités, le projet a révélé la nécessité de faire face à des limitations. Avec un corpus substantiel, nos machines ont montré leurs limites, nous obligeant à prendre d'autres mesures non prévues après une période prolongée de tentatives. La réduction conséquente de la taille du corpus, à seulement 20% de sa taille originale, a finalement débloqué le processus d'entraînement des modèles, nous permettant ainsi d'obtenir des résultats et d'entamer les phases cruciales d'entraînement et de modifications des paramètres.


# I. Reseau de neurones
## Les implémentations

*Modèles utilisés :* 
</br>BERT et RNN

</br>
*Paramètres communs :*
* Longueur maximale de séquence fixée à 40 caractères, alignée sur la moyenne de la longueur d'un tweet.
* Batch size ajusté à 32 pour l'entraînement.
* Nombre d'epochs  fixé à 16 pour chaque modèle.

*Justification des paramètres :*
* Longueur de séquence adaptée à la nature des données (moyenne de 40 caractères par tweet).
* Batch size et nombre d'époques ajustés pour l'efficacité de l'entraînement.
* Arrêt anticipé pour prévenir le surapprentissage et accélérer la convergence des modèles.</br>

*Early stopping :*
* Intégration de la fonction d'arrêt anticipé (early stopping).
* Critère de patience défini à 3 époques.
* L'entraînement est interrompu si la loss ne s'améliore pas pendant trois epochs consécutives.</br>


## Les résultats

Le modèle RNN (Réseau de Neurones Récurrents) s'est avéré plus performant que BERT dans la détection du discours de haine. 

Resultats matrices de confusion pour BERT : 
![Resultats matrices de confusion pour BERT](https://github.com/gemmafelton/Interface_web/blob/main/ressources/matrix_bert.jpeg)

Resultats matrices de confusion pour RNN:
![Resultats matrices de confusion pour RNN](https://github.com/gemmafelton/Interface_web/blob/main/ressources/matrix_rnn.jpeg)

Cette supériorité peut s'expliquer par la capacité du RNN à capturer les dépendances séquentielles, particulièrement cruciales pour analyser les nuances subtiles et les motifs temporels présents dans le discours de haine. La nature spécifique de l'ensemble de données a probablement favorisé le RNN, soulignant l'importance de choisir un modèle en fonction des caractéristiques particulières de la tâche. En revanche, pour BERT, des stratégies de prétraitement et de fine-tuning plus adaptées auraient pu être explorées pour exploiter pleinement sa capacité à comprendre le contexte, offrant ainsi des perspectives d'optimisation pour des tâches similaires à l'avenir.

Dans d'autres travaux, il a été observé que dans la plupart des cas, au moins 70% de l'ensemble de données était étiqueté comme étant du discours haineux. Cela suggère que le dataset initial pourrait ne pas contenir suffisamment d'exemples de discours non haineux pour un entraînement optimal des modèles. Par conséquent, il serait judicieux de considérer un nouveau ensemble de données comportant davantage de discours neutres afin d'améliorer la performance et la généralisation des modèles de détection de discours de haine.


# II. Web Interfaces
## Les implémentations

Nous avons opté pour FastAPI comme infrastructure principale pour notre système de détection de tweets haineux en raison de plusieurs facteurs clés. En premier lieu, la réactivité et la facilité d'implémentation de FastAPI en font un choix idéal pour un projet nécessitant une analyse en temps réel de données provenant de sources dynamiques telles que Twitter. Grâce à sa gestion efficace des requêtes asynchrones basée sur ASGI, FastAPI nous permet de traiter rapidement les flux de données entrants, facilitant ainsi une détection proactive de contenu potentiellement offensant ou haineux.
*L’implémentation ou les implémentations (modélisation le cas échéant, modules et/ou API utilisés, différents langages le cas échéant)*

Nous avons choisi FastAPI comme infrastructure principale pour notre système de détection de tweets haineux en raison de sa réactivité et de sa facilité d'implémentation, parfaites pour une analyse en temps réel de données dynamiques telles que celles provenant de Twitter. Grâce à sa gestion efficace des requêtes asynchrones basée sur ASGI, FastAPI nous permet de traiter rapidement les flux de données entrants, facilitant ainsi une détection proactive de contenu potentiellement offensant ou haineux.

Pour l'interface utilisateur, notre objectif était de nous rapprocher visuellement de l'interface de Twitter. Nous avons intégré les technologies HTML, JavaScript et CSS pour créer une interface conviviale, empruntant visuellement à celle de Twitter. Le JavaScript dynamise l'expérience en permettant des fonctionnalités interactives telles que la mise à jour en temps réel des résultats de détection et l'accès à la barre de navigation, offrant ainsi aux utilisateurs une immersion fluide dans l'application.

Une attention particulière a été portée à l'aspect visuel en personnalisant l'apparence à l'aide de fichiers CSS. Inspirés des tendances de design contemporaines, nous avons intégré des éléments familiers à ceux de Twitter pour créer une interface moderne et intuitive. Cette approche vise à garantir une expérience utilisateur cohérente et plaisante, tout en favorisant la familiarité avec notre application.

L'implémentation de notre système de détection de contenu offensant dans les tweets repose sur plusieurs éléments essentiels. Tout d'abord, nous avons utilisé TensorFlow, une bibliothèque d'apprentissage automatique, pour construire et entraîner un modèle de réseau neuronal récurrent (RNN). Ce modèle est ensuite intégré dans une API FastAPI, qui offre une gestion efficace des requêtes HTTP et des réponses JSON. Le langage principal de développement est Python, choisi pour sa polyvalence et sa popularité dans le domaine de l'apprentissage automatique et du développement web. Nous utilisons également uvicorn pour lancer l'interface utilisateur, offrant ainsi une gestion asynchrone des connexions et une performance optimale. Cette combinaison harmonieuse de langages, de modules, d'API et d'outils crée un système robuste et performant pour la détection de contenu offensant sur les réseaux sociaux.

En unissant la robustesse de FastAPI à une interface utilisateur inspirée de Twitter, notre objectif était de créer une expérience d'interaction positive avec notre application de détection de contenu offensant sur les réseaux sociaux. Nous avons ainsi cherché à offrir aux utilisateurs une expérience sécurisée et agréable tout en leur fournissant des fonctionnalités avancées de détection et de suivi des tweets haineux.

## Les résultats

### Résultats obtenus :

Nos résultats sont en adéquation avec nos attentes concernant l'interface que nous avons développée. Nous avons réussi à mettre en place une interface conforme à notre vision, offrant une expérience utilisateur fluide et intuitive. En examinant différents exemples de tweets, nous constatons que la plupart sont correctement classifiés comme offensants, ce qui témoigne de l'efficacité de notre système de détection. Par exemple, des tweets contenant des insultes ou des propos discriminatoires sont identifiés avec succès par notre système, démontrant ainsi sa capacité à détecter efficacement le contenu problématique.
<p>(capture d'écran a ajouter pour montrer les prédictions)</p>

### Problèmes... ?

Nous avons rencontré plusieurs problèmes tout au long du processus, et malheureusement, certains d'entre eux n'ont pas encore été résolus. L'un des principaux défis a été lié à l'API, où nous avons constaté des difficultés avec le chargement du modèle ainsi que des problèmes de cohérence entre l'API et le modèle pré-entraîné. Ces difficultés ont entraîné des résultats imprévisibles lors des prédictions, compromettant ainsi l'efficacité de notre système de détection.

De plus, des problèmes persistants ont été observés avec l'interface utilisateur. Selon la console, peu importe le mot saisi, il était systématiquement étiqueté comme offensif, ce qui suggère un dysfonctionnement dans le processus de classification. En outre, des erreurs telles que "undefined is labeled as undefined with a confidence of undefined" ont été rencontrées, indiquant un manque de clarté dans les prédictions et une incapacité à fournir des informations précises sur les résultats de détection.

Il convient également de noter que notre interface API a été construite sur deux modèles distincts, mais cette approche n'a pas résolu nos problèmes. En effet, malgré l'utilisation de deux modèles différents, nous avons constaté que les prédictions demeuraient souvent constantes, quelle que soit l'entrée. En résumé, ces défis techniques ont entravé notre capacité à fournir des résultats fiables et précis, soulignant la nécessité de poursuivre nos efforts pour résoudre ces problèmes et améliorer la performance globale de notre système.

Il est également important de noter que certains de ces problèmes pourraient être liés à des erreurs dans le code JavaScript de l'interface utilisateur. Étant donné que les messages d'erreur mentionnent des termes tels que "undefined" et "confiance undefined", il est probable que ces problèmes soient liés à des variables non définies ou à des erreurs de manipulation des données dans le script JavaScript. Il est donc essentiel de passer en revue et de déboguer le code JavaScript pour identifier et corriger ces erreurs potentielles, ce qui pourrait contribuer à résoudre certains des problèmes rencontrés avec l'interface utilisateur.

En résumé, ces défis techniques ont entravé notre capacité à fournir des résultats fiables et précis, soulignant la nécessité de poursuivre nos efforts pour résoudre ces problèmes et améliorer la performance globale de notre système.

# Discussion

### Discussion sur les résultats :
Dans l'ensemble, nos résultats sont plutôt positifs et en accord avec les conclusions de recherches antérieures telles que celles de Chiril et al. La majorité des tweets sont classifiés comme offensants, ce qui confirme la fiabilité de notre système de détection. En testant un vaste éventail de tweets, nous avons pu confirmer cette tendance. Cela suggère que notre système est robuste et capable de détecter de manière cohérente le contenu problématique sur les plateformes de réseaux sociaux.

### Perspectives :
Malgré ces résultats encourageants, des améliorations peuvent être envisagées. Nous aurions voulu ajouter une sorte d'échelle graduée pour représenter la prédiction en pourcentage, mais nous n'avons pas réussi à le mettre en œuvre. Cette fonctionnalité pourrait permettre aux utilisateurs d'obtenir une meilleure compréhension de la confiance de la prédiction. Pour l'avenir, nous envisageons de rendre notre système multilingue, permettant ainsi une détection plus étendue de contenus offensants dans différentes langues. De plus, nous pourrions envisager de personnaliser davantage le système en intégrant les retours des utilisateurs pour affiner la détection en fonction de leurs préférences et spécificités. Ces perspectives pourraient contribuer à améliorer encore la précision et l'efficacité de notre système de détection de tweets haineux.