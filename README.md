# Projet de fin d'année


# I. Reseau de neurones
## Les objectifs du projet

parle de patricia chiril et les autres articles qu'on a vu

## Les données utilisées




origine, format, statut juridique et les traitements opérés sur celles-ci


L'auteur d'origine, [Tom Davidson](https://huggingface.co/datasets/tdavidson/hate_speech_offensive), a utilisé Twitter API afin d'extraire les tweets anglais en utilisant plusieurs termes spécifiques rélévant du discours haineux ou du langage offensive. Les tweets ont ensuite été annotés manuellement par plusieurs annotateurs.

Le [dataset](https://github.com/gemmafelton/Interface_web/blob/main/corpus) est sous licence MIT. Les données sont téléchargéables sous forme csv et contiennent 22,660 lignes sous forme de tweets. Les colonnes qui se trouvent dans le tableau sont : count (*nombre total d'annotations*), hate_speech_count (*nombre d'annotations classifiant un tweet comme haineux*), offensive_language_count (*nombre d'annotations classifiant un tweet comme du langage offensive*), neither_count (*nombre d'annotations classifiant un tweet comme non haineux non offensive*)

Nous avons fait les prétraitements suivants sur le corpus : 
* Supprimer les mentions et les liens
* Supprimer la ponctuation et les caractères spéciaux
* Mettre en minuscules

Puis on a divisé le corpus en entraînement et test : 
* Taille de l'ensemble d'entraînement : 19826
* Taille de l'ensemble de test : 4957




## La méthodologie
La méthodologie (comment vous vous êtes répartis le travail, comment vous avez identifié les problèmes et les avez résolus, différentes étapes du projet…)

## Les implémentations

L’implémentation ou les implémentations (modélisation le cas échéant, modules et/ou API utilisés, différents langages le cas échéant)

## Les résultats

Les résultats (fichiers output, visualisations…) et une discussion sur ces résultats (ce que vous auriez aimé faire et ce que vous avez pu faire par exemple)








# II. Web Interfaces
