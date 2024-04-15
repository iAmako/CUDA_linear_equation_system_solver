CUDA_linear_equation_system_solver
=======

## Utilisation
--> Voir le notebook "demo_notebook"

## Avant de push le notebook 
Avant de push le notebook il faut penser à en retirer les metadata avec l'outil "nbstripout"

## Format des fichiers

Format d'un fichier d'entrée
-----------

n = nombre d'inconnues
1 ligne par équation
    -> n+1 éléments par ligne
    -> n+1 lignes

Exemple :

```plaintext
2
2 1 5
4 -6 -20
```

-----------

Format d'un fichier de sortie
-----------

n = nombre d'inconnues
1 ligne par équation
 matrice triangularisée en compressant ->on ne garde que la partie importante
->n + 2 lignes

Exemple :

```plaintext
2
1 0.5 2.5
-8 -30
0.625 3.75
```

-----------
