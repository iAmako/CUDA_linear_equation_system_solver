CUDA_linear_equation_system_solver
=======

## Utilisation
--> Voir le notebook "demo_notebook"

pivot de gauss
-> chaque itération : un pivot (1er élément de la diagonale) -> autant d'étapes que de ligne
n^3
->projet : code c

1) itératif (avec tooutes les bonnes propriétés)
1.1) générer des systèmes
1.2) diagonaliser la matrice
    1.2.1) trouver pivot (plus gros coeff) et inversion si besoin est
    1.2.2) propager pivot
1.5) enregistrer données
documentation / commentaires propres :)
2) parallèle
chacune des lignes peut être traitées en parallèle après le pivot -> openmp (pas pour le 8)
théoriquement, pas loin du #pragma omp parallel for ?  
3) gpu
la suite au prochain épisode

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

TODO -> pour le 8 : générer 10 matrices + 10 résultats
