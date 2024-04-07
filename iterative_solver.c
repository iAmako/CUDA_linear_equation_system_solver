#include <stdio.h>
#include <stdlib.h>
#include "iterative_solver.h"



// boucle des colonnes 
//      on  trouve le plus grand nombre de la i-ème colonne 
//      swap la ligne où il y a le plus grand nombre avec la colonne i 
//      pour chaque ligne suivante, additioner / soustraire la ligne qu'on target par la valeur en colonne i ligne target divisé par la valeur réelle du pivot 
// 
//      !! -> garder en mémoire les lignes qu'on swap, (tableau d'entier par lequel on passe pour obtenir les valeurs ?)
// WIP
// system : le système à solve
// path : le path vers le fichier de sortie, qui contiendra la solution
// verbose : sortie uniquement dans le fichier de sortie si verbose = 0, infos basiques dans la sortie standard si verbose = 1, infos avancée si verbose = 2 (à implémenter)
void solve_system(linear_system* system, char* path, int verbose){
    int pivot_line = 0;
    int* lines_link;
    lines_link = (int*)malloc(system->len * sizeof(int));
    
    for (int i = 0; i < system->len; i++)
    {
        lines_link[i] = i;
    }
    

    for(int pivot_row = 0; pivot_row < system->len-1; pivot_row++){

        //Recherche du pivot
        pivot_line = find_pivot_for_row(system, lines_link , pivot_row);

        //Echange des lignes si besoin
        if(pivot_line != pivot_row){
            swap_lines(system, lines_link, pivot_line, pivot_row);
        }
        
        for(int line_to_change = pivot_row+1; line_to_change < system->len; line_to_change++){
            apply_pivot_to_line(system, lines_link, line_to_change, pivot_row);
        }

    }


    save_solution(system, lines_link, path);
    if(verbose > 0){
        printf("Solution sauvegardée : %.64s\n");
    }
    free(lines_link);
}

// Renvoies la ligne du pivot
// row -> la colonne du pivot
// La ligne pivot est celle détenant la plus grande valeur sur la colonne recherchée
int find_pivot_for_row(linear_system* sys, int* lines_link, int row){
    int pivot_line = row;
    int pivot_value = sys->equation[lines_link[row]][row];
    int cur_value = 0;
    for (int cur_line = row+1; cur_line < sys->len; cur_line++)
    {
        cur_value = sys->equation[lines_link[cur_line]][row];
        if(cur_value > pivot_value){
            pivot_line = cur_line;
            pivot_value = cur_value;
        }
    }
    return pivot_line;
}
// On effectue ce changement uniquement dans le tableau d'entier pour garder sa forme originale dans le système linéaire 
// On va donc en permanence accéder aux données via ce tableau 
// Ça va permettre de simplifier la sauvegarde des données de sortie 
// à vérifier : est-ce que ça marche encore si une ligne est swap deux fois d'affilée ? 
// sûrement mieux de changer tout ce système de manière à ce qu'on ne swap pas les adresses mémoires 
void swap_lines(linear_system* sys ,int* lines_link, int line1, int line2){

    int tmp = lines_link[line1];
    lines_link[line1] = lines_link[line2];
    lines_link[line2] = tmp;
}
// applique le pivot sur la ligne passé en paramètre en effectuant la multiplication nécessaire
// comme on a swap les lignes pivot row est normalement = pivot line 
// peut-être besoin de passer par lines link dépendant comment on gère ? 
void apply_pivot_to_line(linear_system* sys, int* lines_link, int target_line, int pivot_row){
    
    // Calcul du coefficient
    float multiplier = 1.0f; 
    multiplier = sys->equation[lines_link[target_line]][pivot_row] / sys->equation[lines_link[pivot_row]][pivot_row];
    
    // Application du pivot sur la ligne
    for (int i = 0; i < sys->len; i++)
    {
        sys->equation[lines_link[target_line]][i] = sys->equation[lines_link[target_line]][i] - multiplier * sys->equation[lines_link[pivot_row]][i];
    }
    
}
// la fonction est bien différente de celle utilisée par le générateur de système 
// puisqu'elle doit prendre en compte que des lignes ont été swaps 
// ligne 1 : n = la taille 
// lignes 2 -> n+1 les n prochaines ligne (juste dans l'ordre, pas besoin de lines link vu qu'on a pas changé l'ordre dans la mémoire)
// ligne n+2 -> la solution  (besoin de lines_link)
// TODO
void save_solution(linear_system* sys, int* lines_link, char* path){

}