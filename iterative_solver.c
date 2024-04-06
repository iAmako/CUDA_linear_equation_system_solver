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
void solve_system(linear_system* system){
    int pivot_line = 0;
    int* lines_link;
    lines_link = (int*)malloc(system->len * sizeof(int));
    
    for (int i = 0; i < system->len; i++)
    {
        
    }
    

    for(int pivot_row = 0; pivot_row < system->len-1; pivot_row++){

        //Recherche du pivot
        pivot_line = find_pivot_for_row(system, lines_link , pivot_row);

        //Echange des lignes si besoin
        if(pivot_line != pivot_row){
            swap_lines(system,  lines_link, pivot_line, pivot_row);
        }
        


    }
}

// Renvoies la ligne du pivot
// row -> la colonne du pivot
// TODO
int find_pivot_for_row(linear_system* sys, int* lines_link, int row);
// On effectue ce changement uniquement dans le tableau d'entier pour garder sa forme originale dans le système linéaire 
// On va donc en permanence accéder aux données via ce tableau 
// Ça va permettre de simplifier la sauvegarde des données de sortie 
// à vérifier : est-ce que ça marche encore si une ligne est swap deux fois d'affilée ? 
// sûrement mieux de changer tout ce système de manière à ce qu'on ne swap pas les adresses mémoires 
void swap_lines(linear_system* sys ,int* lines_link, int line1, int line2){
    
    int* tmp = sys->equation[line1];
    sys->equation[line1] = sys->equation[line2];
    sys->equation[line2] = tmp;

    int tmp = lines_link[line1];
    lines_link[line1] = lines_link[line2];
    lines_link[line2] = tmp;
}

// la fonction est bien différente de celle utilisée par le générateur de système 
// puisqu'elle doit prendre en compte que des lignes ont été swaps 
// TODO
void save_solution(linear_system* sys, int* lines, char* path){

}