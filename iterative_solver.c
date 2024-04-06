#include <stdio.h>
#include <stdlib.h>
#include "iterative_solver.h"



// boucle des colonnes 
//      on  trouve le plus grand nombre de la i-ème colonne 
//      swap la ligne où il y a le plus grand nombre avec la colonne i 
//      pour chaque ligne suivante, additioner / soustraire la ligne qu'on target par la valeur en colonne i ligne target divisé par la valeur réelle du pivot 
// 
//      !! -> garder en mémoire les lignes qu'on swap, (tableau d'entier par lequel on passe pour obtenir les valeurs ?)
void solve_system(linear_system* system){
    int pivot_line = 0;
    for(int pivot_row = 0; pivot_row < system->len-1; pivot_row++){

        //Recherche du pivot
        pivot_line = find_pivot_for_row(system, pivot_row);

        //Echange des lignes si besoin
        if(pivot_line != pivot_row){
            swap_lines(system, pivot_line, pivot_row);
        }
        

    }
}

// Renvoies la ligne du pivot
// row -> la colonne du pivot
int find_pivot_line(linear_system* sys, int row);
void swap_lines(linear_system* sys, int line1, int line2);

// la fonction est bien différente de celle utilisée par le générateur de système 
// puisqu'elle doit prendre en compte que des lignes ont été swaps 
void save_solution(linear_system* sys, int* lines, char* path){

}