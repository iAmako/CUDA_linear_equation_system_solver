#include <stdio.h>
#include <stdlib.h>
#include "iterative_solver.h"

// Renvoies la ligne du pivot
// row -> la colonne du pivot
int find_pivot_line(linear_system* sys, int row);
void swap_lines(linear_system* sys, int line1, int line2);

// boucle des colonnes 
//      on  trouve le plus grand nombre de la i-ème colonne 
//      swap la ligne où il y a le plus grand nombre avec la colonne i 
//      pour chaque ligne suivante, additioner / soustraire la ligne qu'on target par la valeur en colonne i ligne target divisé par la valeur réelle du pivot 
// 
void solve_system(linear_system* system){
    for(int i = 0; i < system->len-1; i++){
        find_pivot_at_line(system, i);

    }
}