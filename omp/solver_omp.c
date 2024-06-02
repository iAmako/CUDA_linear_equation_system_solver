#include <stdio.h>
#include <stdlib.h>
#include "solver.h"
#include "solver_omp.h"
#include <omp.h>
#include <sys/time.h>
#include <math.h>

// boucle des colonnes 
//      on  trouve le plus grand nombre de la i-ème colonne 
//      swap la ligne où il y a le plus grand nombre avec la colonne i 
//      pour chaque ligne suivante, additioner / soustraire la ligne qu'on target par la valeur en colonne i ligne target divisé par la valeur réelle du pivot 
// 
//      !! -> garder en mémoire les lignes qu'on swap, (tableau d'entier par lequel on passe pour obtenir les valeurs ?)
// system : le système à solve
// path : le path vers le fichier de sortie, qui contiendra la solution
// verbose : sortie uniquement dans le fichier de sortie si verbose = 0, infos basiques dans la sortie standard si verbose = 1, infos avancée si verbose = 2 (à implémenter)
/**
 * Résout un système d'équations linéaires 
 * PATH : le chemin vers le fichie de sauvegarde
 * verbose : optionnel | défaut = 1 | définie la précision de la sortie textuel de la fonction
 *           0 = aucune sortie
*/

void solve_system_parallel(linear_system* system, char* path, int verbose){
    int* pivot_line;
    int* lines_link;

    int num_threads = omp_get_max_threads();

    pivot_line = (int *)malloc(sizeof(int)*num_threads);
    for(int i = 0 ; i < num_threads;i++){
        pivot_line[i] = 0;
    }

    lines_link = (int*)malloc(system->len * sizeof(int));
    for (int i = 0; i < system->len; i++)
    {
        lines_link[i] = i;
    }
    
    double tic = wtime();
    for(int pivot_row = 0; pivot_row < system->len-1  ; pivot_row += num_threads){
        
        //Recherche du pivot sur n ligne 
        #pragma omp parallel
        {
            //Vérification dépassement
            if(pivot_row+omp_get_thread_num() < system->len-1)
                pivot_line[omp_get_thread_num()] = find_pivot_for_row(system, lines_link , pivot_row+omp_get_thread_num());
            
        }
        
        
        for(int i = 0 ; i < num_threads && pivot_row+i < system->len ; i++){
            if(pivot_row+i < system->len-1){
                //Echange des lignes si besoin
                if(pivot_line[i] != pivot_row+i){
                    swap_lines(lines_link, pivot_line[i], pivot_row+i);
                }

                #pragma omp parallel for 
                for(int line_to_change = pivot_row+i+1; line_to_change < system->len; line_to_change++){
                    apply_pivot_to_line(system, lines_link, line_to_change, pivot_row+i);
                }
            }
        }
    }
    double tac = wtime();
    printf("%lf s OPENMP \n",tac-tic);

    save_solution(system, lines_link, path);
    if(verbose > 0){
        printf("Solution sauvegardée : %.64s\n",path);
    }
    free(lines_link);
    free(pivot_line);
}