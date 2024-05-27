#include <stdio.h>
#include <stdlib.h>
#include "solver_cuda.h"
#include <sys/time.h>
#include <math.h>

double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

#define THREADS_PER_BLOCK 128

__global__ void find_pivot_kernel(double* matrix, int* lines_link, int len, int pivot_row, int* pivot_line, double* pivot_value) {
    extern __shared__ double shared_data[];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int local_tid = threadIdx.x;

    if (tid < len) {
        shared_data[local_tid] = fabs(matrix[lines_link[tid] * (len + 1) + pivot_row]);
    } else {
        shared_data[local_tid] = -1.0;
    }

    __syncthreads();

    // Reduction to find the maximum value and its index
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (local_tid < stride && (tid + stride) < len) {
            if (shared_data[local_tid] < shared_data[local_tid + stride]) {
                shared_data[local_tid] = shared_data[local_tid + stride];
                lines_link[local_tid] = lines_link[local_tid + stride];
            }
        }
        __syncthreads();
    }

    if (local_tid == 0) {
        if (shared_data[0] > *pivot_value) {
            *pivot_value = shared_data[0];
            *pivot_line = lines_link[0];
        }
    }
}

__global__ void apply_pivot_kernel(double* matrix, int* lines_link, int len, int pivot_row) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= len) return;

    int target_line = tid + pivot_row + 1;
    if (target_line >= len) return;

    double multiplier = matrix[lines_link[target_line] * (len + 1) + pivot_row] / matrix[lines_link[pivot_row] * (len + 1) + pivot_row];
    for (int i = 0; i < len + 1; i++) {
        matrix[lines_link[target_line] * (len + 1) + i] -= multiplier * matrix[lines_link[pivot_row] * (len + 1) + i];
    }
}


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
 * system : le système à résoudre
 * PATH : le chemin vers le fichie de sauvegarde
 * verbose : optionnel | défaut = 1 | définie la précision de la sortie textuel de la fonction
 *           0 = aucune sortie
*/
void solve_system_cuda(linear_system* system, char* path, int verbose) {
    int pivot_line = 0;
    int* lines_link;
    int len = system->len;

    lines_link = (int*)malloc(len * sizeof(int));
    for (int i = 0; i < len; i++) {
        lines_link[i] = i;
    }

    // Allocating memory on GPU
    double* d_matrix;
    int* d_lines_link;
    int* d_pivot_line;
    double* d_pivot_value;
    cudaMalloc(&d_matrix, len * (len + 1) * sizeof(double));
    cudaMalloc(&d_lines_link, len * sizeof(int));
    cudaMalloc(&d_pivot_line, sizeof(int));
    cudaMalloc(&d_pivot_value, sizeof(double));

    // Copy data to GPU
    for (int i = 0; i < len; i++) {
        cudaMemcpy(d_matrix + i * (len + 1), system->equation[i], (len + 1) * sizeof(double), cudaMemcpyHostToDevice);
    }
    cudaMemcpy(d_lines_link, lines_link, len * sizeof(int), cudaMemcpyHostToDevice);

    double tic = wtime();



    for (int pivot_row = 0; pivot_row < len - 1; pivot_row++) {
        cudaMemset(d_pivot_value, 0, sizeof(double));
        find_pivot_kernel<<<(len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(double)>>>(d_matrix, d_lines_link, len, pivot_row, d_pivot_line, d_pivot_value);
        cudaMemcpy(&pivot_line, d_pivot_line, sizeof(int), cudaMemcpyDeviceToHost);

        if (verbose > 1) printf("ligne pivot : %d\n", pivot_line);

        if (pivot_line != pivot_row) {
            if (verbose > 1) printf("échange des lignes %d et %d\n", lines_link[pivot_line], lines_link[pivot_row]);
            int tmp = lines_link[pivot_line];
            lines_link[pivot_line] = lines_link[pivot_row];
            lines_link[pivot_row] = tmp;
            cudaMemcpy(d_lines_link, lines_link, len * sizeof(int), cudaMemcpyHostToDevice);
        }

        apply_pivot_kernel<<<(len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_matrix, d_lines_link, len, pivot_row);
    }

    double tac = wtime();
    printf("%lf s Iteratif \n", tac - tic);

    // Copy result back to CPU
    for (int i = 0; i < len; i++) {
        cudaMemcpy(system->equation[i], d_matrix + i * (len + 1), (len + 1) * sizeof(double), cudaMemcpyDeviceToHost);
    }

    save_solution(system, lines_link, path);
    if (verbose > 0) {
        printf("Solution sauvegardée : %.64s\n", path);
    }

    free(lines_link);
    cudaFree(d_matrix);
    cudaFree(d_lines_link);
    cudaFree(d_pivot_line);
    cudaFree(d_pivot_value);
}
 
/**
 * Echange les adresses de deux lignes de la matrice 
 * lines_link : la matrice des adresses du système d'équations linéaires 
 * line1, line2 : les lignes à échanger
*/
void swap_lines(int* lines_link, int line1, int line2){

    int tmp = lines_link[line1];
    lines_link[line1] = lines_link[line2];
    lines_link[line2] = tmp;
}

// la fonction est bien différente de celle utilisée par le générateur de système 
// puisqu'elle doit prendre en compte que des lignes ont été swaps 
// ligne 1 : n = la taille 
// lignes 2 -> n+1 les n prochaines ligne (juste dans l'ordre, pas besoin de lines link vu qu'on a pas changé l'ordre dans la mémoire)
// ligne n+2 -> la solution  (besoin de lines_link)
/**
 * Sauvegarde la solution du système d'équations linéaires 
 * sys : le système d'équations linéaires étudié
 * lines_link : La matrice des adresses du système d'équations linéaires
 * PATH : le chemin vers le fichier de sauvegarde
*/
void save_solution(linear_system* sys, int* lines_link, char* path){

    FILE * f;
    // Ouverture du fichier
    f = fopen(path,"w");

    // Sauvegarde de la taille
    fprintf(f,"%d\n",sys->len);

    /* //à remettre 
    // Sauvegarde des nouvelles équations
    for (int i = 0; i < sys->len; i++)
    {
        for (int j = i; j <= sys->len; j++)
        {
            fprintf(f,"%lf ",sys->equation[lines_link[i]][j]);
        }
        fprintf(f,"\n");
    }
    */

    // Sauvegarde de la solution
    double* solution;
    solution = get_solution(sys, lines_link);
    for (int i = 0; i < sys->len; i++)
    {
        fprintf(f,"%lf ",solution[i]);
    }

    // Libération de la mémoire 
    free(solution);

    // Fermeture du fichier
    fclose(f);

}

/**
 * Retourne la solution au système d'équations linéaires triangularisé 
 * sys : le système d'équations linéaires étudié
 * lines_link : La matrice des adresses du système d'équations linéaires
*/
double* get_solution(linear_system* sys, int* lines_link){

    double* solution = (double *)malloc(sizeof(double)*sys->len);
    double tmp_membre_droit;
    for (int i = sys->len-1; i >= 0; i--)
    {
        // On récupère le membre de droite dans l'équation dans tous les cas 
        tmp_membre_droit = sys->equation[lines_link[i]][sys->len];
        for (int j = sys->len-1; j > i; j--)
        {
            // "On passe à droite" les éléments déjà connus et leur coeff
           tmp_membre_droit -= solution[j] * sys->equation[lines_link[i]][j];
        }
        // Calcul de la valeur de la variable i grâce aux éléments précédemment calculés 
        solution[i] = tmp_membre_droit / sys->equation[lines_link[i]][i]; 
    }

    return solution;
}