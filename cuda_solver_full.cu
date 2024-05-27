#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>



typedef struct linear_system_struct
{
    double** equation;
    int len;
} linear_system ;

// Taille en entier, tout le reste en double
/**
 * Permet de lire un fichier de système d'équations linéaires en mémoire
 * sys : La variable dans laquelle stocker le système
 * path : le chemin vers le fichier txt du système d'équations linéaires
*/
int read_system(linear_system* system, char* path){
    FILE * f;
    f = fopen(path,"r");
    int file_read = 0;
    

    if( ! (f == NULL)){
        //Recupération du nb de variables
        if(! (fscanf(f,"%d \n",&system->len) ))
            return file_read;
        
        //On alloue le tableau qui stocke les équations
        system->equation = (double **)malloc(sizeof(double *)*system->len);

        //Lecture et remplissage ligne à ligne, /!\ double boucle potentiellement lente.
        for(int i = 0 ; i < system->len ; i++){
            system->equation[i] = (double *)malloc(sizeof(double )*(system->len+1));
            for(int j = 0 ; j < system->len+1 ; j++){
                if(! (fscanf(f,"%lf",&system->equation[i][j])))
                    return file_read;
            }
            
        }
        file_read = 1;
        fclose(f);
    }
    return file_read;
}

// utilisé uniquement par le générateur normalement 
// c'est bien une fonction différente de celle utilisée par le solveur 
/**
 * Sauvegarde dans un fichier passé en paramètre un système d'équations linéaires
 * sys : La variable dans laquelle stocker le système
 * path : le chemin vers le fichier txt du système d'équations linéaires
*/
void save_system(linear_system* system, char* path){
    FILE * f;

    f = fopen(path,"w");

    fprintf(f,"%d \n",system->len);

    for(int i = 0 ; i < system->len ; i++){
        for(int j = 0 ; j < system->len+1 ; j++){
            fprintf(f,"%lf ",system->equation[i][j]);
        }
        fprintf(f,"%s","\n"); 
    }

    fclose(f);

}
void print_system(linear_system* system){
    for(int i = 0 ; i < system->len;i++){
        for(int j = 0 ; j < system->len+1;j++){
            printf("%lf ",system->equation[i][j]);
        }
        printf("\n");
    }
}
void free_system(linear_system * system){
    if(system != NULL){
        for(int i = 0 ; i < system->len;i++){
            free(system->equation[i]);
        }
        free(system->equation);
        free(system);
    }
}

/**
 * Retourne un système d'équations linéaires de longeur len
 * len : la longueur du système généré 
*/
linear_system* generate_system(int len){
    linear_system * syst;

    syst = (linear_system *)malloc(sizeof(linear_system));
    syst->len = len;
    syst->equation = (double **)malloc(sizeof(double *)*syst->len);

    for(int i = 0 ; i < len ; i++){
        syst->equation[i] = (double *)malloc(sizeof(double )*(syst->len+1));

    }
    //Generation des valeurs des variables 
    srand(time(NULL));
    double * tab_var = (double *)malloc(sizeof(double)*len);

    for(int i = 0 ; i < len ; i++){
        //Déf arbitraire de l'intervalle [-25 ;25], potentiellement 0 (problème ? j'pense pas mais voilà)
        tab_var[i] = rand()%51-25;
    }

    //Generation des coeffs pour chaque équation
    for(int i = 0 ; i < len ; i++ ){
        //Init à 0 du membre de droite de la ligne i
         syst->equation[i][len] = 0;
         for(int j = 0 ; j < len ; j++){
            //Coeffs de la ligne i 
            syst->equation[i][j] = rand()%51-25;
            //Calcul cumulatif du membre de droite de la ligne i en ajoutant le coeff ij * var i  
            syst->equation[i][len]+= syst->equation[i][j] * tab_var[j];
         }
    }

    //Plus besoin c'est tchao (à la limite le garder si on veut comparer après une résolution)
    free(tab_var);

    return syst;
}

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
        cudaMemcpy(&lines_link, d_lines_link, sizeof(int), cudaMemcpyDeviceToHost);


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

    cudaMemcpy(d_lines_link, lines_link, len * sizeof(int), cudaMemcpyDeviceToHost);

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


//main solveur de système 
int main(int argc, char const *argv[])
{

    if(argc < 2){
        printf("Utilisation : ./solver.exe PATH VERBOSE\nAvec :\n\tPATH : Chemin vers le fichier\n\tVERBOSE : Le niveau d'affichage du programme (optionnel, défaut = 1)\n");
        return EXIT_SUCCESS;
    }

    int verbose = 1;

    // Verbose arg ? 
    if (argc > 2)
    {
        verbose = atoi(argv[2]);
    }

    char read_path[64];
    snprintf(read_path,sizeof(read_path),"%s",argv[1]);
    
    //Lecture du fichier    
    linear_system* sys;
    
    sys = (linear_system *)malloc(sizeof(linear_system));

    if( ! (read_system(sys,read_path))){
        printf("Erreur lors de la lecture du fichier \n");
        return EXIT_SUCCESS;
    }
    
    char save_path[64];
    snprintf(save_path,sizeof(save_path),"%s_solved.txt",argv[1]); 

    //Calculs & sauvegarde
    solve_system_cuda(sys,save_path,verbose);

    //Libération de la mémoire
    free_system(sys);
    return 0;
}

