#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>

#define THREADS_PER_BLOCKS 16

double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

int read_system(double **sys, int *len, double* solution, const char *path){
    FILE * f;
    f = fopen(path,"r");
    int file_read = 0;
    len = (int*)malloc(sizeof(int));
    
    // On alloue le tableau qui stocke la solution
    solution = (double *)malloc((*len) * sizeof(double));

    if( ! (f == NULL)){
        // Recupération du nb de variables
        if(! (fscanf(f,"%d \n",len) ))
            return file_read;
        
        // On alloue le tableau qui stocke les équations
        sys = (double **)malloc(sizeof(double *)*(*len));

        // Lecture et remplissage ligne à ligne, /!\ double boucle potentiellement lente.
        for(int i = 0 ; i < (*len) ; i++){
            sys[i] = (double *)malloc(sizeof(double )*((*len)+1));
            for(int j = 0 ; j < (*len)+1 ; j++){
                if(! (fscanf(f,"%lf",&sys[i][j])))
                    return file_read;
            }
            
        }
        file_read = 1;
        fclose(f);
    }
    return file_read;
}

void save_solution(double * solution, const int len, char* path){
    FILE * save_file;
    save_file = fopen(path, "w");
    // Taille du système
    fprintf(save_file, "%d\n", len);
    // Sauvegarde de la solution
    for(int i = 0; i < len; i++)
        fprintf(save_file, "%lf ", solution[i]);
    fclose(save_file);
    return;
}

// TODO
__global__ void solve_system_kernel(double* d_system, double* d_solution, const int len){
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid < len) {
    }
}


int main(int argc, char const *argv[]){
    double** h_sys = NULL;
    double** d_sys = NULL;
    double* h_solution = NULL;
    double* d_solution = NULL;
    int* h_len = NULL;
    // Variable partagée entre tous les threads
    char h_read_path[128]= "";
    char save_path[128] = "";

    if(argc < 2){
        printf("Utilisation : ./solver_cuda.exe PATH\nAvec :\n\tPATH : Chemin vers le fichier\n");
        return EXIT_SUCCESS;
    }

    // Lecture du fichier & Initialisation
    snprintf(h_read_path,sizeof(h_read_path),"%s",argv[1]);
    
    if(!(read_system(h_sys,h_len,h_solution,h_read_path))){
        printf("Erreur lors de la lecture du fichier \n");
        return EXIT_SUCCESS;
    }

    double tic = wtime();

    // Allocation de mémoire sur le device
    cudaMalloc((void **)&d_sys, sizeof(double) * (*h_len) * ((*h_len) + 1));
    cudaMalloc((void **)&d_solution, sizeof(double) * (*h_len));

    // Copie des données de l'hôte vers le device
    for(int i = 0; i < h_len; i++) {
        cudaMemcpy(d_sys + i * (h_len + 1), h_sys[i], sizeof(double) * (h_len + 1), cudaMemcpyHostToDevice);
    }

    // Résolution du système
     solve_system_kernel<<<(h_len + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_sys, d_solution, const (*h_len));

    // Retour des données sur l'host
    cudaMemcpy(h_solution, d_solution, sizeof(double) * h_len, cudaMemcpyDeviceToHost);
    for(int i = 0; i < h_len; i++) {
        cudaMemcpy(h_sys[i], d_sys + i * (h_len + 1), sizeof(double) * (h_len + 1), cudaMemcpyDeviceToHost);
    }
    double tac = wtime();
    printf("%lf s CUDA \n",tac-tic);
    
    // Récupération des résultats 
    for (int i = (*h_len) - 1; i >= 0; i--) {
        h_solution[i] = h_sys[i][(*h_len)];
        for (int j = i + 1; j < (*h_len); j++) {
            h_solution[i] -= h_sys[i][j] * h_solution[j];
        }
        h_solution[i] = h_solution[i] / h_sys[i][i];
    }
    
    // Sauvegarde des résultats
    snprintf(save_path,sizeof(save_path),"%s_solved.txt",argv[1]); 
    save_solution(h_solution, *h_len, save_path);

    // Libération de la mémoire
    for(int i = 0; i < (*h_len); i++) {
        free(h_sys[i]);
    }
    free(h_sys);
    free(h_len);
    free(h_solution);
    cudaFree(d_sys);
    cudaFree(d_solution);

    return 0;   
}