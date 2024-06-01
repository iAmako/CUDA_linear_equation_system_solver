#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>

double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

//TODO
void save_solution(double ** sys, const int len, char* path){
    return
}

//TODO
__global__ void solve_system(double** system, const int len){

}

int main(void){
    double** h_sys;
    double** d_sys;
    int* h_len;
    char h_read_path[128];
    char save_path[128];

    if(argc < 1){
        printf("Utilisation : ./solver_cuda.exe PATH\nAvec :\n\tPATH : Chemin vers le fichier\n");
        return EXIT_SUCCESS;
    }

    //Lecture du fichier & Initialisation
    snprintf(read_path,sizeof(read_path),"%s",argv[1]);
    
    if(!(read_system(sys,len,read_path))){
        printf("Erreur lors de la lecture du fichier \n");
        return EXIT_SUCCESS;
    }

    //Passage des données sur le device
    cudaMalloc((void **));...
    cudaMemcpy(, cudaMemcpyHostToDevice);...

    //Résolution du système
    solve_system<<<1,1>>>(d_n, const );...

    //Retour des données sur l'host
    cudaMemcpy(, cudaMemcpyDeviceToHost);...
    
    //Sauvegarde des résultats
    snprintf(save_path,sizeof(save_path),"%s_solved.txt",argv[1]); 
    save_solution(sys, len, save_path);

    //Libération de la mémoire
    free(h_sys);
    free(h_len);
    cudaFree(d_sys);

    return 0;   
}