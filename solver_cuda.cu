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


__global__ void solve(double** system, int* len){

}

int main(void){
    if(argc < 2){
        printf("Utilisation : ./solver_cuda.exe PATH\nAvec :\n\tPATH : Chemin vers le fichier\n");
        return EXIT_SUCCESS;
    }

    for(int i= 0 ; i < 5 ; i++){
        n[i]= i ;
    }

    cudaMalloc((void **));

    cudaMemcpy(d_n, n, sizeof(), cudaMemcpyHostToDevice);

    solve<<<1,1>>>(d_n);

    cudaMemcpy( n, d_n, sizeof(), cudaMemcpyDeviceToHost);
    
    free(n);
    cudaFree(d_n);
    return 0;   
}