#ifndef SOLVER_CUDA_H
#define SOLVER_CUDA_H
#include "linear_system.h"
__global__ void find_pivot_kernel(double* matrix, int* lines_link, int len, int pivot_row, int* pivot_line, double* pivot_value);
void swap_lines(int* lines_link, int line1, int line2);
// applique le pivot sur la ligne passé en paramètre en effectuant la multiplication nécessaire
__global__ void apply_pivot_kernel(double* matrix, int* lines_link, int len, int pivot_row);
double* get_solution(linear_system* sys, int* lines_link);
void save_solution(linear_system* sys, int* lines_link, char* path);
void solve_system_cuda(linear_system* system, char* path, int verbose);
double wtime(void);
#endif