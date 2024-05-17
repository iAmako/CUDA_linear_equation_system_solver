#ifndef SOLVER_H
#define SOLVER_H
#include "linear_system.h"
int find_pivot_for_row(linear_system* sys, int* lines_link, int row);
void swap_lines(int* lines_link, int line1, int line2);
// applique le pivot sur la ligne passé en paramètre en effectuant la multiplication nécessaire
void apply_pivot_to_line(linear_system* sys, int* lines_link, int target_line, int pivot_row);
double* get_solution(linear_system* sys, int* lines_link);
void save_solution(linear_system* sys, int* lines_link, char* path);
void solve_system(linear_system* system, char* path, int verbose);
void solve_system_parallel(linear_system* system, char* path, int verbose);
double wtime(void);
#endif