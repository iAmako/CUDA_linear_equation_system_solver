#ifndef ITERATIVE_SOLVER
#define ITERATIVE_SOLVER
#include "linear_system.h"
int find_pivot_for_row(linear_system* sys, int* lines_link, int row);
void swap_lines(linear_system* sys, int line1, int line2);
// applique le pivot sur la ligne passé en paramètre en effectuant la multiplication nécessaire
void apply_pivot_to_line(linear_system* sys, int* lines_link, int target_line, int pivot_row);
void solve_system(linear_system* system);
void save_solution(linear_system* sys, int* lines_link, char* path);
#endif