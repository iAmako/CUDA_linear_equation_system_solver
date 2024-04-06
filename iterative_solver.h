#ifndef ITERATIVE_SOLVER
#define ITERATIVE_SOLVER
#include "linear_system.h"
int find_pivot_line(linear_system* sys, int row);
void swap_lines(linear_system* sys, int line1, int line2);
void solve_system(linear_system* system);
void save_solution(linear_system* sys, int* lines, char* path);
#endif