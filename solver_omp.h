#ifndef SOLVER_H
#define SOLVER_H
#include "linear_system.h"
void solve_system_parallel(linear_system* system, char* path, int verbose);
#endif