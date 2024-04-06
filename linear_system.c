#include "linear_system.h"

void read_system(linear_system* system, char* path);
void save_system(linear_system* system, char* path);
linear_system* generate_system(int len);
void find_pivot(/*truc*/);
void swap_rows(int* row1, int* row2);
