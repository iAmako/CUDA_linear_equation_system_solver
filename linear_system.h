#ifndef LINEAR_SYSTEM_H
#define LINEAR_SYSTEM_H

#include <string.h>

typedef struct linear_system_struct
{
    int** equation;
    int len;
} linear_system ;

void read_system(linear_system system, char* path);
void save_system(linear_system system, char* path);
int** generate_system(int len);
void find_pivot(/*truc*/);
void swap_rows(int* row1, int* row2);

#endif