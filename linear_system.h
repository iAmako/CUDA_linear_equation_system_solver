#ifndef LINEAR_SYSTEM_H
#define LINEAR_SYSTEM_H

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct linear_system_struct
{
    float** equation;
    int len;
} linear_system ;

int read_system(linear_system* system, char* path);
void save_system(linear_system* system, char* path);
void print_system(linear_system* system);
void free_system(linear_system * system);
linear_system* generate_system(int len);



#endif