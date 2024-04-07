#include "linear_system.h"
#include "iterative_solver.h"
#include <time.h>

int main(void){
    linear_system * syst;
    syst = (linear_system *)malloc(sizeof(linear_system));

    //Récupération du système dans sys1.txt
    read_system(syst,"./systems/sys1.txt");

    //print_system(syst);
 
    solve_system(syst,"./systems/sol1.txt",2);

    free_system(syst);
}