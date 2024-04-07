#include "linear_system.h"
#include "iterative_solver.h"
int main(void){
    linear_system * syst, * syst2;
    syst = (linear_system *)malloc(sizeof(linear_system));

    //Récupération du système dans sys1.txt
    read_system(syst,"./systems/sys1.txt");

    //print_system(syst);
    syst2 = generate_system(3);

    print_system(syst2);
    free_system(syst);
    free_system(syst2);
}