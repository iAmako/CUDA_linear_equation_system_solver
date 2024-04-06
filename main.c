#include "linear_system.h"

int main(void){
    linear_system * syst;
    syst = (linear_system *)malloc(sizeof(linear_system));

    //Récupération du système dans sys1.txt
    read_system(syst,"./systems/sys1.txt");

    print_system(syst);

    save_system(syst,"./systems/sys2.txt");
    free_system(syst);
}