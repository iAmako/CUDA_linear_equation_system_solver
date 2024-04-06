#include "linear_system.h"
#include "iterative_solver.h"

// un exemple de la démo qu'on pourrait avoir 
int main(int argc, char const *argv[])
{
    int len = 3;
    linear_system systems[10];

    char path[] = "./system";

    for (int i = 0; i < 10; i++)
    {
        generate_system(len);
        solve_system(&systems[i]);

        // Nom du système sauvegardé selon l'horodatage et la taille du système
        save_system(&system[i],path);
        len *= 2;
    }

    for (int i = 0; i < 10; i++)
    {
        

    }
    
    
    return 0;
}
