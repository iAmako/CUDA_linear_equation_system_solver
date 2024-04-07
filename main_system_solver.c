#include "linear_system.h"
#include "iterative_solver.h"
#include <time.h>

//main solveur de système 
int main(int argc, char const *argv[])
{

    if(argc < 1 || argc > 2){
        printf("Utilisation : system_solver PATH VERBOSE\nAvec :\n\tPATH : Chemin vers le fichier\n\tVERBOSE : Le niveau d'affichage du programme (optionnel, défaut = 1)");
    }

    int verbose = 1;

    // Verbose arg ? 
    if (argc > 1)
    {
        verbose = atoi(argv[2]);
    }
    
    //Lecture du fichier    
    linear_system* sys;
    read_system(sys,argv[1]);
    
    char save_path[64];
    fprintf(save_path,sizeof(save_path),"%s_solved",argv[1]); 

    //Calculs & sauvegarde    
    solve_system(sys,save_path,save_path,verbose);

    //Libération de la mémoire
    free_system(sys);

    return 0;
}
