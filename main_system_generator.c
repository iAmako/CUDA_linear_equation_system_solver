#include "linear_system.h"
#include "iterative_solver.h"
#include <time.h>

//main generateur de systemes
//enregsitré sous ./systems/sys_horodatage.txt
int main(int argc, char const *argv[])
{
    
    if(argc > 3 || argc < 0){
        printf("Utilisation : system_generator LEN NB\nAvec :\n\tLEN : La taille des systèmes (Défaut : 30) \n\tNB : Le nombre de systèmes à générer (Défaut : 1)");
        return EXIT_FAILURE;
    }

    char filename[64];
    time_t cur_time;

    struct tm *time_str;

    
    int count = 1;
    if(argc == 2 ){
        count = atoi(argv[3]);
    }

    int len = 30;
    if(argc > 0 ){
        len = atoi(argv[2]);
    }

    linear_system* sys;

    for (int i = 0; i < count; i++)
    {
        cur_time = time(NULL);
        time_str = gmtime(&cur_time);
        sprintf(filename, sizeof(filename), "./systems/sys_%d_%Y-%m-%d_%H:%M:%S.txt", len, time_str);
        sys = generate_system(len);
        save_system(system,filename);
        free_system(sys);
    }
    

    return 0;
}