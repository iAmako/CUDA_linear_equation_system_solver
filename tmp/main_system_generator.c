#include "linear_system.h"
#include "solver.h"
#include <time.h>

// main generateur de systemes
// enregsitré sous ./systems/sys_horodatage.txt
// TODO permettre de créer n fichiers avec des tailles allant de x à y (y-x / n = l'augmentation de taille entre chaque itération)

int main(int argc, char const *argv[])
{
    if (argc > 3)
    {
        printf("Utilisation : ./generator.exe LEN NB\nAvec :\n\tLEN : La taille des systèmes (Défaut : 30) \n\tNB : Le nombre de systèmes à générer (Défaut : 1)\n");
        return EXIT_SUCCESS;
    }
    char filename[64];
    time_t cur_time;

    struct tm *time_str;
    int len = 30;
    if(argc > 2 ){
        len = atoi(argv[1]);
    }
    int count = 1;
    if (argc == 3)
    {
        count = atoi(argv[2]);
    }


    linear_system *sys;

    for (int i = 0; i < count; i++)
    {
        cur_time = time(NULL);
        time_str = gmtime(&cur_time);
        snprintf(filename, sizeof(filename), "./systems/sys_%d_%d-%d-%d_%d-%d-%d.txt", len, time_str->tm_year,
                 time_str->tm_mon,
                 time_str->tm_mday,
                 time_str->tm_hour,
                 time_str->tm_min,
                 time_str->tm_sec);
        sys = generate_system(len);
        save_system(sys, filename);
        free_system(sys);
        fprintf(stdout, "%s créé \n",filename);
    }

    return 0;
}