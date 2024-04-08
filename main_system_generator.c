#include "linear_system.h"
#include "iterative_solver.h"
#include <time.h>

// main generateur de systemes
// enregsitré sous ./systems/sys_horodatage.txt
int main(int argc, char const *argv[])
{

    if (argc > 2)
    {
        printf("Utilisation : system_generator LEN NB\nAvec :\n\tLEN : La taille des systèmes (Défaut : 30) \n\tNB : Le nombre de systèmes à générer (Défaut : 1)\n");
    }
    char filename[64];
    time_t cur_time;

    struct tm *time_str;

    int count = 1;
    if (argc == 2)
    {
        count = atoi(argv[2]);
    }

    int len = 30;
    if (argc > 0)
    {
        len = atoi(argv[1]);
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
    }

    return 0;
}