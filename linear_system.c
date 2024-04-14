#include "linear_system.h"
#include <time.h>
// Taille en entier, tout le reste en float
/**
 * Permet de lire un fichier de système d'équations linéaires en mémoire
 * sys : La variable dans laquelle stocker le système
 * path : le chemin vers le fichier txt du système d'équations linéaires
*/
int read_system(linear_system* system, char* path){
    FILE * f;
    f = fopen(path,"r");
    int file_read = 0;
    

    if( ! (f == NULL)){
        //Recupération du nb de variables
        fscanf(f,"%d \n",&system->len);


        //On alloue le tableau qui stocke les équations
        system->equation = (float **)malloc(sizeof(float *)*system->len);

        //Lecture et remplissage ligne à ligne, /!\ double boucle potentiellement lente.
        for(int i = 0 ; i < system->len ; i++){
            system->equation[i] = (float *)malloc(sizeof(float )*(system->len+1));
            for(int j = 0 ; j < system->len+1 ; j++){
                fscanf(f,"%f",&system->equation[i][j]);
            }
            
        }
        file_read = 1
    }
    fclose(f);
    return file_read;
}

// utilisé uniquement par le générateur normalement 
// c'est bien une fonction différente de celle utilisée par le solveur 
/**
 * Sauvegarde dans un fichier passé en paramètre un système d'équations linéaires
 * sys : La variable dans laquelle stocker le système
 * path : le chemin vers le fichier txt du système d'équations linéaires
*/
void save_system(linear_system* system, char* path){
    FILE * f;

    f = fopen(path,"w");

    fprintf(f,"%d \n",system->len);

    for(int i = 0 ; i < system->len ; i++){
        for(int j = 0 ; j < system->len+1 ; j++){
            fprintf(f,"%f ",system->equation[i][j]);
        }
        fprintf(f,"%s","\n"); 
    }

    fclose(f);

}
void print_system(linear_system* system){
    for(int i = 0 ; i < system->len;i++){
        for(int j = 0 ; j < system->len+1;j++){
            printf("%f ",system->equation[i][j]);
        }
        printf("\n");
    }
}
void free_system(linear_system * system){
    if(system != NULL){
        for(int i = 0 ; i < system->len;i++){
            free(system->equation[i]);
        }
        free(system->equation);
        free(system);
    }
}

/**
 * Retourne un système d'équations linéaires de longeur len
 * len : la longueur du système généré 
*/
linear_system* generate_system(int len){
    linear_system * syst;

    syst = (linear_system *)malloc(sizeof(linear_system));
    syst->len = len;
    syst->equation = (float **)malloc(sizeof(float *)*syst->len);

    for(int i = 0 ; i < len ; i++){
        syst->equation[i] = (float *)malloc(sizeof(float )*(syst->len+1));

    }
    //Generation des valeurs des variables 
    srand(time(NULL));
    float * tab_var = (float *)malloc(sizeof(float)*len);

    for(int i = 0 ; i < len ; i++){
        //Déf arbitraire de l'intervalle [-25 ;25], potentiellement 0 (problème ? j'pense pas mais voilà)
        tab_var[i] = rand()%51-25;
    }

    //Generation des coeffs pour chaque équation
    for(int i = 0 ; i < len ; i++ ){
        //Init à 0 du membre de droite de la ligne i
         syst->equation[i][len] = 0;
         for(int j = 0 ; j < len ; j++){
            //Coeffs de la ligne i 
            syst->equation[i][j] = rand()%51-25;
            //Calcul cumulatif du membre de droite de la ligne i en ajoutant le coeff ij * var i  
            syst->equation[i][len]+= syst->equation[i][j] * tab_var[j];
         }
    }

    //Plus besoin c'est tchao (à la limite le garder si on veut comparer après une résolution)
    free(tab_var);

    return syst;
}
