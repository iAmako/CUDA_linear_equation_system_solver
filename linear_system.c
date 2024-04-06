#include "linear_system.h"


void read_system(linear_system* system, char* path){
    FILE * f;
    f = fopen(path,"r");

    if( ! (f == NULL)){
        //Recupération du nb de variables
        fscanf(f,"%d \n",&system->len);

        //On alloue le tableau qui stocke les équations
        system->equation = (int **)malloc(sizeof(int *)*system->len);

        //Lecture et remplissage ligne à ligne, /!\ double boucle potentiellement lente.
        for(int i = 0 ; i < system->len ; i++){
            system->equation[i] = (int *)malloc(sizeof(int )*(system->len+1));
            for(int j = 0 ; j < system->len+1 ; j++){
                fscanf(f,"%d",&system->equation[i][j]);
            }
            
        }
    }
    fclose(f);
}
void save_system(linear_system* system, char* path){
    FILE * f;

    f = fopen(path,"w");

    fprintf(f,"%d \n",system->len);

    for(int i = 0 ; i < system->len ; i++){
        for(int j = 0 ; j < system->len+1 ; j++){
            fprintf(f,"%d ",system->equation[i][j]);
        }
        fprintf(f,"%s","\n"); 
    }

    fclose(f);

}
void print_system(linear_system* system){
    for(int i = 0 ; i < system->len;i++){
        for(int j = 0 ; j < system->len+1;j++){
            printf("%d ",system->equation[i][j]);
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
linear_system* generate_system(int len);
void find_pivot(/*truc*/);
void swap_rows(int* row1, int* row2);
