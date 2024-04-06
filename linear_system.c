#include "linear_system.h"


void read_system(linear_system* system, char* path){
    FILE * f;
    f = fopen(path,"r");

    if( ! (f == NULL)){
        //Recupération du nb de variables
        fscanf(f,"%d \n",&system->len);

        //On alloue le tableau qui stocke les équations
        system->equation = (int **)malloc(sizeof(int *)*system->len);

        //Lecture et remplissage ligne à ligne 
        for(int i = 0 ; i < system->len ; i++){
            system->equation[i] = (int *)malloc(sizeof(int )*system->len+1);
            fscanf(f,"%d %d %d %d",&system->equation[i][0],&system->equation[i][1],&system->equation[i][2],&system->equation[i][3]);
        }
    }
    fclose(f);
}
void save_system(linear_system* system, char* path);
void print_system(linear_system* system){
    for(int i = 0 ; i < system->len;i++){
        for(int j = 0 ; j < system->len+1;j++){
            printf("%d ",system->equation[i][j]);
        }
        printf("\n");
    }
}
linear_system* generate_system(int len);
void find_pivot(/*truc*/);
void swap_rows(int* row1, int* row2);
