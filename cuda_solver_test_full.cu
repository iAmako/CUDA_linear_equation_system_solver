
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <time.h>

typedef struct linear_system_struct
{
    double** equation;
    int len;
} linear_system ;

double wtime(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
}

// Taille en entier, tout le reste en double
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
        if(! (fscanf(f,"%d \n",&system->len) ))
            return file_read;
        
        //On alloue le tableau qui stocke les équations
        system->equation = (double **)malloc(sizeof(double *)*system->len);

        //Lecture et remplissage ligne à ligne, /!\ double boucle potentiellement lente.
        for(int i = 0 ; i < system->len ; i++){
            system->equation[i] = (double *)malloc(sizeof(double )*(system->len+1));
            for(int j = 0 ; j < system->len+1 ; j++){
                if(! (fscanf(f,"%lf",&system->equation[i][j])))
                    return file_read;
            }
            
        }
        file_read = 1;
        fclose(f);
    }
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
            fprintf(f,"%lf ",system->equation[i][j]);
        }
        fprintf(f,"%s","\n"); 
    }

    fclose(f);

}
void print_system(linear_system* system){
    for(int i = 0 ; i < system->len;i++){
        for(int j = 0 ; j < system->len+1;j++){
            printf("%lf ",system->equation[i][j]);
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
    syst->equation = (double **)malloc(sizeof(double *)*syst->len);

    for(int i = 0 ; i < len ; i++){
        syst->equation[i] = (double *)malloc(sizeof(double )*(syst->len+1));

    }
    //Generation des valeurs des variables 
    srand(time(NULL));
    double * tab_var = (double *)malloc(sizeof(double)*len);

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

// Renvoie la ligne du pivot
// row -> la colonne du pivot
// La ligne pivot est celle détenant la plus grande valeur sur la colonne recherchée
/**
 * Renvoies le numéro de ligne du pivot recherché pour la colonne ROW passée en paramètre
 * sys : le système d'équations linéaires étudié
 * lines_link : La matrice des adresses du système d'équations linéaires
 * row : la colonne pour laquelle on recherche le pivot 
*/
int find_pivot_for_row(linear_system* sys, int* lines_link, int row){
    int pivot_line = row;
    double pivot_value = fabs(sys->equation[lines_link[row]][row]);
    double cur_value = 0;
    for (int cur_line = row+1; cur_line < sys->len; cur_line++)
    {
        cur_value = sys->equation[lines_link[cur_line]][row];
        if(fabs(cur_value) > pivot_value){
            pivot_line = cur_line;
            pivot_value = fabs(cur_value);
        }
    }
    return pivot_line;
}
// On effectue ce changement uniquement dans le tableau d'entier pour garder sa forme originale dans le système d'équations linéaires 
// On va donc en permanence accéder aux données via ce tableau 
// Ça va permettre de simplifier la sauvegarde des données de sortie 
// à vérifier : est-ce que ça marche encore si une ligne est swap deux fois d'affilée ? 
// sûrement mieux de changer tout ce système de manière à ce qu'on ne swap pas les adresses mémoires --> ça a été changé mais pas sûr que ça marche 
/**
 * Echange les adresses de deux lignes de la matrice 
 * lines_link : la matrice des adresses du système d'équations linéaires 
 * line1, line2 : les lignes à échanger
*/
void swap_lines(int* lines_link, int line1, int line2){

    int tmp = lines_link[line1];
    lines_link[line1] = lines_link[line2];
    lines_link[line2] = tmp;
}
// applique le pivot sur la ligne passé en paramètre en effectuant la multiplication nécessaire
// comme on a swap les lignes pivot row est normalement = pivot line 
// peut-être besoin de passer par lines link dépendant comment on gère ? 
/**
 * Effectue les opérations d'addition & multiplication pour appliquer le pivot à la ligne passé en paramètre 
 * sys : le système d'équations linéaires étudié
 * lines_link : La matrice des adresses du système d'équations linéaires
 * target_line : la ligne sur laquelle appliquer le pivot 
 * pivot_row : la colonne du pivot 
*/
__device__ void apply_pivot_to_line(linear_system* sys, int* lines_link, int target_line, int pivot_row){
    
    // Calcul du coefficient
    double multiplier = 1.0f; 
    multiplier = sys->equation[lines_link[target_line]][pivot_row] / sys->equation[lines_link[pivot_row]][pivot_row];
    
    // Application du pivot sur la ligne
    // C'est bien sys->len+1 pour modifier l'entièreté de l'équation en prenant en compte le résultat 
    for (int i = 0; i < sys->len+1; i++)
    {
        sys->equation[lines_link[target_line]][i] = sys->equation[lines_link[target_line]][i] - multiplier * sys->equation[lines_link[pivot_row]][i];
    }
    
}


// Retourne la solution d'un système déjà triangularisé (c'est à dire qui est passé par la fonction solve_system)
// besoin d'utiliser les lines link ici
/**
 * Retourne la solution au système d'équations linéaires triangularisé 
 * sys : le système d'équations linéaires étudié
 * lines_link : La matrice des adresses du système d'équations linéaires
*/
double* get_solution(linear_system* sys, int* lines_link){

    double* solution = (double *)malloc(sizeof(double)*sys->len);
    double tmp_membre_droit;
    for (int i = sys->len-1; i >= 0; i--)
    {
        // On récupère le membre de droite dans l'équation dans tous les cas 
        tmp_membre_droit = sys->equation[lines_link[i]][sys->len];
        for (int j = sys->len-1; j > i; j--)
        {
            // "On passe à droite" les éléments déjà connus et leur coeff
           tmp_membre_droit -= solution[j] * sys->equation[lines_link[i]][j];
        }
        // Calcul de la valeur de la variable i grâce aux éléments précédemment calculés 
        solution[i] = tmp_membre_droit / sys->equation[lines_link[i]][i]; 
    }

    return solution;
}


// la fonction est bien différente de celle utilisée par le générateur de système 
// puisqu'elle doit prendre en compte que des lignes ont été swaps 
// ligne 1 : n = la taille 
// lignes 2 -> n+1 les n prochaines ligne (juste dans l'ordre, pas besoin de lines link vu qu'on a pas changé l'ordre dans la mémoire)
// ligne n+2 -> la solution  (besoin de lines_link)
/**
 * Sauvegarde la solution du système d'équations linéaires 
 * sys : le système d'équations linéaires étudié
 * lines_link : La matrice des adresses du système d'équations linéaires
 * PATH : le chemin vers le fichier de sauvegarde
*/
void save_solution(linear_system* sys, int* lines_link, char* path){

    FILE * f;
    // Ouverture du fichier
    f = fopen(path,"w");

    // Sauvegarde de la taille
    fprintf(f,"%d\n",sys->len);

    /* //à remettre 
    // Sauvegarde des nouvelles équations
    for (int i = 0; i < sys->len; i++)
    {
        for (int j = i; j <= sys->len; j++)
        {
            fprintf(f,"%lf ",sys->equation[lines_link[i]][j]);
        }
        fprintf(f,"\n");
    }
    */

    // Sauvegarde de la solution
    double* solution;
    solution = get_solution(sys, lines_link);
    for (int i = 0; i < sys->len; i++)
    {
        fprintf(f,"%lf ",solution[i]);
    }

    // Libération de la mémoire 
    free(solution);

    // Fermeture du fichier
    fclose(f);

}


__global__ void ApplyPivotToLinesKernel(linear_system* d_system, int* lines_link, int pivot_row){
    int i = blockIdx.x * blockDim.x + threadIdx.x + 1 + pivot_row;
    if(i < d_system->len){
        apply_pivot_to_line(d_system,lines_link,i,pivot_row);
    }
}
//a passer sur GPU
/*
for(int i = pivot_row+1; i < system->len; i++){
        apply_pivot_to_line(system, lines_link, i, pivot_row);
}
*/


void solve_system(linear_system* system, char* path, int verbose){
    int pivot_line = 0;
    int* lines_link;

    int threadsPerBlock = 128;
    int blocksPerGrid = 1;
    linear_system* d_system;


    lines_link = (int*)malloc(system->len * sizeof(int));

    for (int i = 0; i < system->len; i++)
    {
        lines_link[i] = i;
    }
    
    double tic = wtime();
    for(int pivot_row = 0; pivot_row < system->len-1  ; pivot_row++){

        //Recherche du pivot sur n ligne 
        pivot_line = find_pivot_for_row(system, lines_link , pivot_row);
        
        if(verbose > 1) printf("ligne pivot : %d\n",pivot_line);
            
        //Echange des lignes si besoin
        if(pivot_line != pivot_row){
            if(verbose > 1) printf("échange des lignes %d et %d\n",lines_link[pivot_line],lines_link[pivot_row]);

            swap_lines(lines_link, pivot_line, pivot_row);
        }


        cudaMemcpy(system, d_system, sizeof(linear_system), cudaMemcpyHostToDevice);

        ApplyPivotToLinesKernel<<<blocksPerGrid, threadsPerBlock>>>(d_system, lines_link, pivot_row);
        //a passer sur GPU
        /*
        for(int i = pivot_row+1; i < system->len; i++){
             apply_pivot_to_line(system, lines_link, i, pivot_row);
        }
        */
        cudaMemcpy(system, d_system, sizeof(linear_system), cudaMemcpyDeviceToHost);
        
    }
    double tac = wtime();

    printf("%lf s Iteratif \n",tac-tic);

    save_solution(system, lines_link, path);
    if(verbose > 0){
        printf("Solution sauvegardée : %.64s\n",path);
    }
    free(lines_link);
    cudaFree(d_system);
}


//main solveur de système 
int main(int argc, char const *argv[])
{

    if(argc < 2){
        printf("Utilisation : ./solver.exe PATH VERBOSE\nAvec :\n\tPATH : Chemin vers le fichier\n\tVERBOSE : Le niveau d'affichage du programme (optionnel, défaut = 1)\n");
        return EXIT_SUCCESS;
    }

    int verbose = 1;

    // Verbose arg ? 
    if (argc > 2)
    {
        verbose = atoi(argv[2]);
    }

    char read_path[64];
    snprintf(read_path,sizeof(read_path),"%s",argv[1]);
    
    
    //Lecture du fichier    
    linear_system* sys;
    
    sys = (linear_system *)malloc(sizeof(linear_system));

    if( ! (read_system(sys,read_path))){
        printf("Erreur lors de la lecture du fichier \n");
        return EXIT_SUCCESS;
    }
    
    char save_path[64];
    snprintf(save_path,sizeof(save_path),"%s_solved.txt",argv[1]); 

    //Calculs & sauvegarde
    
    solve_system(sys,save_path,verbose);
    


   
    //Libération de la mémoire
    free_system(sys);
/*
    #ifdef _OPENMP
    sys = (linear_system *)malloc(sizeof(linear_system));

    read_system(sys,read_path);

    //Calculs & sauvegarde
   
    solve_system_parallel(sys,save_path,verbose);
  

    
    //Libération de la mémoire
    free_system(sys);
    #endif
*/
    return 0;
}
