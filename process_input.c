#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/stat.h>   // for numbering of output files
#include <mpi.h>
#include <assert.h>
#include "admm.h"

extern FILE *output;
extern FILE *traindata;
extern Parameters params;
extern Problem *SP;             
extern Problem *PP;
extern double maxcut_density;
extern int maxcut_size;
extern int BabPbSize;
extern SVM * svm1;
extern SVM * svm2;

// macro to handle the errors in the input reading
#define READING_ERROR(file,cond,message)\
        if ((cond)) {\
            fprintf(stderr, "\nError: "#message"\n");\
            fclose(file);\
            return 1;\
        }


void print_symmetric_matrix(double *Mat, int N) {

    double val;

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            val = (i >= j) ? Mat[i + j*N] : Mat[j + i*N];
            printf("%24.16e", val);
        }
        printf("\n");
    }
}        

int processCommandLineArguments(int argc, char **argv, int rank) {

    int read_error = 0;

    if (argc != 4) {
        if (rank == 0)
            fprintf(stderr, "Usage: ./admm file.rudy file.params jobname\n");
        read_error = 1;
        return read_error;
    }

    //Create training data file
    char traindata_path[200];
    sprintf(traindata_path, "%s_%s.traindata%d", argv[1], argv[3], rank);
    traindata = fopen(traindata_path, "w");
    /***** only master process creates output file and reads the whole graph *****/

    // Control the command line arguments
    if (rank == 0) {

//        // Create the output file
//        char output_path[200];
//        sprintf(output_path, "%s.output", argv[1]);
//
//
//        // Check if the file already exists, if so aappend _<NUMBER> to the end of the output file name
//        struct stat buffer;
//        int counter = 1;
//
//        while (stat(output_path, &buffer) == 0)
//            sprintf(output_path, "%s.output_%d", argv[1], counter++);
//
//        output = fopen(output_path, "w");
//        if (!output) {
//            fprintf(stderr, "Error: Cannot create output file.\n");
//            read_error = 1;
//            MPI_Bcast(&read_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
//            return read_error;
//        }

        // Create the output file
        char output_path[200];
        sprintf(output_path, "%s_%s.output", argv[1], argv[3]);

        output = fopen(output_path, "w");
        if (!output) {
            fprintf(stderr, "Error: Cannot create output file.\n");
            read_error = 1;
            MPI_Bcast(&read_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
            return read_error;
        }


        // Read the input file instance
        read_error = readData(argv[1]);

        // bcast first read_error then whole graph
        MPI_Bcast(&read_error, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (read_error)
            return read_error;
        else {
            MPI_Bcast(&(SP->n), 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(SP->L, SP->n * SP->n, MPI_DOUBLE, 0, MPI_COMM_WORLD); 
            MPI_Bcast(&maxcut_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
            MPI_Bcast(&maxcut_density, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        }



            
    }
    else {
        MPI_Bcast(&read_error, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (read_error) 
            return read_error;    

        // allocate memory for original problem SP and subproblem PP
        alloc(SP, Problem);
        alloc(PP, Problem);

        MPI_Bcast(&(SP->n), 1, MPI_INT, 0, MPI_COMM_WORLD);

        // allocate memory for objective matrices for SP and PP
        alloc_matrix(SP->L, SP->n, double);
        alloc_matrix(PP->L, SP->n, double);

        MPI_Bcast(SP->L, SP->n * SP->n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // IMPORTANT: last node is fixed to 0
        // --> BabPbSize is one less than the size of problem SP
        BabPbSize = SP->n - 1; // num_vertices - 1;
        PP->n = SP->n;

        int N2 = SP->n * SP->n;
        int incx = 1;
        int incy = 1;
        dcopy_(&N2, SP->L, &incx, PP->L, &incy);

        MPI_Bcast(&maxcut_size, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Bcast(&maxcut_density, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }    
    
    //read svm classifier and broadcast
    
    svm1 = read_svm(1);
    svm2 = read_svm(2);    
    // Read the parameters from a user file
    read_error = readParameters(argv[2], rank);
    if (read_error)
        return read_error;

    /* adjust parameters */
    // change number of added cutting planes per iteration to n*10
    if (params.adjust_TriIneq)
        params.TriIneq = SP->n * 10;


    if (rank == 0) {
        // print parameters to output file
        fprintf(output, "Parameters:\n");
#define P(type, name, format, def_value)\
            fprintf(output, "%20s = "format"\n", #name, params.name);
        PARAM_FIELDS
#undef P
    }

    return read_error;
}



/* Read parameters contained in the file given by the argument */
int readParameters(const char *path, int rank) {

    FILE* paramfile;
    char s[128];            // read line
    char param_name[50];

    // Initialize every parameter with its default value
#define P(type, name, format, def_value)\
    params.name = def_value;
    PARAM_FIELDS
#undef P

    // open parameter file
    if ( (paramfile = fopen(path, "r")) == NULL ) {
        if (rank == 0)
            fprintf(stderr, "Error: parameter file %s not found.\n", path);
        return 1;
    }
    
    while (!feof(paramfile)) {
        if ( fgets(s, 120, paramfile) != NULL ) {
        
            // read parameter name
            sscanf(s, "%[^=^ ]", param_name);

            // read parameter value
#define P(type, name, format, def_value)\
            if(strcmp(#name, param_name) == 0)\
            sscanf(s, "%*[^=]="format"\n", &(params.name));
            PARAM_FIELDS
#undef P

        }
    }
    fclose(paramfile);   
   
    return 0;
}


/*** read graph file ***/
int readData(const char *instance) {

    // open input file
    FILE *f = fopen(instance, "r");
    if (f == NULL) {
        fflush(stdout);
        fprintf(stderr, "Error: problem opening input file %s\n", instance);
        return 1;
    }
    printf("Input file: %s\n", instance);	
    fprintf(output,"Input file: %s\n", instance);

    int num_vertices;
    int num_edges;

    READING_ERROR(f, fscanf(f, "%d %d \n", &num_vertices, &num_edges) != 2,
                  "Problem reading number of vertices and edges");
    READING_ERROR(f, num_vertices <= 0, "Number of vertices has to be positive");

    // OUTPUT information on instance
    fprintf(stdout, "\nGraph has %d vertices and %d edges.\n", num_vertices, num_edges);
    fprintf(output, "\nGraph has %d vertices and %d edges.\n", num_vertices, num_edges);

    //get the properties of the adjacency matrix
    maxcut_size = num_vertices;
    maxcut_density = (double) num_edges / num_vertices / (num_vertices - 1) * 2;

    // read edges and store them in matrix Adj
    // NOTE: last node is fixed to 0
    int i, j;
    double weight;


    // Adjacency matrix Adj: allocate and set to 0 
    double *Adj;
    alloc_matrix(Adj, num_vertices, double);

    for (int edge = 0; edge < num_edges; ++edge) {
        
        READING_ERROR(f, fscanf(f, "%d %d %lf \n", &i, &j, &weight) != 3,
                      "Problem reading edges of the graph"); 

        READING_ERROR(f, ((i < 1 || i > num_vertices) || (j < 1 || j > num_vertices)),
                      "Problem with edge. Vertex not in range");  
        
        Adj[ num_vertices * (j - 1) + (i - 1) ] = weight;
        Adj[ num_vertices * (i - 1) + (j - 1) ] = weight;      
    }   

    fclose(f);

    // allocate memory for original problem SP and subproblem PP
    alloc(SP, Problem);
    alloc(PP, Problem);

    // size of matrix L
    SP->n = num_vertices;                 

    // allocate memory for objective matrices for SP and PP
    alloc_matrix(SP->L, SP->n, double);
    alloc_matrix(PP->L, SP->n, double);


    // IMPORTANT: last node is fixed to 0
    // --> BabPbSize is one less than the size of problem SP
    BabPbSize = SP->n - 1; // num_vertices - 1;
    PP->n = SP->n;
    

    /********** construct SP->L from Adj **********/
    /*
     * SP->L = [ Laplacian,  Laplacian*e; (Laplacian*e)',  e'*Laplacian*e]
     */
    // NOTE: we multiply with 1/4 afterwards when subproblems PP are created!
    //       (in function createSubproblem)
    // NOTE: Laplacian is stored in upper left corner of L

    // (1) construct vec Adje = Adj*e 
    double *Adje;
    alloc_vector(Adje, num_vertices, double);

    for (int ii = 0; ii < num_vertices; ++ii) {
        for (int jj = 0; jj < num_vertices; ++jj) {
            Adje[ii] += Adj[jj + ii * num_vertices];
        }
    }

    // (2) construct Diag(Adje)
    double *tmp;
    alloc_matrix(tmp, num_vertices, double);
    Diag(tmp, Adje, num_vertices);

    // (3) fill upper left corner of L with Laplacian = tmp - Adj,
    //     vector parts and constant part      
    double sum_row = 0.0;
    double sum = 0.0;

    // NOTE: skip last vertex!!
    for (int ii = 0; ii < num_vertices; ++ii) {            
        for (int jj = 0; jj < num_vertices; ++jj) {

            // matrix part of L
            if ( (ii < num_vertices - 1) && (jj < num_vertices - 1) ) {
                SP->L[jj + ii * num_vertices] = tmp[jj + ii * num_vertices] - Adj[jj + ii * num_vertices]; 
                sum_row += SP->L[jj + ii * num_vertices];       
            }
            // vector part of L
            else if ( (jj == num_vertices - 1) && (ii != num_vertices - 1)  ) {
                SP->L[jj + ii * num_vertices] = sum_row;
                sum += sum_row;
            }
            // vector part of L
            else if ( (ii == num_vertices - 1) && (jj != num_vertices - 1)  ) {
                SP->L[jj + ii * num_vertices] = SP->L[ii + jj * num_vertices];
            }
            // constant term in L
            else { 
                SP->L[jj + ii * num_vertices] = sum;
            }
        }
        sum_row = 0.0;
    } 

    int N2 = SP->n * SP->n;
    int incx = 1;
    int incy = 1;
    dcopy_(&N2, SP->L, &incx, PP->L, &incy);


    free(Adj);
    free(Adje);
    free(tmp);  

    return 0;
}

char** str_split(char* a_str, const char a_delim)
{
    char** result    = 0;
    size_t count     = 0;
    char* tmp        = a_str;
    char* last_comma = 0;
    char delim[2];
    delim[0] = a_delim;
    delim[1] = 0;

    /* Count how many elements will be extracted. */
    while (*tmp)
    {
        if (a_delim == *tmp)
        {
            count++;
            last_comma = tmp;
        }
        tmp++;
    }

    /* Add space for trailing token. */
    count += last_comma < (a_str + strlen(a_str) - 1);

    /* Add space for terminating null string so caller
       knows where the list of returned strings ends. */
    count++;

    result = malloc(sizeof(char*) * count);

    if (result)
    {
        size_t idx  = 0;
        char* token = strtok(a_str, delim);

        while (token)
        {
            assert(idx < count);
            *(result + idx++) = strdup(token);
            token = strtok(0, delim);
        }
        assert(idx == count - 1);
        *(result + idx) = 0;
    }

    return result;
}




SVM * read_svm(int ncuts){
    char sncuts[10];
    sprintf(sncuts, "%d", ncuts);
    char svmfile[] = "svm"; 
    strcat(svmfile, sncuts);
    strcat(svmfile, ".txt");
    FILE* file = fopen(svmfile, "r"); /* should check the result */
    char line[10000];
    SVM * svm = malloc(sizeof(SVM));
    svm->n_features = 5;
    svm->scaler_mean = malloc( svm->n_features * sizeof(double));
    svm->scaler_std = malloc( svm->n_features * sizeof(double));

    int line_num = 0;
    while (fgets(line, sizeof(line), file)) {
        /* note that fgets don't strip the terminating \n, checking its
           presence would allow to handle lines longer that sizeof(line) */
        // printf("%s", line); 

         // Extract the first token
      char** tokens;
      tokens = str_split(line, ' ');

       if (tokens)
       {
           int i;
           for (i = 0; *(tokens + i); i++)
           {
               
               double data;
               char *eptr;
               data =  strtod(*(tokens + i), &eptr);

               if (line_num == 0){
                  svm->scaler_mean[i] = data;
               }

               else if (line_num == 1){
                  svm->scaler_std[i] = data;
               }

               else if (line_num == 2){
                  svm->intercept = data;
               }

               else if (line_num == 3){
                  svm->gamma = data;
               }               

               else if (line_num == 4){
                  svm->n_support = (int) data;
                  svm->dual_coef = malloc( svm->n_support * sizeof(double));
                  
                  svm->support_vectors = (double **)malloc(svm->n_support * sizeof(double*));
                  for(int i = 0; i < svm->n_support; i++) svm->support_vectors[i] = (double *)malloc(svm->n_features * sizeof(double));
               }

               else {
                  if (i == 0){
                     svm->dual_coef[line_num-5] = data;
                  }
                  else{
                     svm->support_vectors[line_num-5][i-1] = data;
                  }
               }


               free(*(tokens + i));
           }
           free(tokens);
       }
       line_num +=1;   
    }
    /* may check feof here to make a difference between eof and io failure -- network
       timeout for instance */

    fclose(file);
   //print out svm 
   return svm;
}
