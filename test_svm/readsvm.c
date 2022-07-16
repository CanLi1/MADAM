#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>

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


typedef struct SVM {
   int n_support;
   int n_features;
   double gamma;
   double * scaler_mean;
   double * scaler_std;
   double ** support_vectors;
   double * dual_coef;
   double intercept;
} SVM;

SVM * read_svm(){
   FILE* file = fopen("svm.txt", "r"); /* should check the result */

    char line[10000];
    SVM * svm = malloc(sizeof(SVM));
    svm->n_features = 6;
    svm->scaler_mean = malloc( svm->n_features * sizeof(double));
    svm->scaler_std = malloc( svm->n_features * sizeof(double));
  
    // svm->scaler_std[5] = 1.2;
    // printf("%.3f\n", svm->scaler_std[5]);
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

double svm_predict(SVM * svm, double * x){
   // for (int i =0; i<svm->n_features; i++){
   //    x[i] = (x[i]- svm->scaler_mean[i]) / svm->scaler_std[i];
   // }
   double val = svm->intercept;
   double gamma = svm->gamma;
   for (int i = 0; i < svm->n_support; i++){
      double dot_product = 0.0;
      for (int j=0; j< svm->n_features; j++){
      dot_product += (x[j] - svm->support_vectors[i][j]) * (x[j] - svm->support_vectors[i][j]);
   }
   val += svm->dual_coef[i] * exp(-gamma * dot_product);
   }
   return val;
}

int main()
{
   SVM * svm = read_svm();
   double x[6] = {1, 2.0, 3, 4, 5, 6};
   double val = svm_predict(svm, x);
   printf("%.3f\n", val);
   // printf("the scale of the problem is\n");
   // for (int i = 0; i< svm->n_features; i++){
   //    printf("mean %.3f std %.3f\n", svm->scaler_mean[i], svm->scaler_std[i]);
   // }

   // for(int i=0; i< svm->n_support; i++){
   //    printf("the %d th dual coefficient is %.3f\n", i, svm->dual_coef[i]);
   //    printf("the %d th support vector", i);
   //    for(int j=0; j< svm->n_features; j++){
   //       printf("%.3f ", svm->support_vectors[i][j]);
   //    }
   //    printf("\n");
   // }

    return 0;
}



