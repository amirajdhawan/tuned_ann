#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <omp.h>
#include <limits.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>

#include "utilities.h"

//Debugging level required
#define DEBUG_LEVEL 1
#define METHOD NAIVE
#define NUM_THREADS 10
//#define DEBUG_PRINT 1

matrix_t* xTr;
matrix_t* xTr_with_bias;
matrix_t* yTr;

matrix_t* W_prime;
matrix_t* W;

matrix_t* alpha_delta_W_prime;
matrix_t* alpha_delta_W = NULL;

matrix_t* a_prime;
matrix_t* z_prime;
matrix_t* z_prime_with_bias;
matrix_t* a;
matrix_t* z;
matrix_t* (*f)(matrix_t*);
matrix_t* (*g)(matrix_t*);
matrix_t* (*der_f)(matrix_t*);
matrix_t* (*der_g)(matrix_t*);
double alpha = 0.1;
double tolerance = 1.0e-2;

void create_ann(double* xTr_data, double* yTr_data, int n_input, int n_hidden, int n_output,int n);
void feedforward();
matrix_t* add_bias(matrix_t* z);
double compute_loss();
void back_prop();

void create_ann(double* xTr_data, double* yTr_data, int n_input, int n_hidden, int n_output,int n) {

    xTr = make_mat(n_input,n,xTr_data);
    yTr = make_mat(n_output,n,yTr_data);
    
    W_prime = (matrix_t*)malloc(sizeof(matrix_t));
    W_prime->first_dim = n_hidden;
    W_prime->second_dim  = xTr->first_dim + 1;
    W_prime->mat_data = (double *) calloc(1,W_prime->first_dim * W_prime->second_dim * sizeof(double));
    for(int i = 0; i < W_prime->first_dim;i++) {
        W_prime->mat_data[MATRIX_ACCESS(i,W_prime->second_dim - 1,W_prime->second_dim)] = 1;
    }
    
    W = (matrix_t*)malloc(sizeof(matrix_t));
    W->first_dim = yTr->first_dim;
    W->second_dim  = n_hidden + 1;
    W->mat_data = (double *) calloc(1,W->first_dim * W->second_dim * sizeof(double));
    for(int i = 0; i < W->first_dim;i++) {
        W->mat_data[MATRIX_ACCESS(i,W->second_dim - 1,W_prime->second_dim)] = 1;
    }
    
    f = &activation_func_mat;
    g = &activation_func_mat;
    
    der_f = &der_activation_func_mat;
    der_g = &der_activation_func_mat;

}

void feedforward() {

    #ifdef DEBUG_PRINT
    printf("Entered feedforward\n");
    #endif

    mat_free(xTr_with_bias);
    xTr_with_bias = add_bias(xTr);
    //printf("Printing Matrix xTr_with_bias\n");
    //print_matrix(xTr_with_bias);
    
    mat_free(a_prime);
    a_prime = mat_mul(W_prime,xTr_with_bias,METHOD);
    //printf("\nPrinting Matrix a_prime\n");
    //print_matrix(a_prime);
    
    mat_free(z_prime);
    z_prime = f(a_prime);
    //printf("\nPrinting Matrix z_prime\n");
    //print_matrix(z_prime);
    
    mat_free(z_prime_with_bias);
    z_prime_with_bias = add_bias(z_prime);
    //printf("\nPrinting Matrix z_prime_with_bias\n");
    //print_matrix(z_prime_with_bias);
    
    mat_free(a);
    a = mat_mul(W,z_prime_with_bias,METHOD);
    //printf("\nPrinting Matrix a\n");
    //print_matrix(a);
    
    mat_free(z);
    z = g(a);
    //printf("\nPrinting Matrix z\n");
    //print_matrix(z);

    #ifdef DEBUG_PRINT
    printf("exiting feedforward\n");
    #endif

}

matrix_t* add_bias(matrix_t* z) {
    matrix_t* z_with_bias = (matrix_t*)malloc(sizeof(matrix_t));
    z_with_bias->first_dim = z->first_dim + 1;
    z_with_bias->second_dim = z->second_dim;
    z_with_bias->mat_data = (double *)malloc(z_with_bias->first_dim * z_with_bias->second_dim * sizeof(double));
    memcpy(z_with_bias->mat_data,z->mat_data,z->first_dim * z->second_dim * sizeof(double));
    for(int j = 0; j < z_with_bias->second_dim;j++) {
        z_with_bias->mat_data[MATRIX_ACCESS(z_with_bias->first_dim - 1,j,z_with_bias->second_dim)] = 1;
    }
    return z_with_bias;
}

double compute_loss() {

    matrix_t* loss = (matrix_t*)malloc(sizeof(matrix_t));
    loss->first_dim = yTr->first_dim;
    loss->second_dim = 1;
    loss->mat_data = (double *)malloc(loss->first_dim * loss->second_dim * sizeof(double));
    
    double avg_loss = 0;
    for(int i = 0; i < yTr->first_dim; i++) {
        double loss_temp = 0;
        #pragma omp parallel for shared(z,yTr) reduction(+ : loss_temp)
        for(int j = 0; j < yTr->second_dim; j++) {
            loss_temp += pow(z->mat_data[MATRIX_ACCESS(i, j, z->second_dim)]
               - yTr->mat_data[MATRIX_ACCESS(i, j, yTr->second_dim)],2);
        }
        loss->mat_data[i] = loss_temp;
        loss->mat_data[i] /= (2 * yTr->second_dim);
        avg_loss += loss->mat_data[i];
    }
    avg_loss /= yTr->first_dim;
    
    //printf("\nPrinting Matrix loss\n");
    //print_matrix(loss);
    mat_free(loss);
    return avg_loss;
}

void back_prop() {

    #ifdef DEBUG_PRINT
    printf("Entered backprop\n");
    #endif

    matrix_t* der_g_a = der_g(a);
    matrix_t* delta = mat_subtract(z,yTr);
    matrix_t* avg_delta = mat_mul_scalar(delta,1.0/yTr->second_dim);
    matrix_t* delta_2 = mat_mul_element(avg_delta, der_g_a);
    mat_free(der_g_a);
    mat_free(delta);
    mat_free(avg_delta);

    //printf("\nPrinting Delta 2\n");
    //print_matrix(delta_2);
    
    matrix_t* z_prime_trans = mat_transpose(z_prime_with_bias);
    matrix_t* delta_W = mat_mul(delta_2,z_prime_trans,METHOD);
    mat_free(z_prime_trans);
    //mat_free(z_prime_with_bias);

    //printf("\nPrinting Matrix delta W\n");
    //print_matrix(delta_W);
    
    matrix_t* W_trans = mat_transpose(W);
    //printf("\nPrinting Matrix W_trans\n");
    //print_matrix(W_primetrans);
    matrix_t* W_trans_delta_2 = mat_mul(W_trans,delta_2,METHOD);
    mat_free(delta_2);
    //printf("-----after delta 2-------------\n");
    mat_free(W_trans);
    //printf("-----after w_trans-------------\n");
    //printf("\nPrinting Matrix W_trans_delta_2\n");
    //print_matrix(W_trans_delta_2);
    matrix_t* der_f_a_prime = der_f(a_prime);
    matrix_t* der_f_a_prime_with_bias = add_bias(der_f_a_prime);
    mat_free(der_f_a_prime);
    
    //printf("\nPrinting Matrix der_f a_prime\n");
    //print_matrix(der_f_a_prime_with_bias);
    matrix_t* avg_delta_1 = mat_mul_element(der_f_a_prime_with_bias,W_trans_delta_2);
    matrix_t* delta_1 = mat_mul_scalar(avg_delta_1,1.0/xTr->second_dim);
    delta_1->first_dim = delta_1->first_dim-1;
    mat_free(der_f_a_prime_with_bias);
    mat_free(avg_delta_1);
    
    //printf("\nPrinting Matrix delta_1\n");
    //print_matrix(delta_1);
    
    mat_free(alpha_delta_W);
    alpha_delta_W = mat_mul_scalar(delta_W,alpha);
    matrix_t* new_W = mat_subtract(W, alpha_delta_W);
    mat_free(W);
    W = new_W;

    mat_free(delta_W);
    mat_free(W_trans_delta_2);
    //printf("\nPrinting Matrix W\n");
    //print_matrix(W);
    
    matrix_t* x_trans = mat_transpose(xTr_with_bias);
    matrix_t* delta_W_prime = mat_mul(delta_1,x_trans,METHOD);
    mat_free(delta_1);
    mat_free(x_trans);

    //printf("\nPrinting Matrix delta W_prime\n");
    //print_matrix(delta_W_prime);
    mat_free(alpha_delta_W_prime);
    alpha_delta_W_prime = mat_mul_scalar(delta_W_prime,alpha);
    matrix_t* new_W_prime = mat_subtract(W_prime,alpha_delta_W_prime);
    mat_free(W_prime);
    W_prime = new_W_prime;
    mat_free(delta_W_prime);

    //printf("\nPrinting Matrix W_prime\n");
    //print_matrix(W_prime);

    #ifdef DEBUG_PRINT
    printf("exiting backprop\n");
    #endif
}

void gradient_descent(int max_iter) {

    #ifdef DEBUG_PRINT
    printf("Entered gd\n");
    #endif

    //#pragma omp parallel num_threads(NUM_THREADS)
    //int tno = omp_get_thread_num();
    feedforward();    
    double loss = compute_loss();
    
    if(loss < tolerance) {
        //printf("\nFinal Loss: %.4lf\n",loss);
        return;
    }

    for(int i = 0; i < max_iter; i++) {

        back_prop();
        
        feedforward();
        double new_loss = compute_loss();
        if(new_loss <= loss) {
            if(new_loss < tolerance) {
                //printf("\nFinal Loss: %.4lf\n",new_loss);
                break;
            }
            loss = new_loss;
            alpha = 1.01 * alpha;
        } else {

            matrix_t* new_W = mat_add(W,alpha_delta_W);
            mat_free(W);
            W = new_W;
            matrix_t* new_W_prime = mat_add(W_prime,alpha_delta_W_prime);
            mat_free(W_prime);
            W_prime = new_W_prime;
            
            alpha = 0.5 * alpha;
        }
        //printf("\nLoss after Iteration %d: %.4lf\n",i+1,loss);
        //printf("\nalpha:%lf\n",alpha);
    }
    #ifdef DEBUG_PRINT
    printf("exit gd\n");
    #endif
}

void read_data(double** xTr, double** yTr){
    *xTr = (double*) malloc(sizeof(double) * 10000000);
    *yTr = (double*) malloc(sizeof(double) * 1000000);

    FILE* fp = fopen("poker-hand-training.data", "r");

    if(fp == NULL){
        printf("Unable to read the file!");
        exit(0);
    }

    char line[256];
    int xTr_pointer = 0;
    int yTr_pointer = 0;
    int counter = 0;
    char *token;
    
    while(fgets(line, sizeof(line), fp)){
        counter = 0;
        token = NULL;

        token = strtok(line, ",");

        while(token != NULL){

            counter++;
            if(counter < 11){
                (*xTr)[xTr_pointer++] = atof(token);
            }
            else
                (*yTr)[yTr_pointer++] = atof(token);
            
            token = strtok(NULL, ",");
        }
    }

    fclose(fp);
    return;
}

void read_test_data(double** xTe, double** yTe){
    *xTr = (double*) malloc(sizeof(double) * 250100);
    *yTr = (double*) malloc(sizeof(double) * 25010);

    FILE* fp = fopen("poker-hand-testing.data", "r");

    if(fp == NULL){
        printf("Unable to read the file!");
        exit(0);
    }

    char line[256];
    int xTe_pointer = 0;
    int yTe_pointer = 0;
    int counter = 0;
    char *token;
    
    while(fgets(line, sizeof(line), fp)){
        counter = 0;
        token = NULL;

        token = strtok(line, ",");

        while(token != NULL){

            counter++;
            if(counter < 11){
                (*xTe)[xTe_pointer++] = atof(token);
            }
            else
                (*yTe)[yTe_pointer++] = atof(token);
            
            token = strtok(NULL, ",");
        }
    }

    fclose(fp);
    return;
}

int main(int argc, char** argv){

    double* xTr_data;
    double* yTr_data;
    double* xTe_data;
    double* yTe_data;

    read_data(&xTr_data, &yTr_data);
    read_test_data(&xTe_data, &yTe_data);

    /*for(int i = 0; i < 20; i++){
        printf("xtr %d -> %.1f\t", i, xTr_data[i]);
    }
    printf("\n\n");
    for(int i = 0; i < 2; i++){
        printf("ytr %d -> %.1f\t", i, yTr_data[i]);
    }
    printf("\n\n");*/
    
    /*for(int i = 0; i < 11000000; i++) {
        xTr_data[i] = i;
    }
    
    for(int i = 0; i < 1100000; i++) {
        yTr_data[i] = i;
    }
    */

    printf("Training the neural network");
    fflush(stdout);

    int tid = fork();

    if(tid > 0){
        int status = 0;
        while((waitpid(-1, &status, WNOHANG)) <= 0){
            printf(".");
            fflush(stdout);
            sleep(2);
        }
    }
    else{
        create_ann(xTr_data,yTr_data,10,100,1,1025010);

        double t0 = omp_get_wtime();
        gradient_descent(10);
        double t1 = omp_get_wtime();

        printf("\nDone training!\n");
        printf("Time Taken: %lf\n",t1-t0);

        printf("Now running Testing data");
        fflush(stdout);

        int nid = fork();

        if(nid > 0){
            int status = 0;
            while((waitpid(-1, &status, WNOHANG)) <= 0){
                printf(".");
                fflush(stdout);
                sleep(2);
            }
        }
        else{
            //Run testing
            
        }
    }
    return 0;
}
