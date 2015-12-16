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
#define METHOD BLAS
#define NUM_THREADS 10
//#define DEBUG_PRINT 1

#define GD_ITERATIONS 1000
#define TEST_POINTS 1000000
#define TRAINING_POINTS 25010
#define TEST_FILE "poker-hand-training.data"
#define TRAINING_FILE "poker-hand-testing.data"
#define NO_FEATURE 10
#define NO_OUTPUT 1
#define NO_HIDDEN_NODES 500
#define NO_CLASSES 10

//#define TEST_POINTS 1
//#define TRAINING_POINTS 5
//#define TEST_FILE "x_square_testing.data"
//#define TRAINING_FILE "x_square_training.data"

matrix_t* xTr;
matrix_t* xTr_with_bias;
matrix_t* yTr;

matrix_t* xTe;
matrix_t* xTe_with_bias;
matrix_t* yTe;
matrix_t* hypo_yte;

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
double alpha = 0.0001;
double tolerance = 1.0e-8;

void create_ann(double* xTr_data, double* yTr_data, int n_input, int n_hidden, int n_output,int n);
void feedforward();
matrix_t* add_bias(matrix_t* z);
double compute_loss();
void back_prop();
void normalize_inputs(matrix_t* mat);

void create_ann(double* xTr_data, double* yTr_data, int n_input, int n_hidden, int n_output,int n) {

    xTr = make_mat(n,n_input,xTr_data);
    yTr = make_mat(n,n_output,yTr_data);

    matrix_t* temp = mat_transpose(xTr);
    mat_free(xTr);
    xTr = temp;

    temp = mat_transpose(yTr);
    mat_free(yTr);
    yTr = temp;

    normalize_inputs(xTr);
    normalize_inputs(yTr);

    W_prime = (matrix_t*)malloc(sizeof(matrix_t));
    W_prime->first_dim = n_hidden;
    W_prime->second_dim  = xTr->first_dim + 1;
    W_prime->mat_data = (double *) calloc(1, W_prime->first_dim * W_prime->second_dim * sizeof(double));
    for(int i = 0; i < W_prime->first_dim * W_prime->second_dim; i++) {
        W_prime->mat_data[i] = (double) rand() / RAND_MAX;
    }
    //print_matrix(W_prime);
    W = (matrix_t*)malloc(sizeof(matrix_t));
    W->first_dim = yTr->first_dim;
    W->second_dim  = n_hidden + 1;
    W->mat_data = (double *) calloc(1,W->first_dim * W->second_dim * sizeof(double));
    for(int i = 0; i < W->first_dim * W->second_dim;i++) {
        W->mat_data[i] = (double) rand() / RAND_MAX;
    }
    //print_matrix(W);
    //print_matrix(W_prime);
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
    //print_matrix(W);
    //print_matrix(z_prime_with_bias);
    mat_free(z);
    z = g(a);
    /*printf("\nPrinting Matrix z\n");
    
    for(int i = 0; i < 10; i++){
        printf("i-> %d, yTr-> %.4lf, z-> %.4lf\n", i, yTr->mat_data[i], z->mat_data[i]);
    }
    */
    //print_matrix(z);
    #ifdef DEBUG_PRINT
    printf("exiting feedforward\n");
    #endif

}

void feedforward_for_test_data() {

    #ifdef DEBUG_PRINT
    printf("Entered feedforward_for_test_data\n");
    #endif

    mat_free(xTe_with_bias);
    xTe_with_bias = add_bias(xTe);
    //printf("Printing Matrix xTr_with_bias\n");
    //print_matrix(xTr_with_bias);

    mat_free(a_prime);
    a_prime = mat_mul(W_prime,xTe_with_bias,METHOD);
    //printf("\nPrinting Matrix a_prime\n");

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
    //print_matrix(W);

    mat_free(hypo_yte);
    hypo_yte = g(a);
    //printf("\nPrinting Matrix z\n");
    //print_matrix(hypo_yte);

    #ifdef DEBUG_PRINT
    printf("exiting feedforward_for_test_data\n");
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
        //#pragma omp parallel for shared(z,yTr) reduction(+ : loss_temp)
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
    //print_matrix(der_g_a);
    matrix_t* delta = mat_subtract(z,yTr);
    //print_matrix(delta);
    matrix_t* avg_delta = mat_mul_scalar(delta,1.0/yTr->second_dim);
    //print_matrix(avg_delta);
    matrix_t* delta_2 = mat_mul_element(avg_delta, der_g_a);
    //print_matrix(delta_2);
    mat_free(der_g_a);
    mat_free(delta);
    mat_free(avg_delta);

    //printf("\nPrinting Delta 2\n");
    //print_matrix(delta_2);

    matrix_t* z_prime_trans = mat_transpose(z_prime_with_bias);
    //print_matrix(z_prime_trans);
    matrix_t* delta_W = mat_mul(delta_2,z_prime_trans,METHOD);
    mat_free(z_prime_trans);
    //mat_free(z_prime_with_bias);

    //printf("\nPrinting Matrix delta W\n");
    //print_matrix(delta_W);

    matrix_t* W_trans = mat_transpose(W);
    //printf("\nPrinting Matrix W_trans\n");
    //print_matrix(W);
    matrix_t* W_trans_delta_2 = mat_mul(W_trans,delta_2,METHOD);
    //print_matrix(delta_2);

    //printf("-----after delta 2-------------\n");
    mat_free(W_trans);
    //printf("-----after w_trans-------------\n");
    //printf("\nPrinting Matrix W_trans_delta_2\n");
    //print_matrix(W_trans_delta_2);
    matrix_t* der_f_a_prime = der_f(a_prime);
    //print_matrix(a_prime);
    //print_matrix(der_f_a_prime);
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
    //print_matrix(delta_W);

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
    /*
    #pragma omp parallel num_threads(NUM_THREADS)
    int tno = omp_get_thread_num();
    */
    feedforward();    
    double loss = compute_loss();

    if(loss < tolerance) {
        //printf("\nFinal Loss: %.4lf\n",loss);
        return;
    }
    int flag = 0;

    for(int i = 0; i < max_iter; i++) {

        back_prop();

        feedforward();
        double new_loss = compute_loss();
        if(new_loss <= loss) {
            if(new_loss < tolerance) {
                flag = 1;
                loss = new_loss;
                //printf("\nFinal Loss: %.4lf\n", new_loss);
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

            alpha = 0.99 * alpha;
        }
        //print_matrix(W);
        //printf("\nLoss after Iteration %d: %.4lf\n",i+1,loss);
        //printf("\nalpha:%lf\n",alpha);
    }

    //feedforward();
    //print_matrix(xTr);
    if(flag == 1){
        printf("\nFinal Loss: %.4lf\n", loss);
    }
    else{
        printf("\nFinal Loss after max iters: %.4lf\n", loss);
    }
    

    #ifdef DEBUG_PRINT
    printf("exit gd\n");
    #endif
}

void read_data(double** xTr, double** yTr){
    *xTr = (double*) malloc(sizeof(double) * TRAINING_POINTS * NO_FEATURE);
    *yTr = (double*) malloc(sizeof(double) * TRAINING_POINTS);

    FILE* fp = fopen(TRAINING_FILE, "r");

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
            if(counter <= NO_FEATURE){
                (*xTr)[xTr_pointer++] = (double)(atof(token));
            }
            else{
                (*yTr)[yTr_pointer++] = (double)(atof(token));
            }

            token = strtok(NULL, ",");
        }
    }

    fclose(fp);
    return;
}

void read_test_data(double** xTe, double** yTe){
    *xTe = (double*) malloc(sizeof(double) * TEST_POINTS * NO_FEATURE);
    *yTe = (double*) malloc(sizeof(double) * TEST_POINTS);

    FILE* fp = fopen(TEST_FILE, "r");

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
            if(counter <= NO_FEATURE){
                (*xTe)[xTe_pointer++] = (double) (atof(token));
            }
            else
                (*yTe)[yTe_pointer++] = (double) (atof(token));

            token = strtok(NULL, ",");
        }
    }

    fclose(fp);
    return;
}

double calc_accuracy(matrix_t* hypo_yte, matrix_t* yTe){
    int incorrect = 0;
    int hypo = 0;
    double interval = (double) 1.0 / NO_CLASSES;
    double label_interval = (double) 1.0 / (NO_CLASSES - 1);

    double* label_intervals = (double*) malloc(sizeof(double) * NO_CLASSES); 
    double* interval_vals = (double*) malloc(sizeof(double) * (NO_CLASSES));

    for(int i = 0; i < NO_CLASSES; i++){
        interval_vals[i] = interval*(i+1);
        label_intervals[i] = label_interval*i;
    }

    for(int i = 0; i <= TEST_POINTS; i++){

        /*if(i < 12){
            printf("i-> %d, hypo: %.4lf, yTe: %.4f\n", i, hypo_yte->mat_data[i], yTe->mat_data[i]);
        }*/

        for(int j = 0; j < NO_CLASSES; j++){
            if(hypo_yte->mat_data[i] < interval_vals[j]){
                hypo_yte->mat_data[i] = label_intervals[j];
                break;
            }  
        }
        
        /*if(i < 12){
            printf("i-> %d, hypo: %.4lf, yTe: %.4f\n", i, hypo_yte->mat_data[i], yTe->mat_data[i]);
        }*/
        if(!check_double_eq(floor(hypo_yte->mat_data[i] + 0.5), yTe->mat_data[i])){
            incorrect++;
        }
        if(!check_double_eq(hypo_yte->mat_data[i], 0.6212)){
            hypo++;
        }
    }

    //printf("\n----Hypo= %d------\n", hypo);

    double acc = ((double)(TEST_POINTS - incorrect) / TEST_POINTS) * 100;
    return acc;
}

double calc_accuracy_train(matrix_t* hypo_yte, matrix_t* yTe){
    int incorrect = 0;

    for(int i = 0; i <= TRAINING_POINTS; i++){
        if(i < 10){
            printf("i-> %d, hypo: %.4lf, yTe: %.4f\n", i, hypo_yte->mat_data[i], yTe->mat_data[i]);
        }
        if(!check_double_eq(floor(hypo_yte->mat_data[i] + 0.5), yTe->mat_data[i])){
            incorrect++;
        }
    }

    double acc = ((double)(TRAINING_POINTS - incorrect) / TRAINING_POINTS) * 100;
    return acc;
}

void normalize_inputs(matrix_t* mat){

    for(int i = 0; i < mat->first_dim; i++){
        double max = DBL_MIN;
        double min = DBL_MAX;

        for(int j = 0; j < mat->second_dim; j++){
            if(mat->mat_data[MATRIX_ACCESS(i,j,mat->second_dim)] < min){
                min = mat->mat_data[MATRIX_ACCESS(i,j,mat->second_dim)];
            }
            if(mat->mat_data[MATRIX_ACCESS(i,j,mat->second_dim)] > max){
                max = mat->mat_data[MATRIX_ACCESS(i,j,mat->second_dim)];
            }
        }

        for(int j = 0; j < mat->second_dim; j++){
            double temp = mat->mat_data[MATRIX_ACCESS(i,j,mat->second_dim)];
            mat->mat_data[MATRIX_ACCESS(i,j,mat->second_dim)] = (temp - min) / (max - min);
        }
    }
}

int main(int argc, char** argv){

    double* xTr_data;
    double* yTr_data;
    double* xTe_data;
    double* yTe_data;

    read_data(&xTr_data, &yTr_data);
    read_test_data(&xTe_data, &yTe_data);

    xTe = make_mat(TEST_POINTS, NO_FEATURE, xTe_data);
    yTe = make_mat(TEST_POINTS, NO_OUTPUT, yTe_data);

    matrix_t* t1 = mat_transpose(xTe);
    mat_free(xTe);
    xTe = t1;

    t1 = mat_transpose(yTe);
    mat_free(yTe);
    yTe = t1;

    normalize_inputs(xTe);
    normalize_inputs(yTe);

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

        create_ann(xTr_data,yTr_data,NO_FEATURE,NO_HIDDEN_NODES,NO_OUTPUT,TRAINING_POINTS);

        double t0 = omp_get_wtime();
        gradient_descent(GD_ITERATIONS);
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
            feedforward_for_test_data();
            //feedforward();
            normalize_inputs(hypo_yte);
            double accuracy = calc_accuracy(hypo_yte, yTe);

            printf("\n----Accuracy for Test Data: %.2lf\n", accuracy);
        }
    }
    return 0;
}
