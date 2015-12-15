#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <omp.h>

#include "utilities.h"

#ifndef BLOCK_SIZE
#define BLOCK_SIZE ((int) 32)
#endif

//#define DEBUG_PRINT 1

//Striped Parallel Matrix Multiplication

/*
 A is M-by-K
 B is K-by-N
 C is M-by-N
 */

//Performs Multiplication of a row stripe of A with a column stripe of B to compute a square tile of C
//Restrict Pointers: restrict is used to limit effects of pointer aliasing, aiding optimizations
void basic_dgemm(const int lda_row, const int lda_mid, const int lda_col, const int M, const int K, const int N,
                 const double * restrict A, const double * restrict B, double * restrict C)
{
    //printf("M: %d, K %d, N %d\n",M,K,N);
    int i, j, k;
    
    //Copy Optimization: Memory Aligned Buffers for Matrix Blocks
    double A_buf[ BLOCK_SIZE * lda_mid ] __attribute__((aligned( 32 )));
    double B_buf[ lda_mid * BLOCK_SIZE ] __attribute__((aligned( 32 )));
    double C_buf[ BLOCK_SIZE * BLOCK_SIZE ] __attribute__((aligned( 32 )));
    
    //---Copy Blocks of A,B and C into A_buf, B_buf and C_buf respectively---//
    
    //printf("A_buf\n");
    for( i = 0; i < M; ++i ){
        for( k = 0; k < lda_mid; ++k ) {
            //printf("%.1f ",A[ lda_mid * i + k ]);
            A_buf[ lda_mid * i + k ] = A[ lda_mid * i + k ];
            //printf("%.1f ",A_buf[ lda_mid * i + k ]);
        }
        //printf("\n");
    }
    
    //Copy Block of B into B_buf in transpose form to aid vectorization of matrix multiply's innermost loop
    //printf("B_buf\n");
    for( j = 0; j < N; ++j ) {
        for( k = 0; k < lda_mid; ++k ) {
            B_buf[ k + lda_mid * j ] = B[ lda_col * k + j ];
            //printf("%.1f ",B_buf[ k + lda_mid * j ]);
        }
        //printf("\n");
    }
    
    for( i = 0; i < M; ++i ) {
        for( j = 0; j < N; ++j ) {
            C_buf[ N * i + j ] = C[ lda_col * i + j ];
        }
    }
    
    //---End of Copy Optimization---//
    
    //---Vectorized Matrix Multiply Kernel: Dot Product of Rows of A_buf with Columns of B_buf
    // to produce Columns of C_buf---//
    
    for ( i = 0; i < M; ++i ) {
        for ( j = 0; j < N; ++j ) {
            
            #pragma vector aligned
            for ( k = 0; k < lda_mid; ++k ) {
                C_buf[ N * i + j ] += A_buf[ lda_mid * i + k ] * B_buf[ k + lda_mid * j ];
            }
            
        }
    }
    
    //---End of Matrix Multiply Kernel---//
    
    //---Copy back the computed C_buf block into C---//
    
    for( i = 0; i < M; ++i ) {
        for( j = 0; j < N; ++j ) {
            C[ lda_col * i + j ] = C_buf[ N * i + j ];
        }
    }
    
    //---End of Copy Back---//
}

//Calls the dgemm function using appropriate stripes of A and B to compute a block of C
void do_block(const int lda_row, const int lda_mid, const int lda_col,
              const double * restrict A, const double * restrict B, double * restrict C,
              const int i, const int j)
{
    const int M = ( i + BLOCK_SIZE > lda_row ? lda_row-i : BLOCK_SIZE );
    const int K = lda_mid;
    const int N = ( j + BLOCK_SIZE > lda_col ? lda_col-j : BLOCK_SIZE );
    
    basic_dgemm( lda_row, lda_mid, lda_col, M, K, N,
                A + lda_mid * i, B + j, C + lda_col * i + j );
}

//The Parallel Striped Matrix Matrix Multiply Algorithm to Multiple an M x K matrix A with a K x N matrix B to compute an M x N matrix C
void dgemm(const int M, const int K, const int N, const double * restrict A, const double * restrict B, double * restrict C)
{
    const int n_blocks_row = M / BLOCK_SIZE + ( M % BLOCK_SIZE ? 1 : 0 );
    const int n_blocks_col = N / BLOCK_SIZE + ( N % BLOCK_SIZE ? 1 : 0 );
    #pragma omp parallel for shared(A,B,C)
    for ( int bi = 0; bi < n_blocks_row; ++bi ) {
        const int i = bi * BLOCK_SIZE;
        for ( int bj = 0; bj < n_blocks_col; ++bj ) {
            const int j = bj * BLOCK_SIZE;
            do_block( M, K, N, A, B, C, i, j );
        }
    }
}

void dgemm_naive(matrix_t* matrix_a, matrix_t* matrix_b, matrix_t* matrix_c) {
    
    #ifdef DEBUG_PRINT
    printf("Entered dgemm naive\n");
    #endif

    for(int i = 0; i < matrix_a->first_dim; i++){
        for(int j = 0; j < matrix_b->second_dim; j++){
            double sum = 0;
            for(int k = 0; k < matrix_a->second_dim; k++){
                sum += matrix_a->mat_data[MATRIX_ACCESS(i, k, matrix_a->second_dim)] * matrix_b->mat_data[MATRIX_ACCESS(k, j, matrix_b->second_dim)];
            }
            matrix_c->mat_data[MATRIX_ACCESS(i, j, matrix_c->second_dim)] = sum;
        }
    }

    #ifdef DEBUG_PRINT
    printf("Exiting dgemm naive\n");
    #endif
}

matrix_t* mat_mul(matrix_t* matrix_a, matrix_t* matrix_b, int method){

    #ifdef DEBUG_PRINT
    printf("Mat mul called\n");
    #endif

	if(matrix_a->second_dim != matrix_b->first_dim){
		return NULL;
	}

	matrix_t* matrix_c = (matrix_t*) malloc(sizeof(matrix_t));

	matrix_c->first_dim = matrix_a->first_dim;
	matrix_c->second_dim = matrix_b->second_dim;

	matrix_c->mat_data = (double*)malloc(matrix_c->first_dim * matrix_c->second_dim * sizeof(double));
    
    if(method == OPTIMAL) {
        dgemm(matrix_a->first_dim, matrix_a->second_dim, matrix_b->second_dim,
                 matrix_a->mat_data, matrix_b->mat_data, matrix_c->mat_data);
    } else {
        dgemm_naive(matrix_a,matrix_b,matrix_c);
    }

    #ifdef DEBUG_PRINT
    printf("exiting mat mul\n");
    #endif

	return matrix_c;
}

matrix_t* mat_transpose(matrix_t* mat){
	
	matrix_t* trans_mat = (matrix_t*) malloc(sizeof(matrix_t));
	trans_mat->mat_data = (double*) malloc(sizeof(double) * mat->first_dim * mat->second_dim);

	trans_mat->first_dim = mat->second_dim;
	trans_mat->second_dim = mat->first_dim;

	for(int i = 0; i < mat->first_dim; i++){
		for(int j = 0; j < mat->second_dim; j++){
			trans_mat->mat_data[MATRIX_ACCESS(j, i, trans_mat->second_dim)] = mat->mat_data[MATRIX_ACCESS(i, j, mat->second_dim)];
		}
	}

	return trans_mat;
}

matrix_t* mat_add(matrix_t* A, matrix_t* B){
    
    matrix_t* C = (matrix_t*) malloc(sizeof(matrix_t));
    C->mat_data = (double*) malloc(sizeof(double) * A->first_dim * A->second_dim);
    
    C->first_dim = A->first_dim;
    C->second_dim = A->second_dim;
    
    for(int i = 0; i < A->first_dim; i++){
        for(int j = 0; j < A->second_dim; j++){
            C->mat_data[MATRIX_ACCESS(i, j, C->second_dim)] = A->mat_data[MATRIX_ACCESS(i, j, A->second_dim)] + B->mat_data[MATRIX_ACCESS(i, j, B->second_dim)];
        }
    }
    
    return C;
}

matrix_t* mat_subtract(matrix_t* A, matrix_t* B){
    
    matrix_t* C = (matrix_t*) malloc(sizeof(matrix_t));
    C->mat_data = (double*) malloc(sizeof(double) * A->first_dim * A->second_dim);
    
    C->first_dim = A->first_dim;
    C->second_dim = A->second_dim;
    
    for(int i = 0; i < A->first_dim; i++){
        for(int j = 0; j < A->second_dim; j++){
            C->mat_data[MATRIX_ACCESS(i, j, C->second_dim)] = A->mat_data[MATRIX_ACCESS(i, j, A->second_dim)] - B->mat_data[MATRIX_ACCESS(i, j, B->second_dim)];
        }
    }
    
    return C;
}

matrix_t* mat_mul_element(matrix_t* A, matrix_t* B){
    
    matrix_t* C = (matrix_t*) malloc(sizeof(matrix_t));
    C->mat_data = (double*) malloc(sizeof(double) * A->first_dim * A->second_dim);
    
    C->first_dim = A->first_dim;
    C->second_dim = A->second_dim;
    
    for(int i = 0; i < A->first_dim; i++){
        for(int j = 0; j < A->second_dim; j++){
            C->mat_data[MATRIX_ACCESS(i, j, C->second_dim)] = A->mat_data[MATRIX_ACCESS(i, j, A->second_dim)] * B->mat_data[MATRIX_ACCESS(i, j, B->second_dim)];
        }
    }
    
    return C;
}

matrix_t* mat_mul_scalar(matrix_t* M, double alpha){
    
    matrix_t* C = (matrix_t*) malloc(sizeof(matrix_t));
    C->mat_data = (double*) malloc(sizeof(double) * M->first_dim * M->second_dim);
    
    C->first_dim = M->first_dim;
    C->second_dim = M->second_dim;
    
    for(int i = 0; i < M->first_dim; i++){
        for(int j = 0; j < M->second_dim; j++){
            C->mat_data[MATRIX_ACCESS(i, j, C->second_dim)] =  M->mat_data[MATRIX_ACCESS(i, j, M->second_dim)] * alpha;
        }
    }
    
    return C;
}


matrix_t* make_mat(int first_dim, int second_dim, double* data){
    
    matrix_t* new_mat = (matrix_t*) malloc(sizeof(matrix_t));
    new_mat->first_dim = first_dim;
    new_mat->second_dim = second_dim;
    
    new_mat->mat_data = data;
    return new_mat;
}

void print_matrix(matrix_t* mat){

	printf("\n---Printing Matrix---\n");

	for(int i = 0; i < mat->first_dim; i++){
		for(int j = 0; j < mat->second_dim; j++){
			printf(" %.4f", mat->mat_data[MATRIX_ACCESS(i, j, mat->second_dim)]);
		}
		printf("\n");
	}
}

double activation_func(double value){
	//Using Rectified Linear Units (ReLU)
	return value >= 0 ? value : 0;
}

double der_activation_func(double value){
    //Derivative of the activation function
    return value >= 0 ? 1 : 0;
}

matrix_t* activation_func_mat(matrix_t* M) {
    matrix_t* N = (matrix_t*) malloc(sizeof(matrix_t));
    N->first_dim = M->first_dim;
    N->second_dim = M->second_dim;
    N->mat_data = (double *)malloc(N->first_dim * N->second_dim * sizeof(double));
    for (int i = 0; i < M->first_dim * M->second_dim; i++){
        N->mat_data[i] = activation_func(M->mat_data[i]);
    }
    return N;
}

matrix_t* der_activation_func_mat(matrix_t* M) {
    matrix_t* N = (matrix_t*) malloc(sizeof(matrix_t));
    N->first_dim = M->first_dim;
    N->second_dim = M->second_dim;
    N->mat_data = (double *)malloc(N->first_dim * N->second_dim * sizeof(double));
    for (int i = 0; i < M->first_dim * M->second_dim; i++){
        N->mat_data[i] = der_activation_func(M->mat_data[i]);
    }
    return N;
}

bool check_double_eq(double value1, double value2){
	return (fabs(value1 - value2) < DOUBLE_ERROR) ? true : false;
}

bool check_matrix_same(matrix_t* mat_a, matrix_t* mat_b){
	if(mat_a->first_dim != mat_b->first_dim || mat_a->second_dim != mat_b->second_dim){
		return false;
	}

	for (int i = 0; i < mat_a->first_dim * mat_a->second_dim; i++){
		
		if(!check_double_eq(mat_a->mat_data[i], mat_b->mat_data[i])){
			return false; 
		}
	}
	return true;
}

void mat_free(matrix_t* a){
    if(a == NULL)
        return;
    free(a->mat_data);
    free(a);
    return;
}