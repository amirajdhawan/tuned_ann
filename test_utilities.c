#include <stdio.h>
#include <stdlib.h>
#include "utilities.h"

int main(int argc, char** argv){
	
	bool error_flag = false;

	double mata[6] = {1,2,3,4,5,6};
	double matb[6] = {7,8,9,10,11,12};
	double transa[6] = {1,4,2,5,3,6};
	double multiab[4] = {58, 64, 139, 154};
	
	matrix_t* matrix_a = make_mat(2, 3, mata);
	matrix_t* matrix_b = make_mat(3, 2, matb);

	printf("Printing Matrix A\n");
	print_matrix(matrix_a);

	printf("\nPrinting Matrix B\n");
	print_matrix(matrix_b);
    
    //Testing mat_transpose
    printf("\nTesting check_matrix_same on A\n");

	/* Testing Matrix Check same */
	if(!check_matrix_same(make_mat(2, 3, mata), make_mat(2, 3, mata))){
		error_flag = true;
		printf("!!! Check Matrix same is incorrect !!!\n");
	}
    printf("check_matrix_same is correct\n");
    
    //Testing mat_transpose
    printf("\nTesting mat_transpose with input matrix A\n");

	/* Testing Matrix Transpose */
	matrix_t* c = mat_transpose(matrix_a);
	if(!check_matrix_same(c, make_mat(3, 2, transa))){
		error_flag = true;
		printf("!!! Matrix transpose is incorrect !!!\n");
	}
    
    printf("mat_transpose with input matrix A is correct\n");
		
	/* Testing Matrix Multiplication */
    
    printf("\nTesting mat_mul of A & B\n");
	matrix_t* d = mat_mul(matrix_a, matrix_b, OPTIMAL);
	if(!check_matrix_same(d, make_mat(2, 2, multiab))){
		error_flag = true;
		printf("!!! Matrix Multiplication is incorrect !!!\n");
	}
    
    printf("mat_mul of A & B is correct\n");
    
    printf("\nTesting mat_mul of Large Square Matrices\n");

    double mat_large1[100],mat_large2[100];
    for(int i = 0; i < 10;i++) {
        for(int j = 0; j < 10;j++) {
            mat_large1[i*10 + j] = i+j;
            mat_large2[i*10 + j] = i+j;
        }
    }
    
    matrix_t* matrix_large1 = make_mat(100, 1, mat_large1);
    matrix_t* matrix_large2 = make_mat(1, 100, mat_large2);
    
    matrix_t* e = mat_mul(matrix_large1, matrix_large2,OPTIMAL);
    matrix_t* f = mat_mul(matrix_large1, matrix_large2,NAIVE);
    if(!check_matrix_same(e,f)){
        error_flag = true;
        printf("!!! Matrix Multiplication is incorrect !!!\n");
    }
    
    printf("mat_mul of Large Square Matrices is correct\n");

	if(error_flag){
		printf("At least one error. Testing failed\n");
	}

	free(matrix_a);
	free(matrix_b);
	free(c);
	free(d);
    free(e);
    free(f);
	return 0;
}