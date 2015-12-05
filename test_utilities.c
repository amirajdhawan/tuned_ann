#include <stdio.h>
#include <stdlib.h>
#include "utilities.h"

matrix_t* make_mat(int first_dim, int second_dim, double* data){

	matrix_t* new_mat = (matrix_t*) malloc(sizeof(matrix_t));
	new_mat->first_dim = first_dim;
	new_mat->second_dim = second_dim;

	new_mat->mat_data = data;

	return new_mat;
}

int main(int argc, char** argv){
	
	bool error_flag = false;

	double mata[6] = {1,2,3,4,5,6};
	double matb[6] = {7,8,9,10,11,12};
	double transa[6] = {1,4,2,5,3,6};
	double multiab[4] = {58, 64, 139, 154};
	
	matrix_t* matrix_a = make_mat(2, 3, mata);
	matrix_t* matrix_b = make_mat(3, 2, matb);

	/*
	matrix_a.first_dim = 2;
	matrix_a.second_dim = 3;
	matrix_a.mat_data = arr1;

	matrix_b.first_dim = 3;
	matrix_b.second_dim = 2;
	matrix_b.mat_data = arr2;*/

	/*printf("Printing Matrix A\n");
	print_matrix(&matrix_a);

	printf("\nPrinting Matrix B\n");
	print_matrix(&matrix_b);*/

	//Testing mat_transpose
	/*printf("\nTesting mat_transpose with input matrix a\n");*/
	
	/* Testing Matrix Check same */cs
	if(!check_matrix_same(make_mat(2, 3, mata), make_mat(2, 3, mata))){
		error_flag = true;
		printf("!!! Check Matrix same is incorrect !!!\n");
	}

	/* Testing Matrix Transpose */
	matrix_t* c = mat_transpose(matrix_a);
	if(!check_matrix_same(c, make_mat(3, 2, transa))){
		error_flag = true;
		printf("!!! Matrix transpose is incorrect !!!\n");
	}
	
	print_matrix(matrix_a);
	print_matrix(c);
		
	/* Testing Matrix Multiplication */
	matrix_t* d = mat_mul(matrix_a, matrix_b);
	if(!check_matrix_same(d, make_mat(2, 2, multiab))){
		error_flag = true;
		printf("!!! Matrix Multiplication is incorrect !!!\n");
	}
	
	//printf("\nPrinting Matrix Multiplication of A & B\n");
	//print_matrix(d);

	if(error_flag){
		printf("At least one error. Testing failed\n");
	}

	free(matrix_a);
	free(matrix_b);
	free(c);
	free(d);
	return 0;
}