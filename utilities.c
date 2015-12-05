#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#include "utilities.h"

//Matrix Multiplication
matrix_t* mat_mul(matrix_t* matrix_a, matrix_t* matrix_b){

	if(matrix_a->second_dim != matrix_b->first_dim){
		return NULL;
	}

	matrix_t* c = (matrix_t*) malloc(sizeof(matrix_t));

	c->first_dim = matrix_a->first_dim;
	c->second_dim = matrix_b->second_dim;

	c->mat_data = (double*)malloc(c->first_dim * c->second_dim * sizeof(double));
	double sum = 0;

	for(int i = 0; i < matrix_a->first_dim; i++){
		for(int j = 0; j < matrix_b->second_dim; j++){
			for(int k = 0; k < matrix_a->second_dim; k++){
				sum += matrix_a->mat_data[MATRIX_ACCESS(i, k, matrix_a->second_dim)] * matrix_b->mat_data[MATRIX_ACCESS(k, j, matrix_b->second_dim)];
			}

			c->mat_data[MATRIX_ACCESS(i, j, c->second_dim)] = sum;
			sum = 0;
		}
	}
	return c;
}

matrix_t* mat_transpose(matrix_t* mat){
	
	matrix_t* trans_mat = (matrix_t*) malloc(sizeof(matrix_t));
	trans_mat->mat_data = (double*) malloc(sizeof(mat->mat_data));

	trans_mat->first_dim = mat->second_dim;
	trans_mat->second_dim = mat->first_dim;

	for(int i = 0; i < mat->first_dim; i++){
		for(int j = 0; j < mat->second_dim; j++){
			trans_mat->mat_data[MATRIX_ACCESS(j, i, mat->first_dim)] = mat->mat_data[MATRIX_ACCESS(i, j, mat->second_dim)];
		}
	}

	return trans_mat;
}

void print_matrix(matrix_t* mat){

	printf("\n---Printing Matrix---\n");

	for(int i = 0; i < mat->first_dim; i++){
		for(int j = 0; j < mat->second_dim; j++){
			printf(" %.1f", mat->mat_data[MATRIX_ACCESS(i, j, mat->second_dim)]);
		}
		printf("\n");
	}
}

double activation_func(double value){
	//Using Rectified Linear Units (ReLU)
	return value > 0 ? value : 0;
}

bool check_double_eq(double value1, double value2){
	return (abs(value1 - value2) < DOUBLE_ERROR) ? true : false;
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