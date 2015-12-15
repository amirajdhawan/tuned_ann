#ifndef UTILITIES_H
#define UTILITIES_H

#include <stdio.h>
#include <stdbool.h>

#define MATRIX_ACCESS(x, y, dim_second)  ((dim_second) * (x) + y)
#define DOUBLE_ERROR 1.0e-5
#define NAIVE 1
#define OPTIMAL 2

struct matrix{
	double* mat_data;
	int first_dim;
	int second_dim;
};

typedef struct matrix matrix_t;

matrix_t* make_mat(int first_dim, int second_dim, double* data);

matrix_t* mat_mul(matrix_t* matrix_a, matrix_t* matrix_b, int method);

matrix_t* mat_transpose(matrix_t* mat);

matrix_t* mat_mul_element(matrix_t* A, matrix_t* B);

matrix_t* mat_mul_scalar(matrix_t* M, double alpha);

matrix_t* mat_add(matrix_t* A, matrix_t* B);

matrix_t* mat_subtract(matrix_t* A, matrix_t* B);

void print_matrix(matrix_t* mat);

double activation_func(double value);

double der_activation_func(double value);

matrix_t* activation_func_mat(matrix_t* M);

matrix_t* der_activation_func_mat(matrix_t* M);

bool check_double_eq(double value1, double value2);

bool check_matrix_same(matrix_t* mat_a, matrix_t* mat_b);

void mat_free(matrix_t* a);

#endif