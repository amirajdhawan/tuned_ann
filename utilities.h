#include <stdio.h>
#include <stdbool.h>

#define MATRIX_ACCESS(x, y, dim_first)  y + (dim_first * x)
#define DOUBLE_ERROR 1.0e-5

typedef struct matrix matrix_t;

struct matrix{
	double* mat_data;
	int first_dim;
	int second_dim;
};

matrix_t* mat_mul(matrix_t* matrix_a, matrix_t* matrix_b);

matrix_t* mat_transpose(matrix_t* mat);

void print_matrix(matrix_t* mat);

double activation_func(double value);

bool check_double_eq(double value1, double value2);

bool check_matrix_same(matrix_t* mat_a, matrix_t* mat_b);