#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl_types.h"
#include "Sparse_Matrix.h"


#ifndef __ITERATIVE_SOLVERS__
#define __ITERATIVE_SOLVERS__

// Generalised Function which calls the Iterative solvers routine 
// Solver Type - 1 - Direct Solver ( UMFPACK )
// Solver Type - 2 - Modified Jacobi Solver 
// Solver Type - 3 - SOR
// Solver Type - 4 - Congugate Gradient 
// Solver Type - 5 - Congugate Gradient Jacobi Preconditioner
void Solver_Iterative(int solver, Sparse_Matrix* matrix, double* b, double* x, double tolerance, double Max_iter);

// Modified Jacobi Solver 
void Jacobi_blas_solver_CSR(Sparse_Matrix* matrix, double* b, double* x, double tolerance, double Max_iter);

// SOR Method
void SOR_CSR(Sparse_Matrix& matrix, double* b, double* x, double tolerance, double Max_iter);

// Conjugate Gradient
void Conjugate_Gradient_CSR(Sparse_Matrix& matrix, double* b, double* x, double tolerance, double Max_iter);

void Conjugate_Gradient_Preconditioned_Jacobi_CSR(Sparse_Matrix& matrix, double* b, double* x, double tolerance, double Max_iter);

void FOR_RESTARTED_CSR(Sparse_Matrix& matrix, double* b, double* x, double tolerance, double Max_iter,Sparse_Matrix* H,int restartFOMFactor, sparse_matrix_t& A1,sparse_matrix_t& H1);

void GMRES_RESTARTED_CSR(Sparse_Matrix& matrix, double* b, double* x, double tolerance, double Max_iter,Sparse_Matrix* H,int restartFOMFactor, sparse_matrix_t& A1,sparse_matrix_t& H1);

double residual_norm(sparse_matrix_t& A1,double* b,double* x,int N_DOF);
void Intel_Sparse_error(std::string);
#endif

