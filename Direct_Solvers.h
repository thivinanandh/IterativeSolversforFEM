#include "umfpack.h"
#include "Sparse_Matrix.h"
#include "FE_1D.h"

#ifndef __DIRECT_SOLVERS__
#define __DIRECT_SOLVERS__


void Solver_Direct(Sparse_Matrix* matrix,double* FGlobal, double* Solution);

void Solver_Direct_At(Sparse_Matrix* matrix,double* FGlobal, double* Solution);
void Solver_Direct(Sparse_Matrix* matrix,double* FGlobal, double* Solution,int row, int col);

#endif
