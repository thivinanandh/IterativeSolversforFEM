#include "Direct_Solvers.h"
void Solver_Direct(Sparse_Matrix* matrix,double* FGlobal, double* Solution)
{
    std::cout << " Solver Type : DIRECT Solver - UMFPACK " <<std::endl;
    void *Numeric_Factor;
    double *null = (double *) NULL ;
    void *Symbolic;
    std::cout << " Row : "<< matrix->row <<std::endl;
    std::cout << " col : "<< matrix->col <<std::endl;
    std::cout << "  NNZ : "<< matrix->NNZSize <<std::endl;
    auto start = clock();

    
    //l = 0;
    int s1 = umfpack_di_symbolic (matrix->row, matrix->col, &(matrix->rowPtr[0]), &(matrix->colPtr[0]), &(matrix->values[0]), &Symbolic, null, null) ;
    if(s1!=0)
        std::cout<<"error in 1    "<< s1<<std::endl;
    s1 = umfpack_di_numeric (&(matrix->rowPtr[0]), &(matrix->colPtr[0]), &(matrix->values[0]), Symbolic, &Numeric_Factor, null, null) ;
    if(s1!=0)
        std::cout<<"error in 2"<<std::endl;


    s1 = umfpack_di_solve (UMFPACK_A, &(matrix->rowPtr[0]), &(matrix->colPtr[0]), &(matrix->values[0]), Solution,FGlobal, Numeric_Factor, null, null) ;
    if(s1!=0)
        std::cout<<"error in 3"<<std::endl;
    umfpack_di_free_numeric(&Numeric_Factor);

    auto end =  clock();
    auto duration = (end -start)/double(CLOCKS_PER_SEC)*1000; 
    std::cout << " Time Taken to Solve : "<<duration << " ms"<<std::endl;
}

void Solver_Direct_At(Sparse_Matrix* matrix,double* FGlobal, double* Solution)
{
    std::cout << " Solver Type : DIRECT Solver - UMFPACK " <<std::endl;
    void *Numeric_Factor;
    double *null = (double *) NULL ;
    void *Symbolic;

    auto start = clock();
    //l = 0;
    int s1 = umfpack_di_symbolic (matrix->row, matrix->col, matrix->rowPtr.data(), matrix->colPtr.data(), matrix->values.data(), &Symbolic, null, null) ;
    if(s1!=0)
        std::cout<<"error in 1"<<std::endl;

    s1 = umfpack_di_numeric (matrix->rowPtr.data(), matrix->colPtr.data(), matrix->values.data(), Symbolic, &Numeric_Factor, null, null) ;
    if(s1!=0)
        std::cout<<"error in 2"<<std::endl;


    s1 = umfpack_di_solve (UMFPACK_At, matrix->rowPtr.data(), matrix->colPtr.data(), matrix->values.data(), Solution,FGlobal, Numeric_Factor, null, null) ;
    if(s1!=0)
        std::cout<<"error in 3"<<std::endl;
    umfpack_di_free_numeric(&Numeric_Factor);

    auto end =  clock();
    auto duration = (end -start)/double(CLOCKS_PER_SEC)*1000; 
    std::cout << " Time Taken to Solve : "<<duration << " ms"<<std::endl;
}

void Solver_Direct(Sparse_Matrix* matrix,double* FGlobal, double* Solution,int row, int col)
{
    //std::cout << " Solver Type : DIRECT Solver - UMFPACK " <<std::endl;
    void *Numeric_Factor;
    double *null = (double *) NULL ;
    void *Symbolic;

    //l = 0;
    int s1 = umfpack_di_symbolic (row, col, matrix->rowPtr.data(), matrix->colPtr.data(), matrix->values.data(), &Symbolic, null, null) ;
    if(s1!=0)
        std::cout<<"error in 1"<<std::endl;

    s1 = umfpack_di_numeric (matrix->rowPtr.data(), matrix->colPtr.data(), matrix->values.data(), Symbolic, &Numeric_Factor, null, null) ;
    if(s1!=0)
        std::cout<<"error in 2"<<std::endl;


    s1 = umfpack_di_solve (UMFPACK_A, matrix->rowPtr.data(), matrix->colPtr.data(), matrix->values.data(), Solution,FGlobal, Numeric_Factor, null, null) ;
    if(s1!=0)
        std::cout<<"error in 3"<<std::endl;
    umfpack_di_free_numeric(&Numeric_Factor);


}