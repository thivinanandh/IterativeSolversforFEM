#include "InputData.h"
#include "mkl.h"
#include "mkl_spblas.h"
#include "mkl_types.h"
#include "Direct_Solvers.h"
#include "Iterative_Solvers.h"
#include "Sparse_Matrix.h"
#include <ctime>
#include<cstring>

void Intel_Sparse_error(std::string error_string)
{
    if(error_string == "SPARSE_STATUS_SUCCESS")
	{
		std::cout<<"Sparse Matrix Created"<<std::endl;
	}

    if(error_string == "SPARSE_STATUS_NOT_INITIALIZED")
	{
		std::cout<<"SPARSE_STATUS_NOT_INITIALIZED"<<std::endl;
	}

    if(error_string == "SPARSE_STATUS_ALLOC_FAILED")
	{
		std::cout<<"SPARSE_STATUS_ALLOC_FAILED"<<std::endl;
	}

     if(error_string == "SPARSE_STATUS_INVALID_VALUE")
	{
		std::cout<<"SPARSE_STATUS_INVALID_VALUE"<<std::endl;
	}
     if(error_string == "SPARSE_STATUS_EXECUTION_FAILED")
	{
		std::cout<<"SPARSE_STATUS_EXECUTION_FAILED"<<std::endl;
	}
     if(error_string == "SPARSE_STATUS_INTERNAL_ERROR")
	{
		std::cout<<"SPARSE_STATUS_INTERNAL_ERROR"<<std::endl;
	}
     if(error_string == "SPARSE_STATUS_NOT_SUPPORTED")
	{
		std::cout<<"SPARSE_STATUS_NOT_SUPPORTED"<<std::endl;
	}
}

double residual_norm(sparse_matrix_t &A1,double* b,double* x,int N_DOF)
{
	double r1 = 0;

	sparse_status_t sA;		
    std::vector<double> b_false(N_DOF,0);

    memcpy(&(b_false[0]),b,N_DOF*sizeof(double));
	
	matrix_descr des;
	des.type = SPARSE_MATRIX_TYPE_GENERAL;
	sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A1,des,x,-1.0,&(b_false[0]));
	//   if(sA == SPARSE_STATUS_SUCCESS)
	// {
	// 	std::cout<<"Sparse Matrix Created"<<std::endl;
	// }
	r1 = cblas_dnrm2(N_DOF,&(b_false[0]),1);
	//mkl_sparse_destroy(A1);
	return r1;
}



void Jacobi_solver_CSR(Sparse_Matrix& matrix, double* b, double* x, double tolerance, double Max_iter)
{
   std::cout << " Solver : Jacobi " << std::endl;
    int N_DOF = matrix.row;
	double residual;
    bool residual_activate = 0;
	unsigned int iteration = 0;
	std::vector<double> x_old(N_DOF,0);
	double norm =0.;
    double norm1 = 0.;
    double sqrt_norm = sqrt(tolerance);
    double temp =0.;
	// Jacobi Iteration PARAMETER
	double omega = InputData::RELAXATION_JACOBI ;
	std::cout << " Relaxation Parameter :  " <<omega<< std::endl;

	double alpha = -1.0;
    int alpha1 = 1.0;
	// size of array in NDOF
	int array_size = N_DOF*(sizeof(double));

    // Setup Sparse system for Intel MKL Blas
    sparse_status_t sA;
	sparse_index_base_t p1 = SPARSE_INDEX_BASE_ZERO;
	sparse_matrix_t A1;
	int row = matrix.row;
	int col = matrix.col;
	int row_start = 0;
	int row_end = 0;
	int col_no = 0;

	sA = mkl_sparse_d_create_csr(&A1,SPARSE_INDEX_BASE_ZERO,
										 row,col,&(matrix.rowPtr[0]),&(matrix.rowPtr[1]),&(matrix.colPtr[0]),&(matrix.values[0]));
    

	double start = omp_get_wtime(); 
		do {
			iteration++;
			if(norm < sqrt_norm && iteration > 1)
				residual_activate = 1;
			norm1 = 0;
			norm = 0.;

			for( int i = 0 ; i < N_DOF; i++)
			{	
				x[i] = b[i];
				row_start = matrix.rowPtr[i];
				row_end = matrix.rowPtr[i+1];
				for ( int j = row_start ; j < row_end;j++){
					col_no = matrix.colPtr[j];
					if(i !=col_no)
						x[i] -= matrix.values[j]*x_old[col_no];
				}
				x[i] = x_old[i]*( 1 - omega) +  (x[i]/matrix.getValues(i,i))*omega;
				//norm1 += (x_old[i] - x[i] )*(x_old[i] - x[i]);
				//x_old[i] = x[i];
			}
			
			if(residual_activate != 1 ) // use the 2norm difference between the current and previous solution
			{
				daxpy(&N_DOF,&alpha,&(x[0]),&alpha1,&(x_old[0]),&alpha1);
				norm = cblas_dnrm2(N_DOF,&(x_old[0]),1);
				memcpy(&(x_old[0]),&(x[0]),array_size);
				//std::cout << " Iteration : "<<iteration << " 2norm_blas: "<<norm<<std::endl;
			}
			else{
				norm = residual_norm(A1,&(b[0]),&(x[0]),N_DOF);
				//std::cout << " Iteration : "<<iteration << " Resnorm: "<<norm<<std::endl;
			
			}
				memcpy(&(x_old[0]),&(x[0]),array_size);
			if(iteration % InputData::RESIDUAL_DISPLAY == 0)
				std::cout<<" Iteration : " << iteration << " Error Norm : "<< norm<<std::endl;
		
	}

	while (norm > tolerance && iteration < Max_iter);

	double stop = omp_get_wtime();
	double duration = (stop -start); 

	if(iteration == Max_iter)
	{
	std::cout << " Status :  Jacobi Iteration has [[NOT]]converged  ---- " << std::endl;
	std::cout<<"  Total Iterations  :  "<< iteration << std::endl;
	std::cout<<"  Total Time taken  :  "<< duration << " sec" << std::endl;
	std::cout<<"  Error Norm        :  "<< norm << std::endl;
	std::cout<< " ------------------------------------------------------------------" <<std::endl;
	}

	else
	{
	std::cout << " Status :  Jacobi Iteration has converged    " << std::endl;
	std::cout<<"  Total Iterations  :  "<< iteration << std::endl;
	std::cout<<"  Total Time taken  :  "<< duration << " sec" << std::endl;
	std::cout<<"  Error Norm        :  "<< norm << std::endl;
	std::cout<< " ------------------------------------------------------------------" <<std::endl;
	}
	// for( int i = 0 ; i < N_DOF; i++)
	// 	std::cout << x[i] << std::endl;
}

void Jacobi_blas_solver_CSR(Sparse_Matrix& matrix, double* b, double* x, double tolerance, double Max_iter)
{
  	
	mkl_set_num_threads(InputData::NUM_THREADS);
	mkl_set_dynamic(false);
  
   std::cout << " Solver : Jacobi BLAS" << std::endl;
    int N_DOF = matrix.row;
	double residual;
    bool residual_activate = 0;
	unsigned int iteration = 0;
	double norm =0.;
    double norm1 = 0.;
    double sqrt_norm = sqrt(tolerance);
    double temp =0.;
	// Jacobi Iteration PARAMETER
	double omega = InputData::RELAXATION_JACOBI ;
	std::cout << " Relaxation Parameter :  " <<omega<< std::endl;
		
	std::vector<double> diagonal(N_DOF,0);
	std::vector<double> tmparray(N_DOF,0);
	std::vector<double> x_old(N_DOF,0);

	double alpha = -1.0;
    int alpha1 = 1.0;
	// size of array in NDOF
	int array_size = N_DOF*(sizeof(double));

    // Setup Sparse system for Intel MKL Blas
    sparse_status_t sA;
	sparse_index_base_t p1 = SPARSE_INDEX_BASE_ZERO;
	sparse_matrix_t A1;
	int row = matrix.row;
	int col = matrix.col;
	int row_start = 0;
	int row_end = 0;
	int col_no = 0;

	sA = mkl_sparse_d_create_csr(&A1,SPARSE_INDEX_BASE_ZERO,
										 row,col,&(matrix.rowPtr[0]),&(matrix.rowPtr[1]),&(matrix.colPtr[0]),&(matrix.values[0]));
    	
	matrix_descr des;
    des.type = SPARSE_MATRIX_TYPE_GENERAL;	


	// Get the Diagonal Entry of the System 
	#pragma omp parallel for
	for ( int i = 0; i < row ; i++)   
	{
		int start  =  matrix.rowPtr[i];
		int end    = matrix.rowPtr[i+1];
		for (int j = start ; j < end ; j++)
			if( i == matrix.colPtr[j])          // Diagonal Entry
				diagonal[i] = (matrix.values[j]);
	}

	double start = omp_get_wtime(); 
		do {
			iteration++;
			if(norm < sqrt_norm && iteration > 1)
				residual_activate = 1;
			norm1 = 0;
			norm = 0.;
			
			memcpy(tmparray.data(),b,array_size);
			sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,-1.0,A1,des,x_old.data(),1.0,tmparray.data());
			
			#pragma omp parallel for
			for ( int i = 0 ; i < N_DOF ; i++)
				x[i] = ((tmparray[i] +  diagonal[i]*x_old[i])/diagonal[i])*omega + x_old[i]*(1-omega);
				
		
			if(residual_activate != 1 ) // use the 2norm difference between the current and previous solution
			{
				daxpy(&N_DOF,&alpha,&(x[0]),&alpha1,&(x_old[0]),&alpha1);
				norm = cblas_dnrm2(N_DOF,&(x_old[0]),1);
				//std::cout << " Iteration : "<<iteration << " 2norm_blas: "<<norm<<std::endl;
			}
			else{
				norm = residual_norm(A1,&(b[0]),&(x[0]),N_DOF);
				//std::cout << " Iteration : "<<iteration << " Resnorm: "<<norm<<std::endl;
			}
			memcpy(&(x_old[0]),&(x[0]),array_size);
			if(iteration % InputData::RESIDUAL_DISPLAY == 0)
				std::cout<<" Iteration : " << iteration << " Error Norm : "<< norm<<std::endl;
		
	}

	while (norm > tolerance && iteration < Max_iter);

	double stop = omp_get_wtime();
	double duration = (stop -start); 

	if(iteration == Max_iter)
	{
		std::cout << " Status :  Jacobi Iteration has [[NOT]]converged  ---- " << std::endl;
		std::cout<<"  Total Iterations  :  "<< iteration << std::endl;
		std::cout<<"  Total Time taken  :  "<< duration << " sec" << std::endl;
		std::cout<<"  Error Norm        :  "<< norm << std::endl;
		std::cout<< " ------------------------------------------------------------------" <<std::endl;
	}

	else
	{
		std::cout << " Status :  Jacobi Iteration has converged    " << std::endl;
		std::cout<<"  Total Iterations  :  "<< iteration << std::endl;
		std::cout<<"  Total Time taken  :  "<< duration << " sec" << std::endl;
		std::cout<<"  Error Norm        :  "<< norm << std::endl;
		std::cout<< " ------------------------------------------------------------------" <<std::endl;
	}
	
}

void SOR_CSR(Sparse_Matrix& matrix, double* b, double* x, double tolerance, double Max_iter)
{
   	mkl_set_num_threads(InputData::NUM_THREADS);
	mkl_set_dynamic(false);
   std::cout << " Solver : SOR " << std::endl;
    int N_DOF = matrix.row;
	double residual;
    bool residual_activate = 0;
	unsigned int iteration = 0;
	std::vector<double> x_old(N_DOF,0);
	double norm =0.;
    double norm1 = 0.;
    double sqrt_norm = sqrt(tolerance);
    double temp =0.;
	// Jacobi Iteration PARAMETER
	double omega = InputData::RELAXATION_SOR;
	std::cout << " Relaxation Parameter :  " <<omega<< std::endl;

	double alpha = -1.0;
    int alpha1 = 1.0;
	// size of array in NDOF
	int array_size = N_DOF*(sizeof(double));

    // Setup Sparse system for Intel MKL Blas
    sparse_status_t sA;
	sparse_index_base_t p1 = SPARSE_INDEX_BASE_ZERO;
	sparse_matrix_t A1;
	int row = matrix.row;
	int col = matrix.col;
	int row_start = 0;
	int row_end = 0;
	int col_no = 0;

	sA = mkl_sparse_d_create_csr(&A1,SPARSE_INDEX_BASE_ZERO,
										 row,col,&(matrix.rowPtr[0]),&(matrix.rowPtr[1]),&(matrix.colPtr[0]),&(matrix.values[0]));
    

	double start = omp_get_wtime(); 
	do {
		iteration++;
		if(norm < sqrt_norm && iteration > 1)
			residual_activate = 1;
		norm1 = 0;
		norm = 0.;

		for( int i = 0 ; i < N_DOF; i++)
		{	
			x[i] = b[i];
			row_start = matrix.rowPtr[i];
			row_end = matrix.rowPtr[i+1];
			for ( int j = row_start ; j < row_end;j++){
				col_no = matrix.colPtr[j];
				if(i !=col_no)
					x[i] -= matrix.values[j]*x[col_no];
			}
			x[i] = x_old[i]*( 1 - omega) +  (x[i]/matrix.getValues(i,i))*omega;
			//norm1 += (x_old[i] - x[i] )*(x_old[i] - x[i]);
			//x_old[i] = x[i];
		}
		
		if(residual_activate != 1 ) // use the 2norm difference between the current and previous solution
		{
			daxpy(&N_DOF,&alpha,&(x[0]),&alpha1,&(x_old[0]),&alpha1);
			norm = cblas_dnrm2(N_DOF,&(x_old[0]),1);
			memcpy(&(x_old[0]),&(x[0]),array_size);
			//std::cout << " Iteration : "<<iteration << " 2norm_blas: "<<norm<<std::endl;
		}
		else{
			norm = residual_norm(A1,&(b[0]),&(x[0]),N_DOF);
			//std::cout << " Iteration : "<<iteration << " Resnorm: "<<norm<<std::endl;
		}
			memcpy(&(x_old[0]),&(x[0]),array_size);
		if(iteration % InputData::RESIDUAL_DISPLAY == 0)
			std::cout<<" Iteration : " << iteration << " Error Norm : "<< norm<<std::endl;
	
	}

	while (norm > tolerance && iteration < Max_iter);

	double stop = omp_get_wtime();
	double duration = (stop -start); 

	if(iteration == Max_iter)
	{
	std::cout << " Status :  SOR has [[NOT]]converged  ---- " << std::endl;
	std::cout<<"  Total Iterations  :  "<< iteration << std::endl;
	std::cout<<"  Total Time taken  :  "<< duration << " sec" << std::endl;
	std::cout<<"  Error Norm        :  "<< norm << std::endl;
	std::cout<< " ------------------------------------------------------------------" <<std::endl;
	}

	else
	{
	std::cout << " Status :  SOR  has converged    " << std::endl;
	std::cout<<"  Total Iterations  :  "<< iteration << std::endl;
	std::cout<<"  Total Time taken  :  "<< duration << " sec" << std::endl;
	std::cout<<"  Error Norm        :  "<< norm << std::endl;
	std::cout<< " ------------------------------------------------------------------" <<std::endl;
	}

	// for( int i = 0 ; i < N_DOF; i++)
	// 	std::cout << x[i] << std::endl;
	
}

void Conjugate_Gradient_CSR(Sparse_Matrix& matrix, double* b, double* x, double tolerance, double Max_iter)
{
	mkl_set_num_threads(InputData::NUM_THREADS);
	mkl_set_dynamic(false);
	// Setup Sparse system for Intel MKL Blas
    sparse_status_t sA;
	sparse_index_base_t p1 = SPARSE_INDEX_BASE_ZERO;
	sparse_matrix_t A1;
	int row = matrix.row;
	int col = matrix.col;
	int row_start = 0;
	int row_end = 0;
	int col_no = 0;
	int array_size = row*sizeof(double);

	//Values for Daxpy Solver
	double Alp_minus1 = -1;
	double Alp_plus1  = 1;
	int Alp_incrmnt = 1;

	// Initialise Values for Conjugate Gradient
	double alpha = 0;
	double beta = 0;
	double temp_variable = 0.;
	std::vector<double> p(row,0);
	std::vector<double> residual(row,0);
	std::vector<double> residual_old(row,0);
	std::vector<double> temp(row,0);
	unsigned int iteration = 0;

	std::cout << "-- Solver : Conjugate Gradient --- " << std::endl;
	double start = omp_get_wtime(); 

	sA = mkl_sparse_d_create_csr(&A1,SPARSE_INDEX_BASE_ZERO,
										 row,col,&(matrix.rowPtr[0]),&(matrix.rowPtr[1]),&(matrix.colPtr[0]),&(matrix.values[0]));
    
	matrix_descr des;
	des.type = SPARSE_MATRIX_TYPE_GENERAL;

	
	// Calculate Residual
	
	sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A1,des,x,0.0,residual.data());  // Ax0
	daxpby(&row,&Alp_plus1, &(b[0]),&Alp_incrmnt,&Alp_minus1,residual.data(),&Alp_incrmnt); // ro = b - Ax0
	double norm =  cblas_dnrm2(row,residual.data(),Alp_incrmnt);
	
	// If Converged Exit the Loop
	if( norm < tolerance)
	{
		auto stop = omp_get_wtime();
		auto duration = (stop -start); 
		std::cout << " Status :  --CONJUGATE GRADIENT HAS CONVERGED <SUCCESSFULLY>  " <<std::endl;
		std::cout<<"  Total Iterations  :  "<< iteration << std::endl;
		std::cout<<"  Total Time taken  :  "<< duration << " sec" << std::endl;
		std::cout<<"  Error Norm        :  "<< norm << std::endl;
		std::cout<< " ------------------------------------------------------------------" <<std::endl;
	}

	memcpy(p.data(),residual.data(),array_size);  // po = ro

	norm = 1000;   // Loop entry Criteria
	while ( norm > tolerance && iteration < InputData::MAX_ITER)
	{
		memcpy(temp.data(),p.data(),array_size);
		
		// ------------------ Alpha = < r,r > / <p,A,p> ----------------------- //
		sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A1,des,p.data(),0.0,temp.data()); // <A,p>
		alpha = 1.0/cblas_ddot(row,p.data(),1.0,temp.data(),1.0);  								// <Pt A x >
		temp_variable =  cblas_dnrm2(row,residual.data(),1.0) ;
		alpha *= (temp_variable*temp_variable);                   
		
		// X(k+1) = x(k) + alpha*p(k)
		cblas_daxpy(row,alpha,p.data(),1.0,x,1.0);

		// R(k+1) = r(k) - alpha*p(k)
		cblas_daxpy(row,-1.0*(alpha),temp.data(),Alp_incrmnt,residual.data(),Alp_incrmnt);

		norm = cblas_dnrm2(row,residual.data(),1.0);
		if(iteration % InputData::RESIDUAL_DISPLAY == 0)  std::cout << " Iteration : "<<iteration << " Norm : " << norm<<std::endl;
			
		if( norm < tolerance)
		{
			auto stop = omp_get_wtime();
			auto duration = (stop -start); 
			std::cout << " Status :  CONJUGATE GRADIENT HAS CONVERGED <SUCCESSFULLY>  " <<std::endl;
			std::cout<<"  Total Iterations  :  "<< iteration << std::endl;
			std::cout<<"  Total Time taken  :  "<< duration << " sec" << std::endl;
			std::cout<<"  Error Norm        :  "<< norm << std::endl;
			std::cout<< " ------------------------------------------------------------------" <<std::endl;
			break;
		}

		beta = (norm*norm)/(temp_variable*temp_variable);    // Beta = <R(k+1),R(k+1)/<R(k),R(k)>

		cblas_daxpby(row,1.0,residual.data(),1.0,beta,p.data(),1.0);
		iteration++;

	}
	if (InputData::PRINT_RESIDUAL)
		if( iteration >= Max_iter )
		{
			double stop = omp_get_wtime();
			auto duration = (stop -start); 
			std::cout << " Status :  CONJUGATE GRADIENT HAS [[NOT]] CONVERGED <FAILED>  " <<std::endl;
			std::cout<<"  Total Iterations  :  "<< iteration << std::endl;
			std::cout<<"  Total Time taken  :  "<< duration << " sec" << std::endl;
			std::cout<<"  Error Norm        :  "<< norm << std::endl;
			std::cout<< " ------------------------------------------------------------------" <<std::endl;
		}
}

void Conjugate_Gradient_Preconditioned_Jacobi_CSR(Sparse_Matrix& matrix, double* b, double* x, double tolerance, double Max_iter)
{
	mkl_set_num_threads(InputData::NUM_THREADS);
	mkl_set_dynamic(false);
	
	// Setup Sparse system for Intel MKL Blas
    sparse_status_t sA;
	sparse_index_base_t p1 = SPARSE_INDEX_BASE_ZERO;
	sparse_matrix_t A1;
	int row = matrix.row;
	int col = matrix.col;
	int row_start = 0;
	int row_end = 0;
	int col_no = 0;
	int array_size = row*sizeof(double);

	//Values for Daxpy Solver
	double Alp_minus1 = -1;
	double Alp_plus1  = 1;
	int Alp_incrmnt = 1;

	// Initialise Values for Conjugate Gradient
	double alpha = 0;
	double beta = 0;
	double norm1 = 0;
	double norm = 0;
	double rho = 0;
	double rho_0;
	double temp_variable = 0.;
	std::vector<double> p(row,0);
	std::vector<double> residual(row,0);
	std::vector<double> residual_old(row,0);
	std::vector<double> temp(row,0);
	std::vector<double> Z(row,0);
	unsigned int iteration = 0;

	// Jacobi Preconditioner Matrix
	std::vector<double> Jacobi_Preconditioner(row,0);
	
	
	// Get the Diagonal Entry of the System 
	#pragma omp parallel for
	for ( int i = 0; i < row ; i++)   
	{
		int start  =  matrix.rowPtr[i];
		int end    = matrix.rowPtr[i+1];
		for (int j = start ; j < end ; j++)
			if( i == matrix.colPtr[j])          // Diagonal Entry
				Jacobi_Preconditioner[i] = 1./(matrix.values[j]);
	}
	std::cout << "-- Solver : Conjugate Gradient --- " << std::endl;

	double start = omp_get_wtime(); 
	sA = mkl_sparse_d_create_csr(&A1,SPARSE_INDEX_BASE_ZERO,
										 row,col,&(matrix.rowPtr[0]),&(matrix.rowPtr[1]),&(matrix.colPtr[0]),&(matrix.values[0]));
    
	matrix_descr des;
	des.type = SPARSE_MATRIX_TYPE_GENERAL;
	
	// -------------------- PRECONDITIONED CONJUGATE GRADIENT  -----------------------------------//
	// Calculate Residual
	sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A1,des,x,0.0,residual.data());  // Ax0
	daxpby(&row,&Alp_plus1, &(b[0]),&Alp_incrmnt,&Alp_minus1,residual.data(),&Alp_incrmnt); // ro = b - Ax0
	norm  =  cblas_dnrm2(row,residual.data(),Alp_incrmnt);

	// Zo = C^(-1)ro
	#pragma omp parallel for
	for ( int i = 0 ; i< row ; i++)
		Z[i] = Jacobi_Preconditioner[i] * residual[i] ;	

	rho = cblas_ddot(row,residual.data(),1.0,Z.data(),1.0);

	// If Converged Exit the Loop
	if( norm < tolerance)
	{
		double stop = omp_get_wtime();
		double duration = (stop -start); 
		std::cout << " Status :  -- CONJUGATE GRADIENT HAS CONVERGED <SUCCESSFULLY>  " <<std::endl;
		std::cout<<"  Total Iterations  :  "<< iteration << std::endl;
		std::cout<<"  Total Time taken  :  "<< duration << " sec" << std::endl;
		std::cout<<"  Error Norm        :  "<< norm << std::endl;
		std::cout<< " ------------------------------------------------------------------" <<std::endl;
	}

	memcpy(p.data(),Z.data(),array_size);                   // Zo = po

	norm = 1000.;   // Loop entry Criteria
	while ( norm > tolerance && iteration < Max_iter)
	{
		memcpy(temp.data(),p.data(),array_size);                   // Create a Copy of p0 to temp
		// ------------------ Alpha = < r,r > / <p,A,p> ----------------------- //
		sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A1,des,p.data(),0.0,temp.data());   // <A,p>
		alpha = 1.0/cblas_ddot(row,p.data(),1.0,temp.data(),1.0);  								   // <Pt A P>                          // <Zt rk>
		alpha = rho*(alpha);                                                                       // ALPHA = rhO/
	
		// X(k+1) = x(k) + alpha*p(k)
		cblas_daxpy(row,alpha,p.data(),1.0,x,1.0);
		
		// R(k+1) = r(k) - alpha*p(k)*A
		cblas_daxpy(row,-1.0*(alpha),temp.data(),Alp_incrmnt,residual.data(),Alp_incrmnt);
		
		rho_0 = rho;

		// Z(k+1) = C^(-1)R(k+1)
		#pragma omp parallel for
		for ( int i = 0 ; i< row ; i++)  Z[i] = Jacobi_Preconditioner[i] * residual[i] ;
				
		rho = cblas_ddot(row,Z.data(),1.0,residual.data(),1.0);   
		
		beta = (rho)/(rho_0);    // Beta = <R(k+1),R(k+1)/<R(k),R(k)>
		
		cblas_daxpby(row,1.0,Z.data(),1.0,beta,p.data(),1.0);

		norm = cblas_dnrm2(row,residual.data(),1.0);

		if(iteration % InputData::RESIDUAL_DISPLAY == 0)
			std::cout << " Iteration : "<<iteration << " Norm : " << norm<<std::endl;
		
		iteration++;

	}
			
		
	if( norm < tolerance)
		{
			auto stop = omp_get_wtime();
			auto duration = (stop -start); 
			std::cout << " Status :  JACOBI CONJUGATE GRADIENT HAS CONVERGED <SUCCESSFULLY>  " <<std::endl;
			std::cout<<"  Total Iterations  :  "<< iteration << std::endl;
			std::cout<<"  Total Time taken  :  "<< duration << " sec" << std::endl;
			std::cout<<"  Error Norm        :  "<< norm << std::endl;
			std::cout<< " ------------------------------------------------------------------" <<std::endl;
		}
		if( iteration >= Max_iter )
		{
			auto stop = omp_get_wtime();
			auto duration = (stop -start); 
			std::cout << " Status :  CONJUGATE GRADIENT HAS [[NOT]] CONVERGED <FAILED>  " <<std::endl;
			std::cout<<"  Total Iterations  :  "<< iteration << std::endl;
			std::cout<<"  Total Time taken  :  "<< duration << " sec" << std::endl;
			std::cout<<"  Error Norm        :  "<< norm << std::endl;
			std::cout<< " ------------------------------------------------------------------" <<std::endl;
		}
}



void FOR_RESTARTED_CSR(Sparse_Matrix& matrix, double* b, double* x, double tolerance, double Max_iter,Sparse_Matrix* H, int restartFOMFactor,sparse_matrix_t& A1, sparse_matrix_t& H1 )
{
	mkl_set_num_threads(InputData::NUM_THREADS);
	mkl_set_dynamic(false);
	// Setup Sparse system for Intel MKL Blas
	sparse_status_t sA;
	sparse_index_base_t p1 = SPARSE_INDEX_BASE_ZERO;
	int row = matrix.row;
	int col = matrix.col;
	int array_size = row*sizeof(double);
    double h_val;


	std::vector<double> residual(row,0);
	std::vector<std::vector<double> > Q(restartFOMFactor +1, std::vector<double>(row,0));
	std::vector<double> v(row,0);
	std::vector<double> eo(restartFOMFactor,0);
	std::vector<double> y(restartFOMFactor,0);
	
    matrix_descr des;
    des.type = SPARSE_MATRIX_TYPE_GENERAL;
 

    // calculate Residual ( Ax )
    sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A1,des,x,0.0,residual.data()); // Ax and save in residual
    cblas_daxpby(row,1.0,b,1.0,-1.0,residual.data(),1.0);                                 // Calculate b - Ax and save in Residual
    double residual_norm = cblas_dnrm2(row,residual.data(),1.0);                          // Calculate residual norm
        
    cblas_daxpby(row,(1.0/residual_norm),residual.data(),1.0,0.0,Q[0].data(),1.0);                // Q(:,1)=r/r0norm;

    for ( int k = 0 ;  k < restartFOMFactor ; k++)
    {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A1,des,Q[k].data(),0.0,v.data());
        #pragma omp parallel for
        for ( int j = 0 ; j <= k ; j++ )                                   	// H(k+1,k)=norm(v); Q(:,k+1)=v/H(k+1,k);
        {
            h_val = cblas_ddot(row,Q[j].data(),1.0,v.data(),1.0);
            H->getValues(j,k) = h_val;
            cblas_daxpby(row,-1.0*h_val,Q[j].data(),1.0,1.0,v.data(),1.0);
        }
        double norm2 = cblas_dnrm2(row,v.data(),1.0);

        if( k < restartFOMFactor && norm2 > 1e-8)
        {
            h_val =  cblas_dnrm2(row,v.data(),1.0);
            H->getValues(k+1,k) = h_val;
            cblas_daxpby(row,1/h_val,v.data(),1.0,0.0,Q[k+1].data(),1.0); 		
        }
    }
    // ------- - END - RESIZING THE H MATRIX ------------- DELETE THE LAST ROW   //
    // Make the first row as the initial residual;
    eo[0] = residual_norm;
    int row_mod = H->row-1;
    // Solve y = H\eo
    Solver_Direct(H,eo.data(), y.data(),row_mod,H->col) ;     //y=H\e0
    // Solve X = Xo + Q*Y
    #pragma omp parallel for
    for ( int i = 0 ; i < restartFOMFactor ; i++)
        cblas_daxpy(row,y[i],Q[i].data(),1.0,x,1.0);	

}

void GMRES_RESTARTED_CSR(Sparse_Matrix& matrix, double* b, double* x, double tolerance, double Max_iter,Sparse_Matrix* H,
                        int restartFOMFactor, sparse_matrix_t& A1,sparse_matrix_t& H1 )
{
	mkl_set_num_threads(InputData::NUM_THREADS);
	mkl_set_dynamic(false);
	// Setup Sparse system for Intel MKL Blas
	sparse_status_t sA;
    sparse_matrix_t H2;
    sparse_matrix_t HtH_m;
	sparse_index_base_t p1 = SPARSE_INDEX_BASE_ZERO;
	int row = matrix.row;
	int col = matrix.col;
	int array_size = row*sizeof(double);
    double h_val = 0;
	// ---------------------- SETUP H MATRIX ----------------------------------------
    sA = mkl_sparse_d_create_csr(&H2,SPARSE_INDEX_BASE_ZERO,H->row,H->col,&(H->rowPtr[0]),&(H->rowPtr[0])+1,&(H->colPtr[0]),&(H->values[0]));
	
	matrix_descr des;
	des.type = SPARSE_MATRIX_TYPE_GENERAL;
	// end of filling Sparsity Pattern for H MATRIX

        
    std::vector<double> residual(row,0);
	std::vector<std::vector<double> > Q(restartFOMFactor +1, std::vector<double>(row,0));
	std::vector<double> v(row,0);
	std::vector<double> eo(restartFOMFactor+1,0);  // This will be addditional size for Restarted GMRES
	std::vector<double> y(restartFOMFactor,0);

    // calculate Residual ( Ax )
    sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A1,des,x,0.0,residual.data()); // Ax and save in residual
    cblas_daxpby(row,1.0,b,1.0,-1.0,residual.data(),1.0);                                 // Calculate b - Ax and save in Residual
    double residual_norm = cblas_dnrm2(row,residual.data(),1.0);                          // Calculate residual norm
        
    cblas_daxpby(row,(1.0/residual_norm),residual.data(),1.0,0.0,Q[0].data(),1.0);                // Q(:,1)=r/r0norm;

    for ( int k = 0 ;  k < restartFOMFactor ; k++)
    {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A1,des,Q[k].data(),0.0,v.data());
        #pragma omp parallel for
        for ( int j = 0 ; j <= k ; j++ )                                   	// H(k+1,k)=norm(v); Q(:,k+1)=v/H(k+1,k);
        {
            h_val = cblas_ddot(row,Q[j].data(),1.0,v.data(),1.0);
            H->getValues(j,k) = h_val;
            cblas_daxpby(row,-1.0*h_val,Q[j].data(),1.0,1.0,v.data(),1.0);
        }
        double norm2 = cblas_dnrm2(row,v.data(),1.0);

        if( k < restartFOMFactor && norm2 > 1e-8)
        {
            h_val =  cblas_dnrm2(row,v.data(),1.0);
            H->getValues(k+1,k) = h_val;
            cblas_daxpby(row,1/h_val,v.data(),1.0,0.0,Q[k+1].data(),1.0); 		
        }
    }

    // ------- - END - RESIZING THE H MATRIX ------------- DELETE THE LAST ROW   //

    // Make the first row as the initial residual;
    eo[0] = residual_norm;
    
    Sparse_Matrix* Hee = new Sparse_Matrix;
    sA = mkl_sparse_spmm(SPARSE_OPERATION_TRANSPOSE,H1,H2,&HtH_m);

    mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE,1.0,H1,des,eo.data(),0.0,residual.data());
        
    //int* HTH_rowptr; int* HTH_colindex; double *HTH_val;int HTHnrow;int HTHncol;
    sA = mkl_sparse_d_export_csr(HtH_m,&p1,&(Hee->row),&(Hee->col),&(Hee->rowPtr_c),&(Hee->rowPtr_c)+1,&(Hee->colPtr_c),&(Hee->values_c));

    // Solve y = H\eo   //y=H\e0
    void *Numeric_Factor;
    double *null = (double *) NULL ;
    void *Symbolic;
        //std::cout<<" resstart : "<<restartFOMFactor <<"  "<< HTHnrow<<std::endl;
    //l = 0;
    int s1 = umfpack_di_symbolic (Hee->row, Hee->col, Hee->rowPtr_c, Hee->colPtr_c, Hee->values_c, &Symbolic, null, null) ;
        if(s1!=0)  std::cout<<"error in 1"<<std::endl;	
    s1 = umfpack_di_numeric ( Hee->rowPtr_c, Hee->colPtr_c, Hee->values_c, Symbolic, &Numeric_Factor, null, null) ;
        if(s1!=0) std::cout<<"error in 2"<<std::endl;
    s1 = umfpack_di_solve (UMFPACK_A, Hee->rowPtr_c, Hee->colPtr_c, Hee->values_c, y.data(),residual.data(), Numeric_Factor, null, null) ;
        if(s1!=0) std::cout<<"error in 3"<<std::endl;


    // Solve X = Xo + Q*Y
    #pragma omp parallel for
    for ( int i = 0 ; i < restartFOMFactor ; i++)
        cblas_daxpy(row,y[i],Q[i].data(),1.0,x,1.0);	

    //Check residual


    umfpack_di_free_numeric(&Numeric_Factor);
    umfpack_di_free_symbolic(&Symbolic);
    
    
    mkl_sparse_destroy(H2);
    mkl_sparse_destroy(HtH_m);  


}


void GMRES_CSR_old_backup(Sparse_Matrix& matrix, double* b, double* x, double tolerance, double Max_iter)
{
	mkl_set_num_threads(InputData::NUM_THREADS);
	mkl_set_dynamic(false);
	// Setup Sparse system for Intel MKL Blas
	sparse_status_t sA;
	sparse_index_base_t p1 = SPARSE_INDEX_BASE_ZERO;
	sparse_matrix_t A1;
	sparse_matrix_t H1;
	sparse_matrix_t H2;
	sparse_matrix_t HtH_m;
	int row = matrix.row;
	int col = matrix.col;
	int array_size = row*sizeof(double);

	
	// ---------------------- SETUP H MATRIX ----------------------------------------
	Sparse_Matrix* H = new Sparse_Matrix;
	//Sparse_Matrix* H1 = new Sparse_Matrix;
	unsigned long int restartFOMFactor;
	if(InputData::RESTART_PARAMETER_FOM > matrix.row || InputData::RESTART_PARAMETER_FOM == 0 )
	{
		std::cout <<" THE RESTART PARAMETER OF FOM CANNOT BE GREATER THAN THE MATRIX DIMENSION " <<std::endl;
		std::cout << " THE RESTART VALUE IS SET TO THE MATRIX DIMENSION - "<< matrix.row/2 << std::endl;
		restartFOMFactor = matrix.row/2;
	}
	else
	{
		restartFOMFactor = InputData::RESTART_PARAMETER_FOM;
	}
	
	// Setup Matrix for H //
	H->row = restartFOMFactor + 1 ; 
	H->col =  restartFOMFactor;
	long unsigned int n = (restartFOMFactor+1)*(restartFOMFactor+2);
	n = n/2 + 1 ;
	H->NNZSize = n;
	H->rowPtr.resize(H->row+1);
	H->colPtr.resize(H->NNZSize);
	H->values.resize(H->NNZSize);
	H->rowPtr[0] = 0;
	int colptr_val = 0;
	int rowptr_val = 0;
	double h_val;
	// End -- Setup Matrix for H //

	// Fill Row Pointer for the H MATRIX
	for( int  i = 0 ; i < restartFOMFactor + 1 ; i++)
	{
		if( i < 1)
			for (int j = 0 ; j < restartFOMFactor ; j++ )
			{
				H->colPtr[colptr_val] = j;
				colptr_val ++;
				rowptr_val++;
			}
		else
		{
			for (int j = i-1 ; j < restartFOMFactor ; j++ )
			{
				H->colPtr[colptr_val] = j;
				colptr_val ++;
				rowptr_val++;
			}
			
		}
		
		H->rowPtr[i+1] = H->rowPtr[i] + rowptr_val;
		rowptr_val = 0;
	}

	//Create Instances of MKL Sparse Matrix routines
	// sA = mkl_sparse_d_create_csr(&A1,SPARSE_INDEX_BASE_ZERO,
	// 									row,col,&(matrix.rowPtr[0]),&(matrix.rowPtr[1]),&(matrix.colPtr[0]),&(matrix.values[0]));

	
	matrix_descr des;
	des.type = SPARSE_MATRIX_TYPE_GENERAL;
	// end of filling Sparsity Pattern for H MATRIX
	int iteration = 0;
	double residual_norm1 = 1000;
	Sparse_Matrix *Hee = new Sparse_Matrix;
	//while ( iteration < InputData::MAX_ITERATION_FOM && residual_norm1 > InputData::TOLERANCE )
	//{
		//Create Instances of MKL Sparse Matrix routines
		sA = mkl_sparse_d_create_csr(&A1,SPARSE_INDEX_BASE_ZERO,
											row,col,&(matrix.rowPtr[0]),&(matrix.rowPtr[1]),&(matrix.colPtr[0]),&(matrix.values[0]));
		
		sA = mkl_sparse_d_create_csr(&H1,SPARSE_INDEX_BASE_ZERO,
											H->row,H->col,&(H->rowPtr[0]),&(H->rowPtr[1]),&(H->colPtr[0]),&(H->values[0]));
		
		des.type = SPARSE_MATRIX_TYPE_GENERAL;

        
    std::vector<double> residual(row,0);
	std::vector<std::vector<double> > Q(restartFOMFactor +1, std::vector<double>(row,0));
	std::vector<double> v(row,0);
	std::vector<double> eo(restartFOMFactor+1,0);  // This will be addditional size for Restarted GMRES
	std::vector<double> y(restartFOMFactor,0);

    // calculate Residual ( Ax )
    sA = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A1,des,x,0.0,residual.data()); // Ax and save in residual
    cblas_daxpby(row,1.0,b,1.0,-1.0,residual.data(),1.0);                                 // Calculate b - Ax and save in Residual
    double residual_norm = cblas_dnrm2(row,residual.data(),1.0);                          // Calculate residual norm
        
    cblas_daxpby(row,(1.0/residual_norm),residual.data(),1.0,0.0,Q[0].data(),1.0);                // Q(:,1)=r/r0norm;

    for ( int k = 0 ;  k < restartFOMFactor ; k++)
    {
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A1,des,Q[k].data(),0.0,v.data());
        #pragma omp parallel for
        for ( int j = 0 ; j <= k ; j++ )                                   	// H(k+1,k)=norm(v); Q(:,k+1)=v/H(k+1,k);
        {
            h_val = cblas_ddot(row,Q[j].data(),1.0,v.data(),1.0);
            H->getValues(j,k) = h_val;
            cblas_daxpby(row,-1.0*h_val,Q[j].data(),1.0,1.0,v.data(),1.0);
        }
        double norm2 = cblas_dnrm2(row,v.data(),1.0);

        if( k < restartFOMFactor && norm2 > 1e-8)
        {
            h_val =  cblas_dnrm2(row,v.data(),1.0);
            H->getValues(k+1,k) = h_val;
            cblas_daxpby(row,1/h_val,v.data(),1.0,0.0,Q[k+1].data(),1.0); 		
        }
    }

    // ------- - END - RESIZING THE H MATRIX ------------- DELETE THE LAST ROW   //

    // Make the first row as the initial residual;
    eo[0] = residual_norm;
    

    //sparse_matrix_t 
    sA = mkl_sparse_d_create_csr(&H1,SPARSE_INDEX_BASE_ZERO,H->row,H->col,&(H->rowPtr[0]),&(H->rowPtr[0])+1,&(H->colPtr[0]),&(H->values[0]));
    sA = mkl_sparse_d_create_csr(&H2,SPARSE_INDEX_BASE_ZERO,H->row,H->col,&(H->rowPtr[0]),&(H->rowPtr[0])+1,&(H->colPtr[0]),&(H->values[0]));


    //H1->Display_matrix();

    sA = mkl_sparse_spmm(SPARSE_OPERATION_TRANSPOSE,H1,H2,&HtH_m);

    mkl_sparse_d_mv(SPARSE_OPERATION_TRANSPOSE,1.0,H1,des,eo.data(),0.0,residual.data());
        
    //int* HTH_rowptr; int* HTH_colindex; double *HTH_val;int HTHnrow;int HTHncol;
    sA = mkl_sparse_d_export_csr(HtH_m,&p1,&(Hee->row),&(Hee->col),&(Hee->rowPtr_c),&(Hee->rowPtr_c)+1,&(Hee->colPtr_c),&(Hee->values_c));

    // Solve y = H\eo   //y=H\e0
    void *Numeric_Factor;
    double *null = (double *) NULL ;
    void *Symbolic;
        //std::cout<<" resstart : "<<restartFOMFactor <<"  "<< HTHnrow<<std::endl;
    //l = 0;
    int s1 = umfpack_di_symbolic (Hee->row, Hee->col, Hee->rowPtr_c, Hee->colPtr_c, Hee->values_c, &Symbolic, null, null) ;
        if(s1!=0)  std::cout<<"error in 1"<<std::endl;	
    s1 = umfpack_di_numeric ( Hee->rowPtr_c, Hee->colPtr_c, Hee->values_c, Symbolic, &Numeric_Factor, null, null) ;
        if(s1!=0) std::cout<<"error in 2"<<std::endl;
    s1 = umfpack_di_solve (UMFPACK_A, Hee->rowPtr_c, Hee->colPtr_c, Hee->values_c, y.data(),residual.data(), Numeric_Factor, null, null) ;
        if(s1!=0) std::cout<<"error in 3"<<std::endl;


    // Solve X = Xo + Q*Y
    #pragma omp parallel for
    for ( int i = 0 ; i < restartFOMFactor ; i++)
        cblas_daxpy(row,y[i],Q[i].data(),1.0,x,1.0);	

    //Check residual


    umfpack_di_free_numeric(&Numeric_Factor);
    umfpack_di_free_symbolic(&Symbolic);
    //mkl_sparse_destroy(H1);
    //mkl_sparse_destroy(H2);
    // mkl_sparse_destroy(HtH_m);   */
//}
mkl_sparse_destroy(H1);
	//mkl_sparse_destroy(A1);
	

	
	
}




void Solver_Iterative(int solver, Sparse_Matrix* matrix, double* b, double* x, double tolerance, double Max_iter)
{
    if( solver == 2)
        Jacobi_blas_solver_CSR(*matrix, b, x, tolerance, Max_iter);
    else if( solver == 3)
        SOR_CSR(*matrix, b, x, tolerance, Max_iter);
	else if( solver == 4){
		if (InputData::SYMM_BOUND_COND)
	    	Conjugate_Gradient_CSR(*matrix, b, x, tolerance, Max_iter);
		else
		{ 
			std::cout << " Need Symmetric Matrix for CG " <<std::endl; 
			exit(0);
		}
	}
	else if( solver == 5){
		if (InputData::SYMM_BOUND_COND)
	    	Conjugate_Gradient_Preconditioned_Jacobi_CSR(*matrix, b, x, tolerance, Max_iter);
		else
		{ 
			std::cout << " Need Symmetric Matrix for CG " <<std::endl; 
			exit(0);
		}
	}

	else if( solver == 6){
		std::cout << " --------------- RESTARTED FOR SOLVER ---------------------- " <<std::endl;
		int restartFOMFactor;
		if(InputData::RESTART_PARAMETER_FOM > matrix->row || InputData::RESTART_PARAMETER_FOM == 0 )
		{
			std::cout <<" THE RESTART PARAMETER OF FOM CANNOT BE GREATER THAN THE MATRIX DIMENSION " <<std::endl;
			std::cout << " THE RESTART VALUE IS SET TO THE MATRIX DIMENSION - "<< matrix->row/2 << std::endl;
			restartFOMFactor = matrix->row/2;
		}
			else
		{
			restartFOMFactor = InputData::RESTART_PARAMETER_FOM;
		}
		std::cout << " RESTARTED FOM PARAMETER : " << restartFOMFactor << std::endl;
        auto start = omp_get_wtime();
        int iteration = 0;
        double res_norm = 1000;
        sparse_status_t sA;
        sparse_index_base_t p1 = SPARSE_INDEX_BASE_ZERO;
        sparse_matrix_t A1;
        sparse_matrix_t H1;
        std::vector<double> residual(matrix->row,0);
        sA = mkl_sparse_d_create_csr(&A1,SPARSE_INDEX_BASE_ZERO,
                                    matrix->row,matrix->col,&(matrix->rowPtr[0]),&(matrix->rowPtr[1]),&(matrix->colPtr[0]),&(matrix->values[0]));
        matrix_descr des;
        des.type = SPARSE_MATRIX_TYPE_GENERAL;
        
        // Add the Parameters of H Matrix here 
        
        Sparse_Matrix* H = new Sparse_Matrix;
        H->row = restartFOMFactor + 1 ;
        H->col =  restartFOMFactor;
        long unsigned int n = (restartFOMFactor+1)*(restartFOMFactor+2);
        n = n/2 + 1 ;
        H->NNZSize = n;
        //std::cout<< " H - NNZ Size : "<< H->NNZSize <<std::endl;
        H->rowPtr.resize(H->row+1);
        H->colPtr.resize(H->NNZSize);
        H->values.resize(H->NNZSize);
        H->rowPtr[0] = 0;
        int colptr_val = 0;
        int rowptr_val = 0;
        double h_val;	


        for( int  i = 0 ; i < restartFOMFactor + 1 ; i++)
        {
            if( i < 1)
                for (int j = 0 ; j < restartFOMFactor ; j++ )
                {
                    H->colPtr[colptr_val] = j;
                    colptr_val ++;
                    rowptr_val++;
                }
            else
            {
                for (int j = i-1 ; j < restartFOMFactor ; j++ )
                {
                    H->colPtr[colptr_val] = j;
                    colptr_val ++;
                    rowptr_val++;
                }
                
            }
            
        H->rowPtr[i+1] = H->rowPtr[i] + rowptr_val;
        rowptr_val = 0;
        }
        
        // END of Parameters fo H Matrix //
        
        sA = mkl_sparse_d_create_csr(&A1,SPARSE_INDEX_BASE_ZERO,
                                    matrix->row,matrix->col,&(matrix->rowPtr[0]),&(matrix->rowPtr[1]),&(matrix->colPtr[0]),&(matrix->values[0]));
    
        sA = mkl_sparse_d_create_csr(&H1,SPARSE_INDEX_BASE_ZERO,
                                    H->row,H->col,&(H->rowPtr[0]),&(H->rowPtr[1]),&(H->colPtr[0]),&(H->values[0]));
        
        while ( iteration < InputData::MAX_ITER && res_norm > InputData::TOLERANCE){
            FOR_RESTARTED_CSR(*matrix, b, x, tolerance, Max_iter,H,restartFOMFactor,A1,H1);
        
            mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A1,des,x,0.0,residual.data()); // Ax and save in residual
            cblas_daxpby(matrix->row,1.0,b,1.0,-1.0,residual.data(),1.0);                                 // Calculate b - Ax and save in Residual
            res_norm = cblas_dnrm2(matrix->row,residual.data(),1.0);  
            
            if(iteration % InputData::RESIDUAL_DISPLAY == 0)
            std::cout<<" Iteration : " << iteration << " Error Norm : "<< res_norm<<std::endl;
            iteration++;
        }
        auto stop = omp_get_wtime();
        auto duration =  stop - start;
        if(iteration == Max_iter)
        {
            std::cout << " Status :  RESTARTED FOM  Iteration has [[NOT]]converged  ---- " << std::endl;
            std::cout<<"  Total Iterations  :  "<< iteration << std::endl;
            std::cout<<"  Total Time taken  :  "<< duration << " sec" << std::endl;
            std::cout<<"  Error Norm        :  "<< res_norm << std::endl;
            std::cout<< " ------------------------------------------------------------------" <<std::endl;
        }

        else
        {
            std::cout << " Status :  RESTARTED FOM has Iteration has converged    " << std::endl;
            std::cout<<"  Total Iterations  :  "<< iteration << std::endl;
            std::cout<<"  Total Time taken  :  "<< duration << " sec" << std::endl;
            std::cout<<"  Error Norm        :  "<< res_norm << std::endl;
            std::cout<< " ------------------------------------------------------------------" <<std::endl;
        }
			

	}

	else if( solver == 7){
		std::cout << " --------------- RESTARTED GMRES SOLVER ---------------------- " <<std::endl;
		unsigned long int restartFOMFactor;
		if(InputData::RESTART_PARAMETER_FOM > matrix->row || InputData::RESTART_PARAMETER_FOM == 0 )
		{
			std::cout <<" THE RESTART PARAMETER OF FOM CANNOT BE GREATER THAN THE MATRIX DIMENSION " <<std::endl;
			std::cout << " THE RESTART VALUE IS SET TO THE MATRIX DIMENSION - "<< matrix->row/2 << std::endl;
			restartFOMFactor = matrix->row/2;
		}
			else
		{
			restartFOMFactor = InputData::RESTART_PARAMETER_FOM;
		}
		std::cout << " RESTARTED GMRES PARAMETER : " << restartFOMFactor << std::endl;
        auto start = omp_get_wtime();
        int iteration = 0;
        double res_norm = 1000;
        sparse_status_t sA;
        sparse_index_base_t p1 = SPARSE_INDEX_BASE_ZERO;
        sparse_matrix_t A1;
        sparse_matrix_t H1;
        Sparse_Matrix* H = new Sparse_Matrix;

        
        //----------------------------- SETTING UP H MATRIX --------------//
        H->row = restartFOMFactor + 1 ; 
        H->col =  restartFOMFactor;
        long unsigned int n = (restartFOMFactor+1)*(restartFOMFactor+2);
        n = n/2 + 1 ;
        H->NNZSize = n;
        H->rowPtr.resize(H->row+1);
        H->colPtr.resize(H->NNZSize);
        H->values.resize(H->NNZSize);
        H->rowPtr[0] = 0;
        int colptr_val = 0;
        int rowptr_val = 0;
        double h_val;
        // End -- Setup Matrix for H //

        // Fill Row Pointer for the H MATRIX
        for( int  i = 0 ; i < restartFOMFactor + 1 ; i++)
        {
            if( i < 1)
                for (int j = 0 ; j < restartFOMFactor ; j++ )
                {
                    H->colPtr[colptr_val] = j;
                    colptr_val ++;
                    rowptr_val++;
                }
            else
            {
                for (int j = i-1 ; j < restartFOMFactor ; j++ )
                {
                    H->colPtr[colptr_val] = j;
                    colptr_val ++;
                    rowptr_val++;
                }
                
            }
            
            H->rowPtr[i+1] = H->rowPtr[i] + rowptr_val;
            rowptr_val = 0;
        }
        
        // --------      END SETTING UP H MATRIX---------------------------//
        
        std::vector<double> residual(matrix->row,0);
        
        // ----- Create INstances of MKL Sparcity matrices --------------------//
        sA = mkl_sparse_d_create_csr(&A1,SPARSE_INDEX_BASE_ZERO,
                                    matrix->row,matrix->col,&(matrix->rowPtr[0]),&(matrix->rowPtr[1]),&(matrix->colPtr[0]),&(matrix->values[0]));
        matrix_descr des;
        des.type = SPARSE_MATRIX_TYPE_GENERAL;
		
		sA = mkl_sparse_d_create_csr(&H1,SPARSE_INDEX_BASE_ZERO,
											H->row,H->col,&(H->rowPtr[0]),&(H->rowPtr[1]),&(H->colPtr[0]),&(H->values[0]));
        
        
        // -----  END Create INstances of MKL Sparcity matrices --------------------//
        while ( iteration < InputData::MAX_ITER && res_norm > InputData::TOLERANCE){
            GMRES_RESTARTED_CSR(*matrix, b, x, tolerance, Max_iter,H,restartFOMFactor,A1,H1);
        
        mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,1.0,A1,des,x,0.0,residual.data()); // Ax and save in residual
        cblas_daxpby(matrix->row,1.0,b,1.0,-1.0,residual.data(),1.0);                                 // Calculate b - Ax and save in Residual
        res_norm = cblas_dnrm2(matrix->row,residual.data(),1.0);  

        if(iteration % InputData::RESIDUAL_DISPLAY == 0)
        std::cout<<" Iteration : " << iteration << " Error Norm : "<< res_norm<<std::endl;
        iteration++;
        }
        auto stop = omp_get_wtime();
        auto duration =  stop - start;
        if(iteration == Max_iter)
        {
            std::cout << " Status :  RESTARTED GMRES  Iteration has [[NOT]]converged  ---- " << std::endl;
            std::cout<<"  Total Iterations  :  "<< iteration << std::endl;
            std::cout<<"  Total Time taken  :  "<< duration << " sec" << std::endl;
            std::cout<<"  Error Norm        :  "<< res_norm << std::endl;
            std::cout<< " ------------------------------------------------------------------" <<std::endl;
        }

        else
        {
            std::cout << " Status :  RESTARTED GMRES has Iteration has converged    " << std::endl;
            std::cout<<"  Total Iterations  :  "<< iteration << std::endl;
            std::cout<<"  Total Time taken  :  "<< duration << " sec" << std::endl;
            std::cout<<"  Error Norm        :  "<< res_norm << std::endl;
            std::cout<< " ------------------------------------------------------------------" <<std::endl;
        }
        

    }

    else
    {
        std::cout << " Solver type "<< solver << " has not been implemented "<< std:: endl;
    }
    


}
