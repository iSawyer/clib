#include "math_functions.h"
#include "common.h"
#include <cmath>
#include <cstdlib>
#include <cstring>

template <>
void c_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  (cublasSgemm(Csingleton::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void c_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  (cublasDgemm(Csingleton::cublas_handle(), cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void c_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  (cublasSgemv(Csingleton::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void c_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  (cublasDgemv(Csingleton::cublas_handle(), cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void c_gpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) {
  (cublasSaxpy(Csingleton::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template <>
void c_gpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) {
  (cublasDaxpy(Csingleton::cublas_handle(), N, &alpha, X, 1, Y, 1));
}

template<>
float c_gpu_dot<float>(const int N, const float* X, const float* Y){
	float res;
	cublasSdot(Csingleton::cublas_handle(),N,X,1,Y,1,&res);
	return res;
}

template<>
double c_gpu_dot<double>(const int N, const double* X, const double* Y){
	double res;
	cublasDdot(Csingleton::cublas_handle(),N,X,1,Y,1,&res);
	return res;
}

template<>
int c_gpu_dot<int>(const int N, const int	* X, const int* Y){
   printf("not implement\n");
   return 1;
}
template<>
unsigned int c_gpu_dot<unsigned int>(const int N, const unsigned int* X, const unsigned int* Y){
   printf("not implement\n");
   return 1;
}

template<typename T>
__global__ void scalar_kernel(const int N, const T alpha, T* X){
	CUDA_KERNEL_LOOP(index,N){
		X[index] = X[index] * alpha;
	}
}

template<>
void c_gpu_scalar<float>(const int N, const float alpha, float* X){
	scalar_kernel<float><<<C_GET_BLOCKS(N),C_CUDA_NUM_THREADS>>>(N, alpha, X);
}


template<>
void c_gpu_scalar<double>(const int N, const double alpha, double* X){
	scalar_kernel<double><<<C_GET_BLOCKS(N),C_CUDA_NUM_THREADS>>>(N, alpha, X);
}
template<>
void c_gpu_scalar<int>(const int N, const int alpha, int* X){
;
}
template<>
void c_gpu_scalar<unsigned int>(const int N, const unsigned int alpha, unsigned int* X){
;
}


template<typename T>
__global__ void soft_kernel(const int N, const T lambda, const T* X, T* Y){
	CUDA_KERNEL_LOOP(index,N){
		if(X[index] > lambda){
			Y[index] = X[index] - lambda;
		}
		else if(X[index] < -lambda){
			Y[index] = X[index] + lambda;
		}
		else{
			Y[index] = 0;
		}
	}
}

template<>
void c_gpu_soft<float>(const int N, const float lambda, const float* X, float* Y){
	soft_kernel<float><<<C_GET_BLOCKS(N),C_CUDA_NUM_THREADS>>>(N,lambda,X,Y);
}

template<>
void c_gpu_soft<double>(const int N, const double lambda, const double* X, double* Y){
	soft_kernel<double><<<C_GET_BLOCKS(N),C_CUDA_NUM_THREADS>>>(N,lambda,X,Y);
}
template<>
void c_gpu_soft<int>(const int N, const int lambda, const int* X, int* Y){

}
template<>
void c_gpu_soft<unsigned int>(const int N, const unsigned int lambda, const unsigned int* X, unsigned int* Y){

}


