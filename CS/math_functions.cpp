#include "math_functions.h"
#include "common.h"
#include <boost/random.hpp>
#include <limits>
#include <iostream>

template<>
void c_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void c_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template <>
void c_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void c_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void c_cpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void c_cpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }

template <>
void c_cpu_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void c_cpu_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }



template<typename T>
void c_cpu_soft(const int N, const T lambda, const T* X, T* Y){
	for(int i = 0; i < N; i++){
		Y[i] = abs(X[i]) > lambda ? (X[i] - lambda) : (0);  
	}
}

template<>
void c_cpu_soft<float>(const int N, const float lambda,const float* X, float* Y);

template<>
void c_cpu_soft<double>(const int N, const double lambda, const double* X, double* Y); 

template<typename T>
void c_cpu_scalar(const int N, const T alpha, T* X){
	for(int i = 0; i < N; i++){
		X[i] = alpha * X[i];
	}
}
template
void c_cpu_scalar<float>(const int N, const float alpha, float* X);
template
void c_cpu_scalar<double>(const int N, const double alpha, double* X);


template<>
float c_cpu_dot<float>(const int N, const float* X, const float* Y){
	return cblas_sdot(N, x, 1, y, 1);
}

template<>
double c_cpu_dot<double>(const int N, const double* X, const double* Y){
	return cblas_ddot(n, x, 1, y, 1);
}

template <typename T>
void c_rng_gaussian(const int n, const T u, const T sigma, T* r){
    boost::normal_distribution<T> random_distribution(u, sigma);
    boost::variate_generator<boost::mt19937&, boost::normal_distribution<T> >
      variate_generator(Csingleton::rng(), random_distribution);
    for (int i = 0; i < n; ++i) {
      r[i] = variate_generator();
  }
}


template
void c_rng_gaussian<float>(const int n, const float u,
                               const float sigma, float* r);

template
void c_rng_gaussian<double>(const int n, const double u,
                                const double sigma, double* r);
                                
                                
template <typename T>
void c_copy(const int N, const T* X, T* Y){
	if(X != Y){
		cudaMemcpy(Y,X,sizeof(T)*N,cudaMemcpyDefault);
	}
}

template
void c_copy<float>(const int N, const float* X, float* Y){
	if(X != Y){
		cudaMemcpy(Y,X,sizeof(float)*N,cudaMemcpyDefault);
	}
}


template
void c_copy<double>(const int N, const double* X, double* Y){
	if(X != Y){
		cudaMemcpy(Y,X,sizeof(double)*N,cudaMemcpyDefault);
	}
}

template
void c_copy<int>(const int N, const int* X, int* Y);


template<typename T>
void c_cpu_set(const int N, const T alpha, T* X){
	if(alpha == 0){
		memset(X,0,sizeof(T)*N);
		return;
	}
	for(int i = 0; i < N; i++){
		X[i] = alpha;
	}
}

template
void c_cpu_set<float>(const int N, const float alpha, float* X);

template
void c_cpu_set<double>(const int N, const double alpha, double* X);

template
void c_cpu_set<int>(const int N, const double alpha, int* X);
