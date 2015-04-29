#ifndef MATH_FUNCTIONS_H_
#define MATH_FUNCTIONS_H_

#include <cmath>
#include <math.h>
#include "common.h"
// TODO: add more math functions
extern "C"{
	#include "cblas.h"
}
template <typename T>
void c_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const T alpha, const T* A, const T* B, const T beta,
    T* C);

// y = alpha * A(M,N) * x[M,1] + beta*y
template <typename T>
void c_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const T alpha, const T* A, const T* x, const T beta,
    T* y);


// y[j] = alpha * x[k] + y[j]
template <typename T>
void c_cpu_axpy(const int N, const T alpha, const T* X,
    T* Y);

template <typename T>
void c_cpu_set(const int N, const T alpha, T* X);

template <typename T>
void c_copy(const int N, const T* X, T* Y);


// soft shrinkage
template <typename T>
void c_cpu_soft(const int N, const T lambda, const T* X, T* Y);


// X = alpha * X
template <typename T>
void c_cpu_scalar(const int N, const T alpha,T* X);

// L2-norm
template <typename T>
T c_cpu_dot(const int N,const T* X,const T* Y);

template<typename T>
void c_rng_gaussian(const int n, const T u, const T sigma, T* r);


/*  
	

	GPU version


*/
/* GPU version */
template <typename T>
void c_gpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const T alpha, const T* A, const T* B, const T beta,
    T* C);

template <typename T>
void c_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const T alpha, const T* A, const T* x, const T beta,
    T* y);

template <typename T>
void c_gpu_axpy(const int N, const T alpha, const T* X,
    T* Y);

template<typename T>
void c_gpu_soft(const int N, const T lambda, const T* X,  T* Y);

template<typename T>
T c_gpu_dot(const int N, const T* X, const T* Y);

template<typename T>
void c_gpu_scalar(const int N, const T alpha, T* X); 



#endif


