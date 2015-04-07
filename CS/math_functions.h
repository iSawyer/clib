#ifndef MATH_FUNCTIONS_H_
#define MATH_FUNCTIONS_H_

#include <cmath>
#include <math.h>
#include "common.h"
// TODO: add more math functions
extern "C"{
	#include <cblas.h>
}
template <typename T>
void c_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const T alpha, const T* A, const T* B, const T beta,
    T* C);

template <typename T>
void c_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const T alpha, const T* A, const T* x, const T beta,
    T* y);

template <typename T>
void c_cpu_axpy(const int N, const T alpha, const T* X,
    T* Y);



template<typename T>
inline void c_rng_gaussian(const int n, const T u, const T sigma, T* r);


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
void c_gpu_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

#endif


