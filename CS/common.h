/*
通用头文件, 包含常用头文件， 一个单例

*/
#ifndef COMMON_H_
#define COMMON_H_
#include <boost/shared_ptr.hpp>
#include <boost/random.hpp>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types


class Csingleton{
	// 单例模式
private:
	Csingleton();
	cublasHandle_t cublas_handle;
	static shared_ptr<Csingleton> csingleton;
	static shared_ptr<mt19937> rng;
public:
	~Csingleton();
	static Csingleton& Get(){
		if(!csingleton.get()){
			csingleton.reset(new Csingleton());
			return *csingleton;
		}
		else{
			return *csingleton;
		}
	}
	inline static mt19937& rng();
	inline static cublasHandle_t cublas_handle();
};

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
    const int CAFFE_CUDA_NUM_THREADS = 1024;
#else
    const int CAFFE_CUDA_NUM_THREADS = 512;
#endif

inline int CAFFE_GET_BLOCKS(const int N) {
  return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


#endif