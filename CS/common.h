/*
通用头文件, 包含常用头文件， 一个单例

*/
#ifndef COMMON_H_
#define COMMON_H_

#include <boost/shared_ptr.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>
#include <fstream>
#include <iostream>
#include <utility>
#include <stdlib.h>
#include <unistd.h>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>	
#include <cuda_runtime.h>
#include <curand.h>
#include <driver_types.h>  // cuda driver types
#include <ostream>

//using namespace std;
typedef boost::mt19937 mt19937;
using boost::shared_ptr;



/*
int64_t my_seed(){
  pid_t pid = getpid();
  size_t s = time(NULL);
  int64_t seed_ = abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed_;
}*/


class Csingleton{
	// 单例模式
private:
	Csingleton();
	cublasHandle_t cublas_handle_;
	static shared_ptr<Csingleton> csingleton;
	static shared_ptr<mt19937> rng_;
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
	
	inline static mt19937& rng(){
		//int64_t seed_ = my_seed();
		rng_.reset(new mt19937(1));
		return *(Get().rng_);
	}

	inline static cublasHandle_t cublas_handle() { return Get().cublas_handle_; }
	
};

// CUDA: thread number configuration.
// Use 1024 threads per block, which requires cuda sm_2x or above,
// or fall back to attempt compatibility (best of luck to you).
#if __CUDA_ARCH__ >= 200
    const int C_CUDA_NUM_THREADS = 1024;
#else
    const int C_CUDA_NUM_THREADS = 512;
#endif

inline int C_GET_BLOCKS(const int N) {
  return (N + C_CUDA_NUM_THREADS - 1) / C_CUDA_NUM_THREADS;
}

// CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)


#endif

