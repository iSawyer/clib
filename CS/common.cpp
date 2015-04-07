#include "common.h"
#include <cstdio>
#include <ctime>


int64_t seed(){
  pid = getpid();
  s = time(NULL);
  seed = abs(((s * 181) * ((pid - 83) * 359)) % 104729);
  return seed;
}


share_ptr<Csingleton> Csingleton:: csingleton;
share_ptr<mt19937> Csingleton:: rng;

Csingleton:: Csingleton(){
	int64_t seed = seed();
	rng = new(mt19937(seed));
	cublasCreate(&cublas_handle);
}

Csingleton:: ~Csingleton(){
	if(cublas_handle){
		cublasDestroy(cublas_handle);
	}
}

static mt19937& Csingleton::rng(){
	return *(Get().rng);
}

static cublasHandle_t Csingleton::cublas_handle() { return Get().cublas_handle; }



