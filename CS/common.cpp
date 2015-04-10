#include "common.h"
#include <cstdio>
#include <ctime>
#include <boost/shared_ptr.hpp>
#include <cublas_v2.h>

shared_ptr<Csingleton> Csingleton:: csingleton;
shared_ptr<mt19937> Csingleton:: rng_;

Csingleton:: Csingleton(){
	//int64_t seed_ = seed();
	// share_ptr features
	//rng_ = new(mt19937(seed_));
	//rng_.reset(new(mt19937(seed_)));
	cublasCreate(&cublas_handle_);
}

Csingleton:: ~Csingleton(){
	if(cublas_handle){
		cublasDestroy(cublas_handle_);
	}
}





