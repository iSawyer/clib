#include "common.h"
#include <cstdio>
#include <ctime>
#include <boost/shared_ptr.hpp>
#include <cublas_v2.h>
#include <iostream>
shared_ptr<Csingleton> Csingleton:: csingleton;
shared_ptr<mt19937> Csingleton:: rng_;

Csingleton:: Csingleton(){
	//int64_t seed_ = seed();
	// share_ptr features
	//rng_ = new(mt19937(seed_));
	//rng_.reset(new(mt19937(seed_)));
	cublasStatus_t stat;
	cublasHandle_t h;
	stat = cublasCreate_v2(&cublas_handle_);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        printf ("CUBLAS initialization failed\n");
        exit(0);
    }
}

Csingleton:: ~Csingleton(){
	if(cublas_handle){
		cublasDestroy(cublas_handle_);
	}
}





