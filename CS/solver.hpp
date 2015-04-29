#ifndef SOLVER_HPP_
#define SOLVER_HPP_
#include "container.h"
#include "common.h"
#include "math_functions.h"
#include <iostream> // for debug
/*Class Sovle with method*/

template <typename T>
class Sovler{
public:
	

	static Container<T>& linear_bg_cpu(const Container<T>&A, const Container<T>&y, Container<T>&x, T esp, T lambda, T tau, int iter_max, bool verbose = false);
	static Container<T>& linear_bg_gpu(const Container<T>&A, const Container<T>&y, Container<T>&x, T esp, T lambda, T tau, int iter_max, bool verbose = false);
private:
	Sovler(){}
	
};

// linear bregman algorithm 
template<typename T>
Container<T>& Sovler<T>:: linear_bg_cpu(const Container<T>& A, const Container<T>&y,Container<T>&x,T esp, T lambda, T tau, int iter_max, bool verbose){
	// check
	if( (A.width() != x.height()) || (A.height() != y.height())){
		fprintf(stderr,"dimension dismatch\n");
		return x;
	}
	
	
	const T* A_raw = A.cpu_data();
	const T* y_raw = y.cpu_data();
	T* x_ptr = x.mutable_cpu_data();
	Container<T> gd(A.num(),y.height(),y.width());
	Container<T> v(A.num(),x.height(),x.width());
	
	T* gd_ptr = gd.mutable_cpu_data();
	T* v_ptr = v.mutable_cpu_data();
	T err = INT_MAX;
	int iter = 0;
	while( iter < iter_max){
		
		// A * x
		c_cpu_gemv(CblasNoTrans,A.height(),A.width(),(T) -1., A_raw, x_ptr, (T) 0., gd_ptr);
		if(verbose){
			std::cout<<"in linear bregman function\n";
			std::cout<<"err:"<<err<<std::endl; 
		}
		// y + (-gd) 
		c_cpu_axpy(gd.height(),(T) 1.,y_raw, gd_ptr);
		
		err = gd.sumsq_data();
		if(verbose){
			std::cout<<"err"<<err<<std::endl;
		}
		// v = v + AT * gd 
		c_cpu_gemv(CblasTrans,A.height(),A.width(), (T) 1., A_raw, gd_ptr, (T) 1., v_ptr);
		
		// soft shrinkage
		c_cpu_soft(x.height(),lambda,v_ptr, x_ptr);
		
		
		// repalce with Container's method
		c_cpu_scalar(x.height(),tau,x_ptr);
		
		iter++;
	}
	return x;
}

template <typename T>
Container<T>& Sovler<T>:: linear_bg_gpu(const Container<T>& A, const Container<T>&y,Container<T>&x,T esp, T lambda, T tau, int iter_max, bool verbose){
	if( (A.width() != x.height()) || (A.height() != y.height())){
		fprintf(stderr,"dimension dismatch\n");
		return x;
	}
	
	
	const T* A_raw = A.gpu_data();
	const T* y_raw = y.gpu_data();
	T* x_ptr = x.mutable_gpu_data();
	Container<T> gd(A.num(),y.height(),y.width());
	Container<T> v(A.num(),x.height(),x.width());
	
	T* gd_ptr = gd.mutable_gpu_data();
	T* v_ptr = v.mutable_gpu_data();
	T err = INT_MAX;
	int iter = 0;
	while( iter < iter_max){
		
		// A * x
		c_gpu_gemv(CblasNoTrans,A.height(),A.width(),(T) -1., A_raw, x_ptr, (T) 0., gd_ptr);
		if(verbose){
			std::cout<<"in linear bregman function\n";
			std::cout<<"err:"<<err<<std::endl; 
		}
		// y + (-gd) 
		c_gpu_axpy(gd.height(),(T) 1.,y_raw, gd_ptr);
		
		err = gd.sumsq_data();
		std::cout<<err<<std::endl;
		if(verbose){
			std::cout<<"err"<<err<<std::endl;
		}
		// v = v + AT * gd 
		c_gpu_gemv(CblasTrans,A.height(),A.width(), (T) 1., A_raw, gd_ptr, (T) 1., v_ptr);
		
		// soft shrinkage
		c_gpu_soft(x.height(),lambda,v_ptr, x_ptr);
		
		
		// repalce with Container's method
		c_gpu_scalar(x.height(),tau,x_ptr);
		
		iter++;
	}
	return x;
}








#endif
