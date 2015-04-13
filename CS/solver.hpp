#ifndef SOLVER_HPP_
#define SOLVER_HPP_
#include "container.h"
#include "common.h"
#include "math_functions.h"

/*Class Sovler with method*/

template <typename T>
class Sovler{
public:
	Sovler(){;}

	Container<T>& linear_bg_cpu(const Container<T>&A, const Container<T>&y, Container<T>&x, T esp, T lambda, T tau, int iter_max, bool verbose = false);
	Container<T>& linear_bg_gpu(const Container<T>&A, const Container<T>&y, Container<T>&x, T esp, T lambda, T tau, int iter_max, bool verbose = false);

};

// xian xing bregman suanfa 
template<typename T>
Container<T>& Sovler<T>:: linear_bg_cpu(const Container<T>& A, const Constainer<T>&y,Container<T>&x,T esp, T lambda, T tau, int iter_max, bool verbose = false){
	// check
	if(A.width() != x.height() || A.height() != y.height()){
		fprintf(stderr,"dimension dismatch\n");
		x = new Container<T>(A.num(),A.width(),y.width());
	}
	
	
	const T* A_raw = A.cpu_data();
	const T* y_raw = y.cpu_data();
	T* x_ptr = x.mutable_cpu_data();
	Container<T> gd(A.num(),A.width(),y.width());
	Container<T> v(A.num(),A.width(),y.width());
	Container<T> v_err(A.num(),y.height(),y.width());
	T* err_ptr = v_err.mutable_cpu_data();
	T* gd_ptr = gd.mutable_cpu_data();
	T* v_ptr = v.mutable_cpu_data();
	T err = INT_MAX;
	int  iter = 0;
	/* 
	 *  v_(k+1) = v_(k) + A.T(f - Au_(k))
	 * 
	 * 
	 * 
	*/
	while(err > esp && iter < iter_max){
		// A * u(k)
		c_cpu_gemv(CblasNoTrans,A.height(),A.width(),T 1.0, A_raw, x_ptr, T 0.0, gd_ptr);
		
		// compute err
		c_copy(y.height()*y.width(),y_raw, err_ptr);
		c_cpu_axpy(y.height(), T -1., gd_ptr,err_ptr);
		err = v_err.sumsq_data();
		
		// y - gd ~ gd - y 
		c_cpu_axpy(gd.height(), T -1.,y_raw, gd_ptr);
		
		// -AT * gd
		c_cpu_gemv(CblasTrans,A.height(),A.width(),T -1., A_raw, T 0., gd_ptr);
		
		// v + gd
		c_cpu_axpy(v.height(),T 1.,gd_ptr,v_ptr);
		
		// update v done ,  update u
		// soft shrinkage
		c_cpu_soft(v.height(),lambda,v_ptr, x_ptr);
		
		// repalce with Container's method
		c_cpu_scalar(x.height(),tau,x_ptr);
		
	}
	
	
	//Cmalloc(&v, y.size());
	
	return x;
}




#endif
