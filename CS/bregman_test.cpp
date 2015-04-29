
#include <iostream>
#include "common.h"
#include "cmemory.h"
#include "container.h"
#include "math_functions.h"
#include <unistd.h>
#include <stdlib.h>
#include <cstdlib>
#include "solver.hpp"	
#include <ostream>
#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>
/* 
 * test for bregman algorithm
 */
using namespace std;
using namespace Eigen;
int main(){
	
	Container<float> A(1,50,200);
	Container<float> x_org(1,200,1);
	Container<float> y(1,50,1);
	Container<float> x(1,200,1);
	// read data from file
	const char* file_A = "A";
	const char* file_y = "y";
	const char* file_x = "x";
	A.read_from_text(file_A);
	y.read_from_text(file_y);
	x_org.read_from_text(file_x);
	
	
	float* A_ptr = A.mutable_cpu_data();
	float* y_ptr = y.mutable_cpu_data();
	float* x_org_ptr = x_org.mutable_cpu_data();
	float* x_ptr = x.mutable_cpu_data();
	
	float tau = 0.01;
	float lambda =1000;
	float esp = 10e-5;
	float alpha = 1.0;
	float beta = 0.0;
	int iter = 150;
	std::cout<<"bregman algorithm begin"<<std::endl;
	Sovler<float>:: linear_bg_gpu(A,y,x,esp,lambda,tau,iter,false);
	x.Log_data();

	return 0;
}
