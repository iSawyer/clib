/*************************************************************************
	> File Name: main.cpp
	> Author: 
	> Mail: 
	> Created Time: 2015年04月08日 星期三 12时19分54秒
 ************************************************************************/

#include <iostream>

#include "common.h"
#include "cmemory.h"
#include "container.h"
#include "math_functions.h"
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
int main(){
    int num = 10;
    int h = 10;
    int w = 10;
    Container<float>* test = new Container<float>(num,h,w);
    std::cout<<test->height()<<std::endl;
    float* ptr = test->mutable_cpu_data();
    //test->gpu_data();
    float u = 0.0;
    float sigma = 1.0;
    c_rng_gaussian(h*num*w,u,sigma,ptr);
    for(int i = 0; i < h*num*w;i++){
		std::cout<<ptr[i]<<" ";
	}
	std::cout<<"end"<<std::endl;
	float* gpu_ptr = test->mutable_gpu_data();
	
	
	
	std::cout<<std::endl;
    //cout<<endl;
    return 0;

}
