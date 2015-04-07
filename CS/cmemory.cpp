#include "cmemory.hpp"
#include <cstring>

#include "common.h"
#include "math_functions.h"
	Cmemory::~Cmemory(){
		if(cpu_ptr && own_cpu_data){
			Cfree(cpu_ptr);
		}

		if(gpu_ptr){
			cudaFree(gpu_ptr);
		}

	}


	inline void Cmemory::to_cpu(){
		switch(_head){

			case UNINITIALIZED:
				Cmalloc(&cpu_ptr,size);
				memset(size,0,cpu_ptr);
				_head = HEAD_AT_CPU;
				own_cpu_data = true;
				break;

			case HEAD_AT_GPU:
				if(cpu_ptr){
					// use C_gpu_memcpy in math.cu
					cudaMemcpy(cpu_ptr,gpu_ptr,DeviceToHost);
				}
				else{
					Cmalloc(&cpu_ptr,size);
					cudaMemcpy(cpu_ptr,gpu_ptr,DeviceToHost);
					own_cpu_data = true;
				}
				_head = SYNCED;
				break;

			case HEAD_AT_CPU:
			case SYNCED:
				break;

		}
	}


	inline void Cmemory::to_gpu(){
	
		switch(_head){
			case UNINITIALIZED:
				cudaMalloc(&gpu_ptr,size);
				_head = HEAD_AT_GPU;
				break;
			case HEAD_AT_CPU:
				if(gpu_ptr==NULL){
					cudaMalloc(&gpu_ptr,size);
				}
				cudaMemcpy(gpu_ptr,cpu_ptr,HostToDevice);
				_head = SYNCED;
				break;
			case HEAD_AT_GPU:
			case SYNCED:
				break;
		}
		
	}
	const void* Cmemory::cpu_data(){
		to_cpu();
		return (const void*) cpu_ptr;
	}

	const void* Cmemory::gpu_data(){
		to_gpu();
		return (const void*) gpu_ptr;
	} 

	void* Cmemory::mutable_cpu_data(){
		to_cpu();
		_head = HEAD_AT_CPU;
		return cpu_ptr;
	}

	void* Cmemory::mutable_gpu_data(){
		to_gpu();
		_head = HEAD_AT_GPU;
		return gpu_ptr;
	}

	void Cmemory::set_cpu_data(void* data){
		if(own_cpu_data){
			Cfree(cpu_ptr);
		}
		cpu_ptr = data;
		_head = HEAD_AT_CPU;
		own_cpu_data = false;
	}


