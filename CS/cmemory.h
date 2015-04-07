#ifndef CMEMORY_HPP_
#define CMEMORY_HPP_

#include "common.h"
#include <cstdlib>
#include "math_functions.h"
// include cuda or math 
/* 底层内存控制 */

/* 只负责内存，与类型无关，声明为void  */


	inline void Cmalloc(void** ptr,size_t size){
		*ptr = malloc(size);
	}
	inline void Cfree(void* ptr){
		free(ptr);
	}


	class Cmemory{
	public:
		Cmemory():cpu_ptr(NULL),gpu_ptr(NULL),head(UNINITIALIZED),size(0),own_cpu_data(false){}
		Cmemory(size_t size_): cpu_ptr(NULL),gpu_ptr(NULL),head(UNINITIALIZED),size(size_),own_cpu_data(flase){}
		~Cmemory();
		enum Mem_head { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
		const void* cpu_data();
		const void* gpu_data();
		void set_cpu_data(void* data);
		void* mutable_cpu_data();
		void* mutable_gpu_data();
		Mem_head head()	{ return _head;}
		size_t size()	{ return size; }


	private:
		void to_cpu();
		void to_gpu();
		void* cpu_ptr;
		void* gpu_ptr;
		Mem_head _head;
		size_t size;
		bool own_cpu_data;
		DISABLE_COPY_AND_ASSIGN(Cmemory);
	};


#endif