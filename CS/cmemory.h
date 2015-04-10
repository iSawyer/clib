#ifndef CMEMORY_H_
#define CMEMORY_H_

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
		Cmemory():cpu_ptr(NULL),gpu_ptr(NULL),_head(UNINITIALIZED),size_(0),own_cpu_data(false){}
		Cmemory(size_t size__): cpu_ptr(NULL),gpu_ptr(NULL),_head(UNINITIALIZED),size_(size__),own_cpu_data(false){}
		~Cmemory();
		enum Mem_head { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, SYNCED };
		const void* cpu_data();
		const void* gpu_data();
		void set_cpu_data(void* data);
		void* mutable_cpu_data();
		void* mutable_gpu_data();
		Mem_head head()	{ return _head;}
		size_t size()	{ return size_; }


	private:
		void to_cpu();
		void to_gpu();
		void* cpu_ptr;
		void* gpu_ptr;
		Mem_head _head;
		size_t size_;
		bool own_cpu_data;
		
	};


#endif
