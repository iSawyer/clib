#ifndef CONTAINER_H_
#define CONTAINER_H_


/*c++0x 提供share_ptr功能 */
#include "cmemory.h"
#include "common.h"
#include "math_functions.h"
#include <boost/shared_ptr.hpp>	


template<typename T>
class Container{
public:
	Container(){
	}
	Container(size_t num_, size_t h_, size_t w_);
	
	// no need to write delete function

	const T* cpu_data() const;
	const T* gpu_data() const;
	T* mutable_cpu_data();
	T* mutable_gpu_data();

	size_t offset(size_t num_, size_t h_, size_t w_){
		// TODO: check bound 
		return (num_ * h + h_) * w + w_;
	}



	/* inline function */
	size_t size(){
		return size_;
	}
	size_t num(){
		return num_;
	}
	size_t height(){
		return h;
	}
	size_t width(){
		return w;
	}

	inline const shared_ptr<Cmemory>& data() const{
		return memory_ptr;
	}

	void share_data(const Container& other);

private:
	size_t num_;
	size_t h;
	size_t w;
	size_t size_;
	
	//指向Cmemory的智能指针
	shared_ptr<Cmemory> memory_ptr;


};

#endif
