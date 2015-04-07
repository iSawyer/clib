#ifndef CONTAINER_H__
#define CONTAINER_H__


/*c++0x 提供share_ptr功能 */
#include "cmemory.h"
#include "common.h"
#include "math_functions.h"
template<typename T>
class Container{
public:
	Container(size_t num_, size_t h_, size_t w_): num(num_), h(h_), w(w_), size(num_*h_*w_){
		// TODO:: check bound
		memory_ptr(new Cmemory(size));
	}

	// no need to write delete function

	const T* cpu_data() const;
	const T* gpu_data() const;
	T* mutable cpu_data();
	T* mutable gpu_data();

	size_t offset(size_t num_, size_t h_, size_t w_){
		// TODO: check bound 
		return (num_ * h + h_) * w + w_;
	}



	/* inline function */
	size_t size(){
		return size;
	}
	size_t num(){
		return num;
	}
	size_t height(){
		return h;
	}
	size_t width(){
		return w;
	}

	const shared_ptr<Cmemory>& memory_ptr(){
		return memory_ptr;
	}

	void share_data(const Container& other);

private:
	size_t num;
	size_t h;
	size_t w;
	size_t size;
	
	//指向Cmemory的智能指针
	shared_ptr<Cmemory> memory_ptr;


};

#endif