#include "container.h"
#include "cmemory.h"
#include "common.h"
#include "math_functions.h"

// xu yao zai zui hou shi li hua mo ban

template <typename T>
Container<T>:: Container(size_t num_, size_t h_, size_t w_): num_(num_), h(h_), w(w_), size_(num_*h_*w_){
		// TODO:: check bound		
		memory_ptr.reset(new Cmemory(size_*sizeof(T)));
		
}


template<typename T>
const T* Container<T>:: cpu_data() const {
	return (const T*) memory_ptr->cpu_data();
} 


template<typename T>
const T* Container<T>:: gpu_data() const {
	return (const T*) memory_ptr->gpu_data();

}

/*
template<>
const float* Container<float>:: gpu_data() const {
	return (const float*) memory_ptr->gpu_data();

}*/

template<typename T>
T* Container<T>:: mutable_cpu_data(){
	return (T*) memory_ptr->mutable_cpu_data();
}

template<typename T>
T* Container<T>:: mutable_gpu_data(){
	return (T*) memory_ptr->mutable_gpu_data();
}

template <typename T>
void Container<T>:: share_data(const Container& other){
	memory_ptr = other.data();
}

template <typename T>
T Container<T>:: sumsq_data() const{
	T sumsq;
	const T* data;
	if (!memory_ptr) { return 0; }
	switch (memory_ptr->head()) {
		case Cmemory::HEAD_AT_CPU:
			data = cpu_data();
			sumsq = c_cpu_dot(count_, data, data);
			break;
		case Cmemory::HEAD_AT_GPU:
		case Cmemory::SYNCED:
			/*
			data = gpu_data();
			c_gpu_dot(count_, data, data, &sumsq);
			*/
			printf("not implement\n");
			break;
		case Cmemory::UNINITIALIZED:
			return 0;
		default:
			;
	}
  return sumsq;
}

template <typename T>
void Container<T>:: scale_data(T scale_factor){
	if(!memory_ptr){
		return;
	}
	T * data;
	switch(memory_ptr->head()){
		case Cmemory:: HEAD_AT_CPU:
			data = mutable_cpu_data();
			c_cpu_scalar(size_, scale_factor,data);
			return;
		case Cmemory:: HEAD_AT_GPU:
		case Cmemory:: SYNCED:
			printf("not implement\n");
			break;
		case Cmemory:: UNINITIALIZED:
			printf("UNINITIALIZED\n");
			return;
		default:
			;
	}
}


// deep copy
template<typename T>
Container<T>:: Container(const Container& other){
	if(other.data() != this->memory_ptr){
		this = new(other.num(),other.height(),other.width());
		void* des = this->memory_ptr->mutable_cpu_data();
		void* src = other.data()->mutable_cpu_data();
		memcpy(des,src,this->size());
		
	}
}


// shi li hua
template class Container<float>;
template class Container<double>;
template class Container<int>;
template class Container<unsigned int>;



