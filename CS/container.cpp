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

// shi li hua
template class Container<float>;
template class Container<double>;
template class Container<int>;
template class Container<unsigned int>;



