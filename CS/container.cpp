#include "container.h"
#include "cmemory.h"
#include "common.h"
#include "math_functions.h"
template<typename T>
const T* Container<T>:: cpu_data() const {
	return (const T*) memory_ptr->cpu_data();
} 

template<typename T>
const T* Container<T>:: gpu_data() const {
	return (const T*) memory_ptr->gpu_data();

}

template<typename T>
T* Container<T>:: mutable_cpu_data(){
	return (T*) memory_ptr->mutable_cpu_data();
}

template<typename T>
T* Container<T>:: mutable_gpu_data(){
	return (T*) memory_ptr->mutable_gpu_data();
}

void Container<T>:: share_data(const Container& other){
	memory_ptr = other.memory_ptr();
}