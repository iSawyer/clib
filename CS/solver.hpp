#ifndef SOLVER_HPP_
#define SOLVER_HPP_
#include "container.h"
#include "common.h"
#include "math_functions.h"

/*Class Sovler with method*/

template <typename T>
class Sovler{
public:
	Sovler(){;}

	Container<T>& linear_bg_cpu(const container<T>&A, const container<T>&y);
	Container<T>& linear_bg_gpu(const container<T>&A, const container<T>&y);

};







#endif