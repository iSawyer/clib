

test: common.a container.a cmemory.a  math_functions_gpu.a math_functions_cpu.a  test.o
	g++ -L/usr/local/cuda-7.0/lib64  test.o  ./container.a ./cmemory.a ./math_functions_gpu.a ./math_functions_cpu.a ./common.a -lcuda -lcudart -lcublas -lgfortran -lblas  -lrt -lpthread -ldl cblas_LINUX.a libblas.a   
test.o: test.cpp
	g++ -c test.cpp -o test.o
	
cmemory.a: cmemory.cpp 
	g++ -c cmemory.cpp -o cmemory.o
	ar -rc cmemory.a cmemory.o
common.a: common.cpp
	g++ -c common.cpp -o common.o 
	ar -rc common.a common.o
container.a: container.cpp
	g++ -c container.cpp -o container.o
	ar -rc container.a container.o
math_functions_gpu.a: math_functions.cu
	nvcc -c math_functions.cu -o math_functions_gpu.o
	ar -rc math_functions_gpu.a math_functions_gpu.o
math_functions_cpu.a: math_functions.cpp
	g++ -c  math_functions.cpp -o math_functions_cpu.o
	ar -rc math_functions_cpu.a math_functions_cpu.o
clean:
	rm common.o cmemory.o container.o math_functions_gpu.o math_functions_cpu.o common.a cmemory.a container.a math_functions_gpu.a math_functions_cpu.a test.o
