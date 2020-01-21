
NVCC           :="/usr/local/cuda/bin/nvcc"
NVCCFLAGS      :=-lineinfo -c -x cu -arch sm_75 -std=c++11 -O3

all: kernel art_test

art_test: art_test.cc
	$(CXX) -o art_test art_test.cc -std=c++11 -O3
	
kernel: kernel.o
	$(CXX) -o kernel kernel.o -L/usr/local/cuda/lib64 -lcuda -lcudart
	
kernel.o: kernel.cu
	$(NVCC) $(NVCCFLAGS)  kernel.cu

clean:
	rm -f kernel kernel.o
