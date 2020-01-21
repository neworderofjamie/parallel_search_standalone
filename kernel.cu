// Standard C++ includes
#include <algorithm>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <sstream>

// Standard C includes
#include <cassert>
#include <cmath>

// CUDA includes
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>

// Include nightmare array
#include "big_array.h"

//------------------------------------------------------------------------
// Macros
//------------------------------------------------------------------------
#define CHECK_CUDA_ERRORS(call) {                                                                   \
    cudaError_t error = call;                                                                       \
    if (error != cudaSuccess) {                                                                     \
            std::ostringstream errorMessageStream;                                                  \
            errorMessageStream << "cuda error:" __FILE__ << ": " << __LINE__ << " ";                \
            errorMessageStream << cudaGetErrorString(error) << "(" << error << ")" << std::endl;    \
            throw std::runtime_error(errorMessageStream.str());                                     \
        }                                                                                           \
    }

template<typename T>
using HostDeviceArray = std::pair < T*, T* > ;

//-----------------------------------------------------------------------------
__global__ void testPerThreadBisect(unsigned int offset, uint16_t *d_outIndex, 
                                    const unsigned int* __restrict__ d_mergedPresynapticUpdateGroupStartID1)
{
    const unsigned int id = 32 * blockIdx.x + threadIdx.x; 
    const unsigned int offsetID = offset + id;
    
    unsigned int lo = 0;
    unsigned int hi = 62496;
    while(lo < hi) {
        const unsigned int mid = (lo + hi) / 2;
        if(offsetID < __ldg(d_mergedPresynapticUpdateGroupStartID1 + mid)) {
            hi = mid;
        }
        else {
            lo = mid + 1;
        }
    }
    
    // Write index to output array
    d_outIndex[id] = lo - 1;
}

//-----------------------------------------------------------------------------
// Host functions
//-----------------------------------------------------------------------------
template<typename T>
HostDeviceArray<T> allocateHostDevice(unsigned int count)
{
    T *array = nullptr;
    T *d_array = nullptr;
    CHECK_CUDA_ERRORS(cudaMallocHost(&array, count * sizeof(T)));
    CHECK_CUDA_ERRORS(cudaMalloc(&d_array, count * sizeof(T)));

    return std::make_pair(array, d_array);
}
//-----------------------------------------------------------------------------
template<typename T>
void hostToDeviceCopy(HostDeviceArray<T> &array, unsigned int count, bool deleteHost=false)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.second, array.first, sizeof(T) * count, cudaMemcpyHostToDevice));
    if (deleteHost) {
        CHECK_CUDA_ERRORS(cudaFreeHost(array.first));
        array.first = nullptr;
    }
}
//-----------------------------------------------------------------------------
template<typename T>
void deviceToHostCopy(HostDeviceArray<T> &array, unsigned int count)
{
    CHECK_CUDA_ERRORS(cudaMemcpy(array.first, array.second, count * sizeof(T), cudaMemcpyDeviceToHost));
}
//-----------------------------------------------------------------------------
void loadBalanceSearchDumbCPU(unsigned int offset, unsigned int numThreads, unsigned int numGroups,
                              uint16_t *outIndex, const unsigned int *mergedPresynapticUpdateGroupStartID1)
{
    for(unsigned int i = 0; i < numThreads; i++) {
        const auto u = std::upper_bound(&mergedPresynapticUpdateGroupStartID1[0], &mergedPresynapticUpdateGroupStartID1[numGroups],
                                        i + offset);
        
        outIndex[i] = std::distance(mergedPresynapticUpdateGroupStartID1, u) - 1;
    }
}
//-----------------------------------------------------------------------------
void loadBalanceSearchCPU(unsigned int offset, unsigned int numThreads, unsigned int numGroups,
                          uint16_t *outIndex, const unsigned int *mergedPresynapticUpdateGroupStartID1)
{
    unsigned int t = 0;
    unsigned int g = 0;
    while(t < numThreads || g < numGroups) {
        bool p;
        if(g >= numGroups) {
            p = true;
        }
        else if(t >= numThreads) {
            p = false;
        }
        else {
            p = ((t + offset) < mergedPresynapticUpdateGroupStartID1[g]);
        }
        
        if(p) {
            outIndex[t++] = g - 1;
        }
        else {
            g++;
        }
    }
}
//-----------------------------------------------------------------------------
int main()
{
    const unsigned int numGroups = sizeof(mergedPresynapticUpdateGroupStartID1) / sizeof(unsigned int);
    const unsigned int offset = 32923392;
    const unsigned int numThreads = 1017051648;
    CHECK_CUDA_ERRORS(cudaSetDevice(0));

    // Create events
    cudaEvent_t testStart;
    cudaEvent_t testStop;
    CHECK_CUDA_ERRORS(cudaEventCreate(&testStart));
    CHECK_CUDA_ERRORS(cudaEventCreate(&testStop));

    // Create output array
    auto outIndex = allocateHostDevice<uint16_t>(numThreads);
    hostToDeviceCopy(outIndex, numThreads);

    loadBalanceSearchCPU(offset, numThreads, numGroups, outIndex.first, mergedPresynapticUpdateGroupStartID1);
    // Create device version of presynaptic update group start ids
    /*unsigned int *d_mergedPresynapticUpdateGroupStartID1;
    CHECK_CUDA_ERRORS(cudaMalloc(&d_mergedPresynapticUpdateGroupStartID1, numGroups * sizeof(unsigned int)));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_mergedPresynapticUpdateGroupStartID1, mergedPresynapticUpdateGroupStartID1, 
                                 sizeof(unsigned int) * numGroups, cudaMemcpyHostToDevice));

    // Run kernel
    CHECK_CUDA_ERRORS(cudaEventRecord(testStart));
    const dim3 threads(32, 1);
    const dim3 grid(31782864, 1);
    testPerThreadBisect<<<grid, threads>>>(offset, outIndex.second, d_mergedPresynapticUpdateGroupStartID1);
    CHECK_CUDA_ERRORS(cudaPeekAtLastError());
    CHECK_CUDA_ERRORS(cudaEventRecord(testStop));

    // Get kernel time
    float time;
    CHECK_CUDA_ERRORS(cudaEventSynchronize(testStop));
    CHECK_CUDA_ERRORS(cudaEventElapsedTime(&time, testStart, testStop));
    std::cout << "Search kernel takes " << time << "ms" << std::endl;

    // Copy output indices 
    deviceToHostCopy(outIndex, numThreads);*/

    // Verify binning
    unsigned int nextGroup = 1;
    for(unsigned int i = 0; i < numThreads; i++) {
        if((i + offset) == mergedPresynapticUpdateGroupStartID1[nextGroup]) {
            nextGroup++;
        }
        assert(outIndex.first[i] == nextGroup - 1);
    }
    std::cout << "Output correct!" << std::endl;

    return EXIT_SUCCESS;
}
