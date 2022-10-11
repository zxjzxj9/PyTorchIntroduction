#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCKSIZE 1024

__device__ float sigmoid(float x) {
    return 1.0/(1+expf(-x));
}

__global__ void gelu_fwd_cuda(float* input, float* ret,
                              int64_t size) {
    int64_t idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < size) {
        ret[idx] = input[idx]*sigmoid(1.702*input[idx]);
    }
}

__global__ void gelu_bwd_cuda(float* grad_out, float* input,
                              float* ret, int64_t size) {
    int64_t idx = threadIdx.x + blockIdx.x*blockDim.x;
    if(idx < size) {
        float tmp = sigmoid(1.702*input[idx]);
        ret[idx] = grad_out[idx]*(tmp + 1.702*input[idx]*tmp*(1-tmp));
    }
}

__host__ void gelu_fwd_interface(float* input, float* ret, int64_t size) {
    int64_t nblock = (size + BLOCKSIZE - 1)/BLOCKSIZE;
    gelu_fwd_cuda<<<nblock, BLOCKSIZE>>>(input, ret, size);
}

__host__ void gelu_bwd_interface(float* grad_out, float* input, float* ret,
                        int64_t size) {
    int64_t nblock = (size + BLOCKSIZE - 1)/BLOCKSIZE;
    gelu_bwd_cuda<<<nblock, BLOCKSIZE>>>(grad_out, input,
                                         ret, size);
}
