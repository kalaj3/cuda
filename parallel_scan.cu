// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}
#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this
#define zero 0
#define one 1
#define two 2
#define BSIZE 1024

#define wbCheck(stmt)                                                      \
    do                                                                     \
    {                                                                      \
        cudaError_t err = stmt;                                            \
        if (err != cudaSuccess)                                            \
        {                                                                  \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                    \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err)); \
            return -1;                                                     \
        }                                                                  \
    } while (0)

__global__ void scan(float *input, float *output, int len) {
    __shared__ float seg[BSIZE ];

    unsigned int threadID = threadIdx.x;
    unsigned int blockStart = two * blockIdx.x * blockDim.x;
    
    unsigned int tid2 = threadID*2;
    unsigned int idx1 = blockStart + tid2;
    unsigned int idx2 = blockStart + tid2 + one;
    

   
    seg[tid2] = (idx1 < len) ? input[idx1] : zero;
    seg[tid2 + one] = (idx2 < len) ? input[idx2] :zero;

   
    __syncthreads();

   
    for (int stride = one; stride < BSIZE; stride *= two) {
        int index = (threadID + one) * stride * two - one;
        if (index < BSIZE) {
            seg[index] += seg[index - stride];
        }
        __syncthreads();
    }

   
    for (int stride = BLOCK_SIZE / two; stride > zero; stride /= two) {
        int index = (threadID + one) * stride * two - one;
        if (index + stride < BSIZE) {
            seg[index + stride] += seg[index];
        }
        __syncthreads();
    }

   
    if (idx1 < len) input[idx1] = seg[tid2];
    if (idx2 < len) input[idx2] = seg[tid2+one];

   
    if (len > BSIZE && threadID ==zero) {
        output[blockIdx.x] = seg[BSIZE - one];
    }
}

__global__ void add(float *input, float *output, int len) {
    unsigned int startIdx = blockIdx.x * blockDim.x;
    unsigned int globalIdx = startIdx + threadIdx.x;

    if (blockIdx.x > zero && globalIdx < len) {
        input[globalIdx] += output[blockIdx.x - one];
    }
}

int main(int argc, char **argv)
{
    wbArg_t args;
    float *hostInput;  // The input 1D list
    float *hostOutput; // The output list
    float *deviceInput;
    float *deviceOutput;
    int numElements; // number of elements in the list
    int numBlocks;

    args = wbArg_read(argc, argv);
    wbTime_start(Generic, "init malloc");
    hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
    hostOutput = (float *)malloc(numElements * sizeof(float));
    wbTime_stop(Generic, "init malloc");

    wbLog(TRACE, "The number of input elements in the input is ",
          numElements);

    numBlocks = numElements / (BLOCK_SIZE << 1);
    if (numBlocks == 0 || numBlocks % (BLOCK_SIZE << 1))
    {
        numBlocks++;
    }

    wbLog(TRACE, "The number of blocks is ",
          numBlocks);
    
    
    wbTime_start(GPU, "aloc gpu");
    wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
    wbCheck(cudaMalloc((void **)&deviceOutput, numBlocks * sizeof(float)));
    wbTime_stop(GPU, "aloc gpu");

    wbTime_start(GPU, "memcpy to device");
    wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                       cudaMemcpyHostToDevice));
    wbTime_stop(GPU, "memcpy to device");


    //@@ Initialize the grid and block dimensions here
    dim3 DimGrid(numBlocks, 1, 1);
    dim3 deviceOutputDimGrid(1, 1, 1);
    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    dim3 AddDimBlock(BLOCK_SIZE << 1, 1, 1);

    //@@ Modify this to complete the functionality of the scan
    //@@ on the deivce

    wbTime_start(Compute, "scan");
    scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements);
    scan<<<deviceOutputDimGrid, DimBlock>>>(deviceOutput, NULL, numBlocks);
    add<<<DimGrid, AddDimBlock>>>(deviceInput, deviceOutput, numElements);
    
    cudaDeviceSynchronize();

    wbTime_stop(Compute, "scan");

    wbTime_start(Copy, "memcpy to host");
    wbCheck(cudaMemcpy(hostOutput, deviceInput, numElements * sizeof(float),
                       cudaMemcpyDeviceToHost));
    wbTime_stop(Copy, "memcpy to host");

    cudaFree(deviceInput);
    cudaFree(deviceOutput);

    wbSolution(args, hostOutput, numElements);

    free(hostInput);
    free(hostOutput);

    return 0;
}
