#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#define TILE_WIDTH 16
#define NUM_STREAMS 10

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass as described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch.
    The goal here is to be correct AND fast.

    Function parameter definitions: 
    output - output tensor
    input - input tensor
    mask - kernel/filter
    Batch - batch size (number of images in input)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    const int H_out = Height - K + 1;
    const int W_out = Width - K + 1;

#define output4d(i3, i2, i1, i0) output[(i3) * (Map_out * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define input4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
#define mask4d(i3, i2, i1, i0) mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // int tz = threadIdx.z;
    // int dimx = blockDim.x;
    // int dimy = blockDim.y;
    // int dimz = blockDim.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;

    int blocks_per_row = (W_out - 1) / TILE_WIDTH + 1;

    // Calculate output positions

    int w_out = TILE_WIDTH * (by % blocks_per_row) + tx;
    int h_out = TILE_WIDTH * (by / blocks_per_row) + ty;

    if (h_out < H_out && w_out < W_out)
    {
        float result = 0.0;
        for (int c = 0; c < Channel; c++)
        {
            for (int p = 0; p < K; p++)
            {
                for (int q = 0; q < K; q++)
                {
                    result += input4d(bz, c, h_out + p, w_out + q) * mask4d(bx, c, p, q);
                }
            }
        }
        output4d(bz, bx, h_out, w_out) = result;
    }

#undef output4d
#undef input4d
#undef mask4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(
    const float *host_output, const float *host_input, const float *host_mask,
    float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr,
    const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Initialize
    size_t i;
    size_t Height_out = Height - K + 1;
    size_t Width_out = Width - K + 1;
    size_t inputSize = Batch * Channel * Height * Width * sizeof(float);
    size_t maskSize = Map_out * Channel * K * K * sizeof(float);
    size_t outputSize = Batch * Map_out * Height_out * Width_out * sizeof(float);

    // Calculate chunk sizes for each stream
    size_t input_chunk_size = inputSize / NUM_STREAMS;
    // size_t mask_chunk_size = maskSize / NUM_STREAMS;
    size_t output_chunk_size = outputSize / NUM_STREAMS;

    size_t remaining_input_size = inputSize % NUM_STREAMS;
    // size_t remaining_mask_size = maskSize % NUM_STREAMS;
    size_t remaining_output_size = outputSize % NUM_STREAMS;

    size_t size_of_dimGrid = (int)ceil((float)Height_out / TILE_WIDTH) * (int)ceil((float)Width_out / TILE_WIDTH);
    cudaStream_t streams[NUM_STREAMS];
    for (i = 0; i < NUM_STREAMS; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    // Allocate memory on the device
    cudaMalloc((void **)device_input_ptr, inputSize);
    cudaMalloc((void **)device_mask_ptr, maskSize);
    cudaMalloc((void **)device_output_ptr, outputSize);

    cudaHostRegister((void *)host_input, inputSize, cudaHostRegisterDefault);
    // cudaHostRegister(device_mask_ptr, maskSize, cudaHostRegisterDefault);
    cudaHostRegister((void *)host_output, outputSize, cudaHostRegisterDefault);


    
    cudaMemcpy(*device_mask_ptr, host_mask, maskSize, cudaMemcpyHostToDevice);
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        
        // Calculate offsets for input, mask, and output
        int input_offset = i * input_chunk_size;
        // int mask_offset = i * mask_chunk_size;
        int output_offset = i * output_chunk_size;

        // last tream - un matched elements case - would remove to elim branch div id streams op time mattered
        if (i == NUM_STREAMS - 1)
        {
            input_chunk_size += remaining_input_size;
            // mask_chunk_size += remaining_mask_size;
            output_chunk_size += remaining_output_size;
        }

        // Asynchronous memory copy from host to device for input and mask
        cudaMemcpyAsync(*device_input_ptr + input_offset / sizeof(float), host_input + input_offset / sizeof(float), input_chunk_size, cudaMemcpyHostToDevice, streams[i]);
        // cudaMemcpyAsync(*device_mask_ptr + mask_offset / sizeof(float), host_mask + mask_offset / sizeof(float), mask_chunk_size, cudaMemcpyHostToDevice, streams[i]);

        // Kernel launch
        dim3 dimGrid(Map_out, size_of_dimGrid, Batch/NUM_STREAMS);

        dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);

        conv_forward_kernel<<<dimGrid, dimBlock, 0, streams[i]>>>(*device_output_ptr+ output_offset / sizeof(float), *device_input_ptr + input_offset / sizeof(float), *device_mask_ptr, Batch, Map_out, Channel, Height, Width, K);

        cudaMemcpyAsync((void *)(host_output + output_offset / sizeof(float)), *device_output_ptr + output_offset / sizeof(float), output_chunk_size, cudaMemcpyDeviceToHost, streams[i]);
    }
    
    
    for (int i = 0; i < NUM_STREAMS; i++)
    {
        // cudaStreamSynchronize(streams[i]);
        cudaStreamDestroy(streams[i]);
    }
    cudaHostUnregister((void*)host_input);
    cudaHostUnregister((void*)host_output);
    //cudaHostUnregister(device_mask_ptr);
    // Free device memory
    cudaFree(*device_input_ptr);
    cudaFree(*device_output_ptr);
    cudaFree(*device_mask_ptr);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
}

__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for (int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
        std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
        std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
        std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
        std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
        std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
        std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
        std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
    }
}