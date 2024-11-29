#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>

__device__ __half2 manual_hfma2(__half2 a, __half2 b, __half2 c)
{
    // a*b +c

    __half a_low = __low2half(a);
    __half a_high = __high2half(a);
    __half b_low = __low2half(b);
    __half b_high = __high2half(b);
    __half c_low = __low2half(c);
    __half c_high = __high2half(c);

    float fa_low = __half2float(a_low);
    float fa_high = __half2float(a_high);
    float fb_low = __half2float(b_low);
    float fb_high = __half2float(b_high);
    float fc_low = __half2float(c_low);
    float fc_high = __half2float(c_high);

    float fr_low = fa_low * fb_low + fc_low;
    float fr_high = fa_high * fb_high + fc_high;

    __half r_low = __float2half(fr_low);
    __half r_high = __float2half(fr_high);

    return __halves2half2(r_low, r_high);
}
#define TILE_WIDTH 16
__global__ void conv_forward_kernel(float *__restrict__ output, const float *__restrict__ input, const float *__restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width)
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
    int local_TW = TILE_WIDTH;
    const int H_out = Height - 6;
    const int W_out = Width - 6;
#define output4d(i3, i2, i1, i0) output[(i3) * (Map_out * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define input4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
#define mask4d(i3, i2, i1, i0) mask[(i3) * (Channel * 49) + (i2) * (49) + (i1) * (7) + i0]
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // int tz = threadIdx.z;
    // int dimx = blockDim.x;
    // int dimy = blockDim.y;
    // int dimz = blockDim.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int blocks_per_row = ceil((float)W_out / local_TW);

    // Calculate output positions

    int w_out = local_TW * (by % blocks_per_row) + tx;
    int h_out = local_TW * (by / blocks_per_row) + ty;

    __half2 in_h2, mask_h2;
    float cur_in, mp1, mp2, high_result;

    if (h_out < H_out && w_out < W_out)
    {
        __half2 result = __float2half2_rn(0.0f);
        for (int c = 0; c < Channel; c++)
        {

            for (int p = 0; p < 7; p++)
            {

                for (int q = 0; q < 7; q++)
                {
                    cur_in = input4d(bz, c, h_out + p, w_out + q);

                    in_h2 = __floats2half2_rn(cur_in, cur_in);

                    mp1 = mask4d(2 * bx, c, p, q);
                    mp2 = mask4d(2 * bx + 1, c, p, q); // Always compute mp2

                    mask_h2 = __floats2half2_rn(mp1, mp2);
                    result = manual_hfma2(in_h2, mask_h2, result);
                }
            }

        }

        output4d(bz, 2 * bx, h_out, w_out) = __low2float(result);
        high_result = __high2float(result);
        output4d(bz, 2 * bx + 1, h_out, w_out) = blockIdx.x < blockDim.x - 1 ? high_result : 0.0f;
    }

#undef output4d
#undef input4d
#undef mask4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    size_t outputSize = Batch * Map_out * (Height - 6) * (Width - 6) * sizeof(float);
    size_t inputSize = Batch * Channel * Height * Width * sizeof(float);
    size_t maskSize = Map_out * Channel * 49 * sizeof(float);

    cudaMalloc((void **)device_output_ptr, outputSize);

    cudaMalloc((void **)device_mask_ptr, maskSize);

    cudaMalloc((void **)device_input_ptr, inputSize);

    cudaMemcpy(*device_input_ptr, host_input, inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(*device_mask_ptr, host_mask, maskSize, cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int local_TW = TILE_WIDTH;
    int Height_out = (Height - 6);
    int Width_out = (Width - 6);
    int size_of_dimGrid = (int)ceil((float)Height_out / local_TW) * (int)ceil((float)(Width_out) / local_TW);

    // Calculate the grid and block dimensions for the kernel launch
    int h2_map = ceil((1.0 * Map_out) / 2);
    dim3 dimGrid(h2_map, size_of_dimGrid, Batch);
    dim3 dimBlock(local_TW, local_TW, 1);

    // Launch the convolution forward kernel
    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width);

    cudaDeviceSynchronize();
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    size_t cpy_size = Batch * Map_out * (Height - 6) * (Width - 6) * sizeof(float);
    cudaMemcpy(host_output, device_output, cpy_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
    cudaFree(device_mask);
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