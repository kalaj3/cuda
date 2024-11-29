#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"
#include <cuda_fp16.h>
#define TILE_WIDTH 16
__constant__ __half2 b_pad_cmem[6000]; // low
__constant__ __half2 f_pad_cmem[6000]; // high

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
#define back_pad_mask4d(i3, i2, i1, i0) b_pad_cmem[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define front_pad_mask4d(i3, i2, i1, i0) f_pad_cmem[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // int tz = threadIdx.z;
    // int dimx = blockDim.x;
    // int dimy = blockDim.y;
    // int dimz = blockDim.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    int blocks_per_row = ceil((float)W_out / TILE_WIDTH);

    // Calculate output positions

    int w_out = TILE_WIDTH * (by % blocks_per_row) + tx * 2;
    int h_out = TILE_WIDTH * (by / blocks_per_row) + ty;

    float result = 0.0;
    __half2 in_h2, mask_h2, res_low, res_high, mp1, mp2;
    float cur_in, high_result;

    __half2 results;
    if (h_out < H_out && w_out < W_out)
    {
        __half2 res_low = __float2half2_rn(0.0f);
        __half2 res_high = __float2half2_rn(0.0f);
        for (int c = 0; c < Channel; c++)
        {
            for (int p = 0; p < K; p++)
            {
                for (int q = 0; q < K; q++)
                {

                    cur_in = input4d(bz, c, h_out + p, w_out + q);

                    in_h2 = __floats2half2_rn(cur_in, cur_in);

                    mp1 = front_pad_mask4d(bx, c, p, q); // high (1st)
                    mp2 = back_pad_mask4d(bx, c, p, q);  // low (2nd)

                    res_high = __hfma2(mp1, in_h2, res_high);
                    res_low = __hfma2(mp2, in_h2, res_low);
                }
            }
        }

        output4d(bz, 2 * bx, h_out, w_out) = __low2float(res_high);
        high_result = __high2float(res_low);
        output4d(bz, 2 * bx + 1, h_out, w_out) = blockIdx.x < blockDim.x - 1 ? high_result : 0.0f;
    }

#undef output4d
#undef input4d
#undef mask4d
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    size_t num_input_elems = Batch * Channel * Height * Width;
    size_t num_mask_elems = Map_out * Channel * K * K;
    size_t num_output_elems = Batch * Map_out * (Height - K + 1) * (Width - K + 1);

    int h2_input = ceil((1.0 * num_input_elems) / 2);
    int h2_mask = ceil((1.0 * num_mask_elems) / 2);
    int i;
    // for (i = 0; i < h2_input; i++)
    // {
    //     ((__half2 *)host_input)[i] = __floats2half2_rn(host_input[2 * i], host_input[2 * i + 1]);
    // }

    size_t f_pad_idx = 0;
    size_t b_pad_idx = 0;

    __half2 b_mask_values[h2_mask]; // low
    __half2 f_mask_values[h2_mask]; // high

    for (int i = 0; i < h2_mask; i++)
    {

        int b_index = (i % K == K - 1) ? b_pad_idx++ : b_pad_idx;

        float b_val1 = host_mask[b_index];
        float b_val2 = (i % K == K - 1) ? host_mask[++b_pad_idx] : 0.0f;

        if (i % K != K - 1)
        {
            b_pad_idx += 2;
        }

        int f_index = (i % K == 0) ? f_pad_idx++ : f_pad_idx;

        float f_val1 = host_mask[f_index];
        float f_val2 = (i % K == 0) ? 0.0f : host_mask[++f_pad_idx];

        if (i % K != 0)
        {
            f_pad_idx += 2;
        }

        __half2 cur_mask_val_b = __floats2half2_rn(b_val1, b_val2);
        __half2 cur_mask_val_f = __floats2half2_rn(f_val1, f_val2);

        b_mask_values[i] = __floats2half2_rn(b_val1, b_val2);
        f_mask_values[i] = __floats2half2_rn(f_val1, f_val2);

    }
    cudaMemcpyToSymbol(b_pad_cmem, b_mask_values, sizeof(__half2) * h2_mask);
    cudaMemcpyToSymbol(f_pad_cmem, f_mask_values, sizeof(__half2) * h2_mask);

    cudaMalloc((void **)device_output_ptr, num_output_elems * sizeof(float));
    cudaMalloc((void **)device_input_ptr, num_input_elems * sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, num_input_elems * sizeof(float), cudaMemcpyHostToDevice);
}

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    int local_TW = TILE_WIDTH;
    int Height_out = (Height - K + 1);
    int Width_out = (Width - K + 1);
    int size_of_dimGrid = (int)ceil((float)Height_out / local_TW) * (int)ceil((float)(Width_out) / local_TW);

    // Calculate the grid and block dimensions for the kernel launch
    int h2_map = ceil((1.0 * Map_out) / 2);
    dim3 dimGrid(h2_map, size_of_dimGrid, Batch);
    dim3 dimBlock(local_TW, local_TW, 1);

    // Launch the convolution forward kernel
    conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);

    cudaDeviceSynchronize();
}

__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Copy the output back to host
    size_t cpy_size = Batch * Map_out * (Height - K + 1) * (Width - K + 1);

    //     for (int i = 0; i < cpy_size; i += 2) {
    //     __half2 half_pair = ((__half2*)device_output)[i / 2]; // Read __half2 pair once
    //     float high = __high2float(half_pair);  // Extract the high part
    //     float low = __low2float(half_pair);    // Extract the low part

    //     host_output[i] = high;
    //     host_output[i + 1] = low;
    // }

    cudaMemcpy(host_output, device_output, cpy_size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(device_input);
    cudaFree(device_output);
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
// #include <cmath>
// #include <iostream>
// #include "gpu-new-forward.h"
// #include <cuda_fp16.h>
// #define TILE_WIDTH 16

// __constant__ __half2 b_pad_cmem[6000]; // low
// __constant__ __half2 f_pad_cmem[6000]; // high

// __global__ void conv_forward_kernel(float *__restrict__ output, const float *__restrict__ input, const float *__restrict__ mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, int K)
// {
//     /*
//     Modify this function to implement the forward pass as described in Chapter 16.
//     We have added an additional dimension to the tensors to support an entire mini-batch.
//     The goal here is to be correct AND fast.
//     Function parameter definitions:
//     output - output tensor
//     input - input tensor
//     mask - kernel/filter
//     Batch - batch size (number of images in input)
//     Map_out - number of output feature maps
//     Channel - number of input feature maps
//     Height - input height dimension
//     Width - input width dimension
//     K - kernel height and width (K x K)
//     */
//     int local_TW = TILE_WIDTH;
//     const int H_out = Height - 6;
//     const int W_out = Width - 6;
// #define output4d(i3, i2, i1, i0) output[(i3) * (Map_out * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
// #define input4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
// #define mask4d(i3, i2, i1, i0) mask[(i3) * (Channel * 49) + (i2) * (49) + (i1) * (7) + i0]
// #define back_pad_mask4d(i3, i2, i1, i0) b_pad_cmem[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
// #define front_pad_mask4d(i3, i2, i1, i0) f_pad_cmem[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     // int tz = threadIdx.z;
//     // int dimx = blockDim.x;
//     // int dimy = blockDim.y;
//     // int dimz = blockDim.z;
//     int bx = blockIdx.x;
//     int by = blockIdx.y;
//     int bz = blockIdx.z;
//     int blocks_per_row = ceil((float)W_out / local_TW);

//     // Calculate output positions

//     int w_out = local_TW * (by % blocks_per_row) + tx;
//     int h_out = local_TW * (by / blocks_per_row) + ty;

//     __half2 in_h2, mask_h2, mp1, mp2;
//     float cur_in, high_result;

//     if (h_out < H_out && w_out < W_out)
//     {
//         __half2 result = __float2half2_rn(0.0f);
//         __half2 res_low = __float2half2_rn(0.0f);
//         __half2 res_high = __float2half2_rn(0.0f);
//         for (int c = 0; c < Channel; c++)
//         {

//             for (int p = 0; p < 7; p++)
//             {

//                 for (int q = 0; q < 7; q++)
//                 {
//                     cur_in = input4d(bz, c, h_out + p, w_out + q);

//                     in_h2 = __floats2half2_rn(cur_in, cur_in);

//                     mp1 = front_pad_mask4d(bx, c, p, q); // high (1st)
//                     mp2 = back_pad_mask4d(bx, c, p, q);  // low (2nd)

//                     res_high = __hfma2(in_h2,mp1,res_high);
//                     res_low = __hfma2(in_h2,mp2,res_low);
//                     __half high_half = __high2half(res_high); // Extract the high half from res_high
//                     __half low_half = __low2half(res_low);    // Extract the low half from res_low
//                     result = __halves2half2(low_half, high_half);

                    
//                 }
//             }
//         }

//         output4d(bz, 2 * bx, h_out, w_out) = __low2float(result);
//         high_result = __high2float(result);
//         output4d(bz, 2 * bx + 1, h_out, w_out) = blockIdx.x < blockDim.x - 1 ? high_result : 0.0f;
//     }

// #undef output4d
// #undef input4d
// #undef mask4d
// }

// __host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     size_t num_input_elems = Batch * Channel * Height * Width;
//     size_t num_mask_elems = Map_out * Channel * K * K;
//     size_t num_output_elems = Batch * Map_out * (Height - K + 1) * (Width - K + 1);

//     int h2_input = ceil((1.0 * num_input_elems) / 2);
//     int h2_mask = ceil((1.0 * num_mask_elems) / 2);
//     int i;
//     // for (i = 0; i < h2_input; i++)
//     // {
//     //     ((__half2 *)host_input)[i] = __floats2half2_rn(host_input[2 * i], host_input[2 * i + 1]);
//     // }

//     size_t f_pad_idx = 0;
//     size_t b_pad_idx = 0;
//     __half2 cur_mask_val;
//     for (int i = 0; i < h2_mask; i++)
//     {
//         // Handling b_pad_cmem
//         if (i % K == K - 1)
//         {
//             cur_mask_val = __floats2half2_rn(host_mask[b_pad_idx++], 0.0f);
//         }
//         else
//         {
//             cur_mask_val = __floats2half2_rn(host_mask[b_pad_idx], host_mask[b_pad_idx + 1]);
//             b_pad_idx += 2;
//         }
//         cudaMemcpyToSymbol(b_pad_cmem, &cur_mask_val, sizeof(__half2), i * sizeof(__half2));

//         // Handling f_pad_cmem
//         if (i % K == 0)
//         {
//             cur_mask_val = __floats2half2_rn(0.0f, host_mask[f_pad_idx++]);
//         }
//         else
//         {
//             cur_mask_val = __floats2half2_rn(host_mask[f_pad_idx], host_mask[f_pad_idx + 1]);
//             f_pad_idx += 2;
//         }
//         cudaMemcpyToSymbol(f_pad_cmem, &cur_mask_val, sizeof(__half2), i * sizeof(__half2));
//     }

//     // __half2 b_mask_values[h2_mask]; // low
//     // __half2 f_mask_values[h2_mask]; // high

//     // for (int i = 0; i < h2_mask; i++)
//     // {

//     //     int b_index = (i % K == K - 1) ? b_pad_idx++ : b_pad_idx;

//     //     float b_val1 = host_mask[b_index];
//     //     float b_val2 = (i % K == K - 1) ? host_mask[++b_pad_idx] : 0.0f;

//     //     if (i % K != K - 1)
//     //     {
//     //         b_pad_idx += 2;
//     //     }

//     //     int f_index = (i % K == 0) ? f_pad_idx++ : f_pad_idx;

//     //     float f_val1 = host_mask[f_index];
//     //     float f_val2 = (i % K == 0) ? 0.0f : host_mask[++f_pad_idx];

//     //     if (i % K != 0)
//     //     {
//     //         f_pad_idx += 2;
//     //     }

//     //     __half2 cur_mask_val_b = __floats2half2_rn(b_val1, b_val2);
//     //     __half2 cur_mask_val_f = __floats2half2_rn(f_val1, f_val2);

//     //     b_mask_values[i] = __floats2half2_rn(b_val1, b_val2);
//     //     f_mask_values[i] = __floats2half2_rn(f_val1, f_val2);

//     // }
//     // cudaMemcpyToSymbol(b_pad_cmem, b_mask_values, sizeof(__half2) * h2_mask);
//     // cudaMemcpyToSymbol(f_pad_cmem, f_mask_values, sizeof(__half2) * h2_mask);

//     cudaMalloc((void **)device_output_ptr, num_output_elems * sizeof(float));
//     cudaMalloc((void **)device_input_ptr, num_input_elems * sizeof(float));
//     cudaMemcpy(*device_input_ptr, host_input, num_input_elems * sizeof(float), cudaMemcpyHostToDevice);
// }

// __host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     int local_TW = TILE_WIDTH;
//     int Height_out = (Height - 6);
//     int Width_out = (Width - 6);
//     int size_of_dimGrid = (int)ceil((float)Height_out / local_TW) * (int)ceil((float)(Width_out) / local_TW);

//     // Calculate the grid and block dimensions for the kernel launch
//     int h2_map = ceil((1.0 * Map_out) / 2);
//     dim3 dimGrid(Map_out, size_of_dimGrid, Batch);
//     dim3 dimBlock(local_TW, local_TW, 1);

//     // Launch the convolution forward kernel
//     conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);

//     cudaDeviceSynchronize();
// }

// __host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
// {
//     // Copy the output back to host
//     size_t cpy_size = Batch * Map_out * (Height - 6) * (Width - 6) * sizeof(float);
//     cudaMemcpy(host_output, device_output, cpy_size, cudaMemcpyDeviceToHost);

//     // Free device memory
//     cudaFree(device_input);
//     cudaFree(device_output);
//     cudaFree(device_mask);
// }
// __host__ void GPUInterface::get_device_properties()
// {
//     int deviceCount;
//     cudaGetDeviceCount(&deviceCount);
//     for (int dev = 0; dev < deviceCount; dev++)
//     {
//         cudaDeviceProp deviceProp;
//         cudaGetDeviceProperties(&deviceProp, dev);
//         std::cout << "Device " << dev << " name: " << deviceProp.name << std::endl;
//         std::cout << "Computational capabilities: " << deviceProp.major << "." << deviceProp.minor << std::endl;
//         std::cout << "Max Global memory size: " << deviceProp.totalGlobalMem << std::endl;
//         std::cout << "Max Constant memory size: " << deviceProp.totalConstMem << std::endl;
//         std::cout << "Max Shared memory size per block: " << deviceProp.sharedMemPerBlock << std::endl;
//         std::cout << "Max threads per block: " << deviceProp.maxThreadsPerBlock << std::endl;
//         std::cout << "Max block dimensions: " << deviceProp.maxThreadsDim[0] << " x, " << deviceProp.maxThreadsDim[1] << " y, " << deviceProp.maxThreadsDim[2] << " z" << std::endl;
//         std::cout << "Max grid dimensions: " << deviceProp.maxGridSize[0] << " x, " << deviceProp.maxGridSize[1] << " y, " << deviceProp.maxGridSize[2] << " z" << std::endl;
//         std::cout << "Warp Size: " << deviceProp.warpSize << std::endl;
//     }
// }