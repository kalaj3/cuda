#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define MASK_WIDTH 3
#define TILE_WIDTH 4
//@@ Define constant memory for device kernel here

__constant__ float con_mask[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here

	
__shared__ float N[8][8][8];

int bx = blockIdx.x;
int by = blockIdx.y;
int bz = blockIdx.z;

int tx = threadIdx.x;
int ty = threadIdx.y;
int tz = threadIdx.z;

//target position output per mat
int x_o = bx*TILE_WIDTH+tx; //row
int y_o = by*TILE_WIDTH+ty; //col
int z_o = bz*TILE_WIDTH+tz; //height

//input positions 

int x_i = x_o - 1;
int y_i = y_o - 1;
int z_i = z_o - 1;
	
 if ((x_i >=0) && (x_i < x_size) &&
     (y_i >=0) && (y_i < y_size) &&  
     (z_i >=0) && (z_i < z_size)){
 
 
 N[tz][ty][tx]= input[z_i*(y_size*x_size) + y_i * x_size + x_i];
 
 }
 else {
 N[tz][ty][tx]= 0.0f;
 
 } 

 

__syncthreads();


float p_val = 0.0f;
if ( tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH && 
		x_o < x_size && y_o < y_size && z_o < z_size){
	
	for (int h = 0 ; h < MASK_WIDTH; h++){

		for (int c = 0 ; c < MASK_WIDTH; c++){

			for (int r = 0 ; r < MASK_WIDTH; r++){

				p_val += con_mask[h][c][r]*N[tz+h][ty+c][tx+r];
				 
			}	
		}
	}	
	
 	output[z_o*(y_size*x_size) + y_o * x_size + x_o] = p_val;
 }
 
}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  //@@ Initial deviceInput and deviceOutput here.
 
  float *deviceInput;
  float *deviceOutput;
  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);


  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
 int input_len = inputLength-3;


 cudaMalloc((void **)&deviceInput, input_len *sizeof(float));
 cudaMalloc((void **)&deviceOutput, input_len *sizeof(float));

  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu

	cudaMemcpy(deviceInput,&hostInput[3],input_len*sizeof(float),cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(con_mask,hostKernel, MASK_WIDTH*MASK_WIDTH*MASK_WIDTH*sizeof(float));
  //@@ Initialize grid and block dimensions here

	dim3 dimGrid(ceil((1.0*x_size)/TILE_WIDTH), ceil((1.0*y_size)/TILE_WIDTH),ceil((1.0*z_size)/TILE_WIDTH));
	dim3 dimBlock(8,8,8);
  //@@ Launch the GPU kernel here
	conv3d<<<dimGrid,dimBlock>>>(deviceInput,deviceOutput,z_size,y_size,x_size);
  
	
	cudaDeviceSynchronize();


  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)


cudaMemcpy(&hostOutput[3],deviceOutput,input_len*sizeof(float),cudaMemcpyDeviceToHost);

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  //@@ Free device memory
	cudaFree(deviceInput);
	cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}

