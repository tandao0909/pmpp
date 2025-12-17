#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#define TILE_WIDTH 16

__global__ void matrixMulKernel(float *M, float *N, float *P, int width) {
  __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
  __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

  int bx = blockIdx.x, by = blockIdx.y;
  int tx = threadIdx.x, ty = threadIdx.y;

  // The Row and Width index of the element we try to calculate in P
  int Row = by * TILE_WIDTH + ty;
  int Col = bx * TILE_WIDTH + tx;

  float PValue = 0;
  for (int tile_idx = 0; tile_idx < width / TILE_WIDTH; ++tile_idx) {
    // For tile in M, there are Row rows before it,
    Mds[ty][tx] = M[Row * width + tile_idx * TILE_WIDTH + tx];
    Nds[ty][tx] = N[(tile_idx * TILE_WIDTH + ty) * width + Col];
    __syncthreads();

    for (int k = 0; k < TILE_WIDTH; ++k) {
      PValue += Mds[ty][k] * Nds[k][tx];
    }
    __syncthreads();
  }

  // Write final result to global memory AFTER all tiles are processed
  P[Row * width + Col] = PValue;
}

int main() {
  // Test matrix size (multiple of TILE_WIDTH)
  const int width = 1024;
  const size_t matrix_size = width * width * sizeof(float);

  // Allocate host memory
  float *h_M = (float *)malloc(matrix_size);
  float *h_N = (float *)malloc(matrix_size);
  float *h_P = (float *)malloc(matrix_size);

  // Initialize matrices
  printf("Initializing matrices...\n");
  for (int i = 0; i < width * width; i++) {
    h_M[i] = static_cast<float>(i % 100) / 10.0f; // Simple pattern
    h_N[i] = static_cast<float>((i + 50) % 100) / 10.0f;
    h_P[i] = 0.0f; // Initialize result to zero
  }

  // Allocate device memory
  float *d_M, *d_N, *d_P;
  cudaMalloc(&d_M, matrix_size);
  cudaMalloc(&d_N, matrix_size);
  cudaMalloc(&d_P, matrix_size);

  // Copy data to device
  cudaMemcpy(d_M, h_M, matrix_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_N, h_N, matrix_size, cudaMemcpyHostToDevice);

  // Setup kernel launch parameters
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
  dim3 dimGrid(width / TILE_WIDTH, width / TILE_WIDTH);

  printf("Launching kernel: %dx%d blocks, %dx%d threads per block\n", dimGrid.x,
         dimGrid.y, dimBlock.x, dimBlock.y);

  // Launch kernel
  matrixMulKernel<<<dimGrid, dimBlock, TILE_WIDTH * TILE_WIDTH>>>(d_M, d_N, d_P,
                                                                  width);

  // Check for kernel launch errors
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(error));
    return -1;
  }

  // Copy result back to host
  cudaMemcpy(h_P, d_P, matrix_size, cudaMemcpyDeviceToHost);

  // Verify a few results
  printf("\nVerifying results (checking first 5x5 block):\n");
  printf("Expected P[0][0] = %.2f, Got = %.2f\n",
         h_M[0] * h_N[0] + h_M[1] * h_N[width], h_P[0]);

  // Print small 5x5 section for verification
  printf("\nFirst 5x5 block of result:\n");
  for (int i = 0; i < 5 && i < width; i++) {
    for (int j = 0; j < 5 && j < width; j++) {
      printf("%6.2f ", h_P[i * width + j]);
    }
    printf("\n");
  }

  // Free memory
  free(h_M);
  free(h_N);
  free(h_P);
  cudaFree(d_M);
  cudaFree(d_N);
  cudaFree(d_P);

  printf("\nTest completed successfully!\n");
  return 0;
}