#include <ctime>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

void vecMulHost(int *A, int *B, int *C, int n) {
  for (int i = 0; i < n; ++i) {
    C[i] = A[i] * B[i];
  }
}

__global__ void vecMulKernel(int *A, int *B, int *C, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    C[i] = A[i] * B[i];
  }
}

void vecMulDevice(int *A_h, int *B_h, int *C_h, int n) {
  int *A_d, *B_d, *C_d;

  cudaMalloc((void **)&A_d, n * sizeof(int));
  cudaMalloc((void **)&B_d, n * sizeof(int));
  cudaMalloc((void **)&C_d, n * sizeof(int));

  cudaMemcpy(A_d, A_h, n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, n * sizeof(int), cudaMemcpyHostToDevice);

  dim3 threadDim(128, 1, 1);
  dim3 gridDim(ceil(n / float(128)), 1, 1);

  vecMulKernel<<<gridDim, threadDim>>>(A_d, B_d, C_d, n);

  cudaMemcpy(C_d, C_d, n * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(A_d);
  cudaFree(B_d);
  cudaFree(C_d);
}

int main() {
  int n = 1234567890;
  int *A, *B, *C;

  A = (int *)malloc(n * sizeof(int));
  B = (int *)malloc(n * sizeof(int));
  C = (int *)malloc(n * sizeof(int));

  for (int i = 0; i < n; ++i) {
    A[i] = i;
    B[i] = i;
  }

  clock_t start_host = clock();
  vecMulHost(A, B, C, n);
  clock_t end_host = clock();
  double time_host = (double)(end_host - start_host) / CLOCKS_PER_SEC;

  clock_t start_device = clock();
  vecMulDevice(A, B, C, n);
  clock_t end_device = clock();
  double time_device = (double)(end_device - start_device) / CLOCKS_PER_SEC;

  printf("Time taken for vecMulHost: %f seconds\n", time_host);
  printf("Time taken for vecMulDevice: %f seconds\n", time_device);
  return 0;
}