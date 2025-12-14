#include <__clang_cuda_builtin_vars.h>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <driver_types.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void mat_vec_mul(int *matrix, int *vector, int *result, int width,
                            int height) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= height)
    return;
  int value = 0;
  for (int j = 0; j < width; ++j) {
    value += matrix[i * width + j] * vector[j];
  }
  result[i] = value;
}

// Matrix * Vector
int main(int argc, char *argv[]) {
  int width, height;
  int *h_matrix, *h_vector, *h_result, *d_matrix, *d_vector, *d_result;

  if (argc != 3) {
    printf("Usage: %s <matrix-width>, <matrix-height>\n", argv[0]);
    return 1;
  }

  width = atoi(argv[1]);
  height = atoi(argv[2]);

  h_matrix = (int *)malloc(width * height * sizeof(int));
  h_vector = (int *)malloc(width * sizeof(int));
  h_result = (int *)malloc(height * sizeof(int));

  cudaMalloc((void **)&d_matrix, width * height * sizeof(int));
  cudaMalloc((void **)&d_vector, width * sizeof(int));
  cudaMalloc((void **)&d_result, height * sizeof(int));

  srand(NULL);
  for (int i = 0; i < width * height; ++i) {
    h_matrix[i] = rand() % 256;
  }

  cudaMemcpy(d_matrix, h_matrix, width * height * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_vector, h_vector, width * sizeof(int), cudaMemcpyHostToDevice);

  dim3 dimBlock(128);
  dim3 dimGrid(ceil(height / float(128)));


  return 0;
}