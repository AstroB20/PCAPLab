#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define THREADS_PER_BLOCK 256

__global__ void vectorAdd(float *A, float *B, float *C, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    float *h_A, *h_B, *h_C; // Host vectors
    float *d_A, *d_B, *d_C; // Device vectors

    size_t size = N * sizeof(float);

    // Allocate memory on host
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);

    // Initialize vectors on host
    for (int i = 0; i < N; i++)
    {
        h_A[i] = i * 1.0f;
        h_B[i] = i * 2.0f;
    }

    // Allocate memory on device
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);

    // Copy vectors from host to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Calculate number of blocks
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_A, d_B, d_C, N);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Print result
    printf("Result:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%.1f ", h_C[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
