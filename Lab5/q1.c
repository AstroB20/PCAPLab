#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

// Program 1: Block size as N
__global__ void vectorAdd1(float *A, float *B, float *C, int n)
{
    int i = threadIdx.x;
    if (i < n)
    {
        C[i] = A[i] + B[i];
    }
}

// Program 2: N threads
__global__ void vectorAdd2(float *A, float *B, float *C, int n)
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

    // Program 1 execution: Block size as N
    vectorAdd1<<<1, N>>>(d_A, d_B, d_C, N);

    // Copy result back to host and print
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Program 1 result:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%.1f ", h_C[i]);
    }
    printf("\n");

    // Program 2 execution: N threads with multiple blocks
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd2<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result back to host and print
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("Program 2 result:\n");
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
