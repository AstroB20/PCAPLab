#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 1024
#define THREADS_PER_BLOCK 256

__global__ void computeSine(float *angles, float *sineValues, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        sineValues[i] = sinf(angles[i]);
    }
}

int main()
{
    float *h_angles, *h_sineValues; // Host arrays
    float *d_angles, *d_sineValues; // Device arrays

    size_t size = N * sizeof(float);

    // Allocate memory on host
    h_angles = (float *)malloc(size);
    h_sineValues = (float *)malloc(size);

    // Initialize angles array on host
    for (int i = 0; i < N; i++)
    {
        h_angles[i] = i * 3.14159f / N; // Angles from 0 to pi
    }

    // Allocate memory on device
    cudaMalloc((void **)&d_angles, size);
    cudaMalloc((void **)&d_sineValues, size);

    // Copy angles array from host to device
    cudaMemcpy(d_angles, h_angles, size, cudaMemcpyHostToDevice);

    // Calculate number of blocks
    int blocksPerGrid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Launch kernel
    computeSine<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_angles, d_sineValues, N);

    // Copy result back to host
    cudaMemcpy(h_sineValues, d_sineValues, size, cudaMemcpyDeviceToHost);

    // Print result
    printf("Sine values:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("%.4f ", h_sineValues[i]);
    }
    printf("\n");

    // Free device memory
    cudaFree(d_angles);
    cudaFree(d_sineValues);

    // Free host memory
    free(h_angles);
    free(h_sineValues);

    return 0;
}
