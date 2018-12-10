#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define BLOCK_SIZE 16

__global__ void Muld(float*, float*, int, int, float*);

void Mul(const float* A, const float* B, int hA, int wA, int wB, float* C) {
    int size;
    
    float* Ad;
    size = hA * wA * sizeof(float);
    cudaMalloc((void**)&Ad, size);
    cudaMemcpy(Ad, A, size, cudaMemcpyHostToDevice);
    float* Bd;
    size = wA * wB * sizeof(float);
    cudaMalloc((void**)&Bd, size);
    cudaMemcpy(Bd, B, size, cudaMemcpyHostToDevice);
    
    float* Cd;
    size = hA * wB * sizeof(float);
    cudaMalloc((void**)&Cd, size);
    
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(wB / dimBlock.x, hA / dimBlock.y);
    
    Muld<<<dimGrid, dimBlock>>>(Ad, Bd, wA, wB, Cd);
    
    cudaMemcpy(C, Cd, size, cudaMemcpyDeviceToHost);
    
    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
}


__global__ void Muld(float* A, float* B, int wA, int wB, float* C)
{
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
   
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int aBegin = wA * BLOCK_SIZE * by;
  
    int aEnd   = aBegin + wA - 1;
   
    int aStep  = BLOCK_SIZE;

    int bBegin = BLOCK_SIZE * bx;
 
    int bStep  = BLOCK_SIZE * wB;

    float Csub = 0;
 
    for (int a = aBegin, b = bBegin;
             a <= aEnd;
             a += aStep, b += bStep) {
      
        __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];

        __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];
     
        As[ty][tx] = A[a + wA * ty + tx];
        Bs[ty][tx] = B[b + wB * ty + tx];
     
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k)
           Csub += As[ty][k] * Bs[k][tx];
  
        __syncthreads();
    }

    int c = wB * BLOCK_SIZE * by + BLOCK_SIZE * bx;
    C[c + wB * ty + tx] = Csub;
}