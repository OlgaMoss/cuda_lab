#include <stdio.h>
typedef struct {
  int width;
  int height;
  float* elements;
  int stride;
} Matrix;


#define BLOCK_SIZE 16
__global__ void MatMulKernel(const Matrix, const Matrix, Matrix);