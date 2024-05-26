#define NX 1048576
#define BATCH 10
#define RANK 1

#include <stdio.h>
#include <cufft.h>


int main() {

  cufftHandle plan;
  cufftComplex *data;
  cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*BATCH);
  if (cudaGetLastError() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to allocate\n");
    return -1;
  }

  int dims[] = { NX };
  if (cufftPlanMany(&plan, RANK, dims, NULL, 0, 0,
              NULL, 0, 0, CUFFT_C2C, BATCH) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: Plan creation failed");
    return -1;
  }

  /* Note:
   *  Identical pointers to input and output arrays implies in-place transformation
   */

  if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecC2C Forward failed");
    return -1;
  }

  if (cufftExecC2C(plan, data, data, CUFFT_INVERSE) != CUFFT_SUCCESS){
    fprintf(stderr, "CUFFT error: ExecC2C Inverse failed");
    return -1;
  }

  /*
   *  Results may not be immediately available so block device until all
   *  tasks have completed
   */

  if (cudaDeviceSynchronize() != cudaSuccess){
    fprintf(stderr, "Cuda error: Failed to synchronize\n");
    return -1;
  }

  /*
   *  Divide by number of elements in data set to get back original data
   */

  cufftDestroy(plan);
  cudaFree(data);

  return 0;
}
