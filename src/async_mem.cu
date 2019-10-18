#include <stdio.h>
#include <cufft.h>


typedef struct {
  int nfft;
  cudaStream_t stream;
  cufftHandle plan;
  float2 *d_in;
  float2 *d_out;
  float2 *h_in;
  float2 *h_out;
} fft_stream_struct;

void fft_stream_setup(fft_stream_struct *fss, int nfft) {
  (*fss).nfft = nfft;
  cudaStreamCreate(&(*fss).stream);
  cufftPlan1d(&(*fss).plan, nfft, CUFFT_C2C, 1);
  cufftSetStream((*fss).plan, (*fss).stream);
  cudaMalloc((void**)&(*fss).d_in, nfft * sizeof(float2));
  cudaMalloc((void**)&(*fss).d_out, nfft * sizeof(float2));
  cudaHostAlloc((void**)&(*fss).h_in, sizeof(float2) * nfft, cudaHostAllocPortable);
  cudaHostAlloc((void**)&(*fss).h_out, sizeof(float2) * nfft, cudaHostAllocPortable);
}

void fft_stream_destroy(fft_stream_struct *fss) {
  cudaStreamDestroy((*fss).stream);
  cufftDestroy((*fss).plan);
  cudaFree((*fss).d_in);
  cudaFree((*fss).d_out);
  cudaHostUnregister((*fss).h_in);
  cudaHostUnregister((*fss).h_out);
  cudaFreeHost((*fss).h_in);
  cudaFreeHost((*fss).h_out);
}

void fft_stream_push(fft_stream_struct *fss) {
  int nfft = (*fss).nfft;
  cudaMemcpyAsync((*fss).d_in, (*fss).h_in, nfft * sizeof(float2), cudaMemcpyHostToDevice, (*fss).stream);
  cufftExecC2C((*fss).plan, (cufftComplex*)(*fss).d_in, (cufftComplex*)(*fss).d_out, CUFFT_FORWARD);
  cudaMemcpyAsync((*fss).h_out, (*fss).d_in, nfft * sizeof(float2), cudaMemcpyDeviceToHost, (*fss).stream);
}

int main() {
    const int NUM_STREAMS = 3;
    const int NFFT = 1 << 25;

    // Creates fft streams
    fft_stream_struct streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
      fft_stream_setup(&streams[i], NFFT);
    }
    printf("Finished FFT stream setup\n");
    fflush(stdout);

    // Host input data initialization
    for (int j = 0; j < NUM_STREAMS; j++) {
      for (int i = 0; i < NFFT; i++) {
          streams[j].h_in[i].x = 1.f;
          streams[j].h_in[i].y = 0.f;
      }
    }
    printf("Finished populating input data\n");
    fflush(stdout);

    // Async memcopies and computations
    for (int i = 0; i < 3 * NUM_STREAMS; i++) {
      fft_stream_push(&streams[i % NUM_STREAMS]);
    }
    printf("Finished FFTs\n");
    fflush(stdout);

    for (int i = 0; i < NUM_STREAMS; i++) {
      cudaStreamSynchronize(streams[i].stream);
    }
    printf("Finished synchronizing\n");
    fflush(stdout);

    cudaDeviceSynchronize();

    for (int i = 0; i < NUM_STREAMS; i++) {
      fft_stream_destroy(&streams[i]);
    }
    printf("Finished cleaning up\n");
    fflush(stdout);

    cudaDeviceReset();

    return 0;
}
