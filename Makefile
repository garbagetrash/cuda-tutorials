SRC=src
BIN=bin

add:
	g++ $(SRC)/add.cpp -o $(BIN)/add

add_cuda:
	nvcc $(SRC)/add.cu -o $(BIN)/add_cuda

add_block:
	nvcc $(SRC)/add_block.cu -o $(BIN)/add_block

add_grid:
	nvcc $(SRC)/add_grid.cu -o $(BIN)/add_grid

stream_legacy:
	nvcc $(SRC)/stream_test.cu -o $(BIN)/stream_legacy

stream_per_thread:
	nvcc --default-stream per-thread $(SRC)/stream_test.cu -o $(BIN)/stream_per_thread

pthread_legacy:
	nvcc $(SRC)/pthread_test.cu -o $(BIN)/pthreads_legacy

pthread_per_thread:
	nvcc --default-stream per-thread $(SRC)/pthread_test.cu -o $(BIN)/pthreads_per_thread

fft_single:
	nvcc $(SRC)/fft_single.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcufft -o $(BIN)/fft_single

fft_batch:
	nvcc $(SRC)/fft_batch.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcufft -o $(BIN)/fft_batch

async_mem:
	nvcc $(SRC)/async_mem.cu -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcufft -o $(BIN)/async_mem

.PHONY: clean
clean:
	rm $(BIN)/*
