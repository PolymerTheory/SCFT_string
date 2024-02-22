# Makefile for compiling CUDA code

CC=nvcc
CFLAGS=-O3
LIBS=-lmpi -lcufft
LDFLAGS=-L$(EBROOTCUDA)/lib64
TARGET=scft_gpu
SOURCE=scft.cu

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CC) $(CFLAGS) $(LDFLAGS) $(LIBS) -o $(TARGET) $(SOURCE)

clean:
	rm -f $(TARGET)
