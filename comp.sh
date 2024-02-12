nvcc -O3 -lmpi -lcufft -L$EBROOTCUDA/lib64   scft.cu -o scft_gpu


