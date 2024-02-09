#include <cstdio>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <cuda.h>
 
#pragma once
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


//the function of most (if not all) of these kernels can be obrained from their usage (if it's not clear from their name). Thus, see density.h.

int const threadsPerBlock = 1024;// WARNING: threadsPerBlock MUST = 2^n, WHERE n IS AN INTEGER (DUE TO REDUCTION KERNELS)

void HandleError(cudaError_t err, char const * const file, int const line)
{
    if (err != cudaSuccess) {
        fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}

void HandleError(cufftResult err, char const * const file, int const line)
{
    if (err != CUFFT_SUCCESS) {
        switch(err) {
            case CUFFT_INVALID_PLAN:
                printf ("cufft %s in %s at line %d\n", "CUFFT_INVALID_PLAN", file, line);
                break;
            case CUFFT_INVALID_VALUE:
                printf ("cufft %s in %s at line %d\n", "CUFFT_INVALID_VALUE", file, line);
                break;
            case CUFFT_INTERNAL_ERROR:
                printf ("cufft %s in %s at line %d\n", "CUFFT_INTERNAL_ERROR", file, line);
                break;
            case CUFFT_EXEC_FAILED:
                printf ("cufft %s in %s at line %d\n", "CUFFT_EXEC_FAILED", file, line);
                break;
            case CUFFT_SETUP_FAILED:
                printf ("cufft %s in %s at line %d\n", "CUFFT_EXEC_FAILEDCUFFT_SETUP_FAILED", file, line);
                break;
            default:
                printf ("cufft %s in %s at line %d\n", "CUFFT_EXEC_FAILEDCUFFT_SETUP_FAILED", file, line);
        }
        exit(EXIT_FAILURE);
    }
}

/// ^stuff to handle errors^


//print hello (check if things are working with gpu)
__global__ void print_from_gpu(void) {
    printf("Testing GPUs: [%d,%d] (From device)\n", threadIdx.x,blockIdx.x);
}


__global__ void reduction_sum(double *a, double *c, const int _M) {
    __shared__ double cache[threadsPerBlock];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int cacheIndex = threadIdx.x;

    double temp = 0;

    while (tid < _M) {
        temp += a[tid];
        tid += blockDim.x * gridDim.x;
    }

    // set the cache values //
    cache[cacheIndex] = temp;

    // synchronise threads in this block //
    __syncthreads();

    // for reductions, threadsPerBlock must be a power of 2 //
    int i = blockDim.x / 2;
    while (i != 0) {
        if (cacheIndex < i) {
            cache[cacheIndex] += cache[cacheIndex + i];
        }
        __syncthreads();
        i /= 2;
    }

    // store final value of sum for current block //
    if (cacheIndex == 0) {
        c[blockIdx.x] = cache[0];
    }
}

__global__ void rA_rB_init_Ns_homo(double *dev_rhA, double *dev_rhB, double *dev_rCA, double *dev_rCB, double *dev_q1_Nh, double *dev_q2_Nh, double *dev_q1_Ns, double *dev_q2_Ns, double *dev_q1_n1, double *dev_q2_n2, const int _M)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _M) return;
    dev_rhA[tid]  = 0;// no A homopolymers
    dev_rhB[tid]  = 2.0 * dev_q2_Nh[tid];
    dev_rCA[tid] = dev_q2_Ns[tid] + dev_q1_n1[tid]*dev_q2_n2[tid];
    dev_rCB[tid] = dev_q1_Ns[tid] + dev_q1_n1[tid]*dev_q2_n2[tid];
    
}

__global__ void sumconcs(double *dev_rA, double *dev_rB, double *dev_rCA, double *dev_rCB,  double *dev_rC, const int _M)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _M) return;
    
    dev_rA[tid]  = dev_rA[tid] + dev_rCA[tid];
    dev_rB[tid]  = dev_rB[tid] + dev_rCB[tid];
    dev_rC[tid]  = dev_rCA[tid] + dev_rCB[tid];

}

__global__ void sumconcs2(double *dev_rA, double *dev_rB, double *dev_rh, double *dev_rCB, double *dev_rhA, double *dev_rhB, double *dev_rP, double *dev_rM, double *dev_rCA, double *dev_rC, const int _M)

{ // some of this is unnecessary but it costs very little
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _M) return;
    
    
    dev_rA[tid] = dev_rhA[tid] + dev_rCA[tid];
    dev_rB[tid] = dev_rhB[tid] + dev_rCB[tid];
    dev_rh[tid] = dev_rhA[tid] + dev_rhB[tid];
    dev_rC[tid] = dev_rCA[tid] + dev_rCB[tid];
    dev_rP[tid] = dev_rA[tid] + dev_rB[tid] - 1.0;
    dev_rM[tid] = dev_rA[tid] - dev_rB[tid];
}

__global__ void scale_concentrations(double *dev_rA, double *dev_rB, double *dev_rCA, double *dev_rCB, const double _zA, const double _zB, const double _zC, const double _ds1, const double _ds2, const int _M) //Be careful - written as if in GC. Make inputs work as in C
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _M) return;
    
    dev_rA[tid]  = dev_rA[tid] * _zA*_ds1/3.0;
    dev_rB[tid]  = dev_rB[tid] * _zB*_ds2/3.0;
    
    dev_rCA[tid] = dev_rCA[tid]* _zC*_ds1/3.0;
    dev_rCB[tid] = dev_rCB[tid]* _zC*_ds2/3.0;
}


__global__ void set_W_density_sums_diffs0(double *dev_rA, double *dev_rB, double *dev_A, double *dev_B, const int _M)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _M) return;
    dev_A[tid] = dev_rA[tid];
    dev_B[tid] = dev_rB[tid];
    dev_rA[tid] = dev_A[tid] - dev_B[tid];
    dev_rB[tid] = dev_A[tid] + dev_B[tid] - 1.0;
}

__global__ void rA_rB_mid_segments(double *dev_rX, double *dev_q1_s, double *dev_q2_Nsms, const int _wt, const int _M)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _M) return;
    dev_rX[tid] += _wt * dev_q1_s[tid] * dev_q2_Nsms[tid];
}

__global__ void rA_rB_init_n1_Ns(double *dev_rA, double *dev_rB, double *dev_q1_n1, double *dev_q2_Nsmn1, double *dev_q1_0, double *dev_q2_Ns, double *dev_q2_0, double *dev_q1_Ns, const int _M)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _M) return;
    dev_rA[tid] = dev_q1_n1[tid] * dev_q2_Nsmn1[tid] + dev_q1_0[tid] * dev_q2_Ns[tid];
    dev_rB[tid] = dev_q1_n1[tid] * dev_q2_Nsmn1[tid] + dev_q2_0[tid] * dev_q1_Ns[tid];
}


__global__ void Array_set_value(double *a, const double b, int const _M)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _M) return;
    a[tid] = b;
}

__global__ void Mult_self(cufftDoubleComplex *a, const double *b, int const _M)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _M) return;
    a[tid].x *= b[tid];
    a[tid].y *= b[tid];
}


__global__ void Mult_self(double *a, const double *b, int const _M)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _M) return;
    a[tid] *= b[tid];
}


__global__ void Mult(double *a, const double *b, const double *c, const int _M) {
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _M) return;
    a[tid] = b[tid] * c[tid];
}


__global__ void Richardson(double *dev_out, const double *dev_qs0, const double *dev_qs1, const double *dev_expWds2, const double *dev_expWds, const int _M) {
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _M) return;
    dev_out[tid] = (dev_qs0[tid] * dev_expWds2[tid] * 4 - dev_qs1[tid] * dev_expWds[tid]) / 3;
}



__global__ void Prepare_dev_expKsq2_expKsq(double *dev_expKsq2, double *dev_expKsq, double *dev_k_sq, const double ds, const int _M, const int _Mk)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _Mk) return;
    dev_expKsq2[tid] = exp(-dev_k_sq[tid] * ds / 12) / _M;
    dev_expKsq[tid]  = dev_expKsq2[tid] * dev_expKsq2[tid] * _M;
}

__global__ void Prepare_dev_dev_expWds2_dev_expWds(double *dev_expWds2_p1, double *dev_expWds_p1, double *dev_expWds2_m1, double *dev_expWds_m1, const double *dev_W, const double ds, const int _M)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _M) return;
    
    dev_expWds2_p1[tid] = exp(-(dev_W[tid] + dev_W[tid + _M])*ds / 4);
    dev_expWds_p1[tid]  = dev_expWds2_p1[tid] * dev_expWds2_p1[tid];

    dev_expWds2_m1[tid] = exp(-(-dev_W[tid] + dev_W[tid + _M])*ds / 4);
    dev_expWds_m1[tid]  = dev_expWds2_m1[tid] * dev_expWds2_m1[tid];
}


//calculate coefficiennts for cubic spline
__global__ void cubefit_g(double *x, double *dev_cubes, double *dev_cubesT, const int _M)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _M) return;
    /** Step 0 */
    const int n= strn-1;
    int i, j;
    // Calculate pointers within shared memory for each array
    double* c=dev_cubes + (0*strn + strn*4*tid);
    double* b=dev_cubes + (1*strn + strn*4*tid);
    double* d=dev_cubes + (2*strn + strn*4*tid);
    double* a=dev_cubes + (3*strn + strn*4*tid);

    //this is probably nnot the best way to do it
    //these temporary arrays are never used outside of this kernel so it's slower and less memory-efficient that necessary
    //but my attempts to make it more efficient broke it and I gave up. If you fix it please let me know.
    double* h=dev_cubesT + (0*strn + strn*5*tid);
    double* A=dev_cubesT + (1*strn + strn*5*tid);
    double* l=dev_cubesT + (2*strn + strn*5*tid);
    double* u=dev_cubesT + (3*strn + strn*5*tid);
    double* z=dev_cubesT + (4*strn + strn*5*tid);
    ////
    
    // Step 1
    for (i = 0; i <= n - 1; ++i) h[i] = x[i + 1] - x[i];
  //  /*
    // Step 2
    for (i = 1; i <= n - 1; ++i)
        A[i] = 3 * (a[i + 1] - a[i]) / h[i] - 3 * (a[i] - a[i - 1]) / h[i - 1];

   //   Step 3
    l[0] = 1;
    u[0] = 0;
    z[0] = 0;

    // Step 4
    for (i = 1; i <= n - 1; ++i) {
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * u[i - 1];
        u[i] = h[i] / l[i];
        z[i] = (A[i] - h[i - 1] * z[i - 1]) / l[i];
    }
    // Step 5
    l[n] = 1;
    z[n] = 0;
    c[n] = 0;
// Step 6
    for (j = n - 1; j >= 0; --j) {
        c[j] = z[j] - u[j] * c[j + 1];
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3;
        d[j] = (c[j + 1] - c[j]) / (3 * h[j]);
    }
}
__global__ void transpose_cube(double *dev_W, double *dev_cubes, int i, const int _M, const int _strn)
{
    int const tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= _M) return;
    DEV_CUBES(tid,3,i)=dev_W[tid];
}


