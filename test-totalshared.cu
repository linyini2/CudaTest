#include "error.cuh"
#include <stdio.h>
#include <iostream>
using namespace std;

#ifdef USE_DP
    typedef float real;
#else
    typedef long long real;
#endif

// #CASE 2:
// typedef long long real, BLOCK_SIZE = 78, GRID_SIZE = 400000
// number of threads: 31200000, total share memory: 18.1317G, share memory per block: 47.5312K
// ？？BLOCK_SIZE must be power of 2, or else the sum will be wrong

const int BLOCK_SIZE = 78;
const long long GRID_SIZE = 6000000;
const long long N = BLOCK_SIZE * GRID_SIZE;
const long long M = sizeof(real) * N;
void __global__ reduce_shared(real *d_x, real *d_y);


int main(void)
{
    real *h_x = (real *) malloc(M);
    for (long long n = 0; n < N; ++n)
    {
        h_x[n] = 1;
    }
    real *d_x;
    CHECK(cudaMalloc(&d_x, M));
    CHECK(cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice));

    const long long ymem = sizeof(real) * GRID_SIZE;
    real *d_y;
    CHECK(cudaMalloc(&d_y, ymem));
    real *h_y = (real *) malloc(ymem);

    reduce_shared<<<GRID_SIZE, BLOCK_SIZE>>>(d_x, d_y);

    CHECK(cudaMemcpy(h_y, d_y, ymem, cudaMemcpyDeviceToHost));
    cout << "number of threads: " << N << endl;
    cout << "total shared memory: " << (double)GRID_SIZE * BLOCK_SIZE * BLOCK_SIZE * sizeof(real) / (1024 * 1024 * 1024) << "G" << endl;
    cout << "shared memory per block: " << (double)BLOCK_SIZE * BLOCK_SIZE * sizeof(real) / 1024 << "K" << endl;

    

    real result = 0.0;
    for (long long n = 0; n < GRID_SIZE; ++n)
    {
        result += h_y[n];
    }
    cout << result << endl;
    // printf("%.2f", result);
    
    free(h_x);
    CHECK(cudaFree(d_x));
    free(h_y);
    CHECK(cudaFree(d_y));
    return 0;
}


void __global__ reduce_shared(real *d_x, real *d_y)
{
    const int tid = threadIdx.x;
    const int bid = blockIdx.x;
    const long long n = bid * blockDim.x + tid;
    __shared__ real s_y[BLOCK_SIZE][BLOCK_SIZE];
    for (int i = 0; i < BLOCK_SIZE; i++)
    {
        s_y[tid][i] = (n < N) ? d_x[n] : 0.0;
    }
    
    // s_y[tid][0] = (n < N) ? d_x[n] : 0.0;
    
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1)
    {

        if (tid < offset)
        {
            s_y[tid][0] += s_y[tid + offset][0];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_y[bid] = s_y[0][0];
    }
}
