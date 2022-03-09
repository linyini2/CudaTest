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
const int TILE_DIM = 78;
const int grid_size_x = 240;
const int grid_size_y = grid_size_x;
const dim3 block_size(TILE_DIM, TILE_DIM);
const dim3 grid_size(grid_size_x, grid_size_y);
const long long N = grid_size_x * TILE_DIM;
const long long M = sizeof(real) * N * N;


__global__ void transpose1(const real *A, real *B, const int N);
void print_matrix(const int N, const real *A);

int main(void)
{
    real *h_A = (real *) malloc(M);
    real *h_B = (real *) malloc(M);
    for (int n = 0; n < N * N; ++n)
    {
        h_A[n] = n;
    }
    real *d_A, *d_B;
    CHECK(cudaMalloc(&d_A, M));
    CHECK(cudaMalloc(&d_B, M));
    CHECK(cudaMemcpy(d_A, h_A, M, cudaMemcpyHostToDevice));

    transpose1<<<grid_size,block_size>>>(d_A, d_B, N);
    CHECK(cudaMemcpy(h_B, d_B, M, cudaMemcpyDeviceToHost));

    cout << "number of threads: " << grid_size_x * grid_size_y * TILE_DIM * TILE_DIM << endl;
    cout << "total shared memory: " << (double)TILE_DIM * TILE_DIM * sizeof(real) * grid_size_x * grid_size_y / (1024 * 1024 * 1024) << "G" << endl;
    cout << "shared memory per block: " << (double)TILE_DIM * TILE_DIM * sizeof(real) / 1024 << "K" << endl;
    
    // printf("A =\n");
    // print_matrix(N, h_A);
    // printf("\nB =\n");
    // print_matrix(N, h_B);
    
    free(h_A);
    free(h_B);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    return 0;
}

__global__ void transpose1(const real *A, real *B, const int N)
{
    __shared__ real S[TILE_DIM][TILE_DIM];
    int bx = blockIdx.x * TILE_DIM;
    int by = blockIdx.y * TILE_DIM;

    int nx1 = bx + threadIdx.x;
    int ny1 = by + threadIdx.y;
    if (nx1 < N && ny1 < N)
    {
        S[threadIdx.y][threadIdx.x] = A[ny1 * N + nx1];
    }
    __syncthreads();

    int nx2 = bx + threadIdx.y;
    int ny2 = by + threadIdx.x;
    if (nx2 < N && ny2 < N)
    {
        B[nx2 * N + ny2] = S[threadIdx.x][threadIdx.y];
    }
}

void print_matrix(const int N, const real *A)
{
    for (int ny = 0; ny < N; ny++)
    {
        for (int nx = 0; nx < N; nx++)
        {
            cout << A[ny * N + nx] << " ";
        }
        printf("\n");
    }
}
