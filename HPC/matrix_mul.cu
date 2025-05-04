#include <cuda_runtime.h>
#include <bits/stdc++.h>

using namespace std;
using namespace std::chrono;

__global__ void multiply(int *A, int *B, int *C, int size) {
    // blockIdx = which block
    // threadIdx = which thread within the block
    // blockDim = size of the block
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < size && col < size) {
        int sum = 0;
        for (int i=0; i<size; i++) {
            sum += A[row * size + i] * B[i * size + col];  // dot product
        }
        C[row * size + col] = sum;
    }
}


// optional: sequential matrix multiplication
void seqMultiply(int *A, int *B, int *C, int size) {
    // initialize result matrix to 0
    for (int i = 0; i <size*size; i++) {
        C[i] = 0;
    }

    for (int row=0; row<size; row++) {
        for (int col=0; col<size; col++) {
            int sum = 0;
            for (int k=0; k<size; k++) {
                sum += A[row * size + k] * B[k * size + col];
            }
            C[row * size + col] = sum;
        }
    }
}


void initialize(int *matrix, int size) {
    for (int i=0; i<size*size; i++) {
        matrix[i] = rand() % 10;
    }
}


void print(int *matrix, int size) {
    for (int row=0; row<size; row++) {
        for (int col=0; col<size; col++) {
            cout<<matrix[row*size + col]<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
}


int main() {

    int N = 3;
    int matrixSize = N * N;
    size_t matrixBytes = matrixSize * sizeof(int);

    int *A, *B, *C;
    A = new int[matrixSize];
    B = new int[matrixSize];
    C = new int[matrixSize];

    initialize(A, N);
    initialize(B, N);

    cout<<"Matrix A: \n";
    print(A, N);
    cout<<"Matrix B: \n";
    print(B, N);

    int *X, *Y, *Z;
    cudaMalloc(&X, matrixBytes);
    cudaMalloc(&Y, matrixBytes);
    cudaMalloc(&Z, matrixBytes);

    cudaMemcpy(X, A, matrixBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B, matrixBytes, cudaMemcpyHostToDevice);

    // threads per CTA dimension
    int THREADS = 3;

    // blocks per grid dimension (assumes N is divisible by THREADS)
    int BLOCKS = N / THREADS;

    dim3 threads(THREADS, THREADS);  // each block has 9 threads arranged in a 3×3 grid
    dim3 blocks(BLOCKS, BLOCKS);  // only 1 block is launched

    // launch kernel
    auto start = high_resolution_clock::now();
    multiply<<<blocks, threads>>>(X, Y, Z, N);
    cudaMemcpy(C, Z, matrixBytes, cudaMemcpyDeviceToHost);
    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop-start);

    cout<<"Multiplication A x B: \n";
    print(C, N);
    cout<<"Time: "<<duration.count()<<" microseconds"<<endl;

    delete[] A; delete[] B; delete[] C;
    cudaFree(X); cudaFree(Y); cudaFree(Z);

    return 0;
}