#include <cuda_runtime.h>
#include <bits/stdc++.h>

using namespace std;
using namespace std::chrono;

// a function that runs on the GPU is declared with __global__
__global__ void add(int *A, int *B, int *C, int size) {
    // calculate the global thread ID "id"
    // each GPU thread computes one element of the sum if "id" is within bounds.
    int id = blockDim.x * blockIdx.x + threadIdx.x;  
    if (id < size) {
        C[id] = A[id] + B[id];
    }
}

// optional: sequential vector additionv
void seqAdd(int *A, int *B, int *C, int size) {
    for (int i=0; i<size; i++) {
        C[i] = A[i] + B[i];
    }
}

// fills vector with random integers (0 to 9).
void initialize(int *vector, int size) {
    for (int i=0; i<size; i++) {
        vector[i] = rand() % 10;
    }
}

void print(int *vector, int size) {
    for (int i=0; i<size; i++) {
        cout<<vector[i]<<" ";
    }
    cout<<endl;
}


int main() {
        
    int vectorSize = 4;
    size_t vectorBytes = vectorSize * sizeof(int);

    int *A, *B, *C;
    A = new int[vectorSize];
    B = new int[vectorSize];
    C = new int[vectorSize];

    initialize(A, vectorSize);
    initialize(B, vectorSize);

    cout<<"\nVector A: ";
    print(A, vectorSize);
    cout<<"\nVector B: ";
    print(B, vectorSize);

    // allocate memory on GPU (device) for vectors X, Y, and Z
    int *X, *Y, *Z;
    cudaMalloc(&X, vectorBytes); 
    cudaMalloc(&Y, vectorBytes);
    cudaMalloc(&Z, vectorBytes);

    // copy host data (A, B) to device memory (X, Y)
    cudaMemcpy(X, A, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(Y, B, vectorBytes, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (vectorSize + threadsPerBlock - 1) / threadsPerBlock;

    // launch gpuAdd() on GPU
    // each thread adds one pair of elements from X & Y
    auto start = high_resolution_clock::now();
    add<<<blocksPerGrid, threadsPerBlock>>>(X, Y, Z, vectorSize);
    cudaMemcpy(C, Z, vectorBytes, cudaMemcpyDeviceToHost);  // transfer result vector Z from GPU to CPU (C)
    auto stop = high_resolution_clock::now();
    
    auto duration = duration_cast<microseconds>(stop-start);

    cout<<"\nAddition: ";
    print(C, vectorSize);
    cout<<"\nTime: "<<duration.count()<<" microseconds"<<endl;

    // free all memory on host and device
    free(A); free(B); free(C);
    cudaFree(X); cudaFree(Y); cudaFree(Z);

    return 0;
}