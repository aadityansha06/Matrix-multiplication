/*
      OpenBLAS Matrix Multiplication Benchmark
      
      Compile -> gcc -O3 matrix-openblas.c -o matrix-openblas -lopenblas
      Run     -> ./matrix-openblas
*/

#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cblas.h> // The OpenBLAS Header

int main() {
    struct timespec start, end;
    
    int n = 1024;
    size_t size = n * n * sizeof(double);
    
    // OpenBLAS handles its own internal memory packing, 
    // but we use aligned_alloc to give it a clean starting point.
    double *A = (double *)aligned_alloc(64, size);
    double *B = (double *)aligned_alloc(64, size);
    double *C = (double *)aligned_alloc(64, size);
    
    if (!A || !B || !C) {
        printf("Memory allocation failed!\n");
        return 1;
    }

    // Standard 2D initialization (No stride padding needed for OpenBLAS)
    for (int i = 0; i < n * n; i++) {
        A[i] = 2.0;
        B[i] = 3.0;
        C[i] = 0.0;
    }
    
    printf("\nStarting OpenBLAS benchmark for %d x %d matrix...\n", n, n);
    
    clock_gettime(CLOCK_MONOTONIC, &start);
    
    /* The core OpenBLAS call (Double Precision GEneral Matrix Multiplication)
       Formula: C = alpha * A * B + beta * C
    */
    cblas_dgemm(
        CblasRowMajor, // Our matrices are stored row-by-row in C
        CblasNoTrans,  // Do not transpose A
        CblasNoTrans,  // Do not transpose B
        n,             // Rows of A and C
        n,             // Columns of B and C
        n,             // Columns of A, Rows of B
        1.0,           // Alpha multiplier (1.0)
        A,             // Matrix A
        n,             // Leading dimension of A
        B,             // Matrix B
        n,             // Leading dimension of B
        0.0,           // Beta multiplier (0.0 to overwrite C)
        C,             // Matrix C
        n              // Leading dimension of C
    );
    
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double time_taken = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    
    printf("\nFunction took %f seconds to execute.\n", time_taken);
    
    // Calculating GFLOPS for your specific hardware
    double operations = 2.0 * (double)n * (double)n * (double)n;
    double achieved_gflops = operations / (time_taken * 1e9);
    
    // Your theoretical ceiling
    double theoretical_gflops = 52.8; 
    double efficiency = (achieved_gflops / theoretical_gflops) * 100.0;
    
    printf("\n Achieved GFLOPS: %f\n Theoretical DP GFLOPS: %f\n Efficiency %f%% \n",
           achieved_gflops, theoretical_gflops, efficiency);
           
    printf("Verification: C[0] = %f\n\n", C[0]);
    
    free(A);
    free(B);
    free(C);
    
    return 0;
}
