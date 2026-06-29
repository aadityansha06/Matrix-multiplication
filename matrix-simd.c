/*
      Final matrix with AVX-256 register

      compile ->  gcc -O3 -march=native -fopenmp matrix-simd.c -o matrix-simd


      run ->sudo perf stat -e
cycles,instructions,cache-references,cache-misses,L1-dcache-loads,L1-dcache
-load-misses ./matrix-block

*/

#define _POSIX_C_SOURCE 199309L
// Required for CLOCK_MONOTONIC to measure precise timer without
// moving backward
#include <immintrin.h>
#include <omp.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define Total_SIZE 1032 * 1024
#define BLOCK_SIZE 64

static inline void multiply(int n, int stride, const double *restrict a,
                            const double *restrict b, double *restrict c) {
  // explocitly telling compiler that data is alligned  64-byte boundary
  double *A = (double *)__builtin_assume_aligned(a, 64);
  double *B = (double *)__builtin_assume_aligned(b, 64);
  double *C = (double *)__builtin_assume_aligned(c, 64);

#pragma omp parallel for collapse(2) // from 1.2 sec to 0.607 sec
  for (int i = 0; i < n; i += BLOCK_SIZE) {
    for (int j = 0; j < n; j += BLOCK_SIZE) {
      for (int k = 0; k < n; k += BLOCK_SIZE) {

        // 1. Step ii by 8 (Processing 8 rows of C at once)
        for (int ii = i; ii < BLOCK_SIZE + i; ii += 8) {

          // 2. Step jj by 4 (Processing 4 columns of C at once)
          for (int jj = j; jj < BLOCK_SIZE + j; jj += 4) {

       
            __m256d c0 = _mm256_setzero_pd();
            __m256d c1 = _mm256_setzero_pd();
            __m256d c2 = _mm256_setzero_pd();
            __m256d c3 = _mm256_setzero_pd();
            __m256d c4 = _mm256_setzero_pd();
            __m256d c5 = _mm256_setzero_pd();
            __m256d c6 = _mm256_setzero_pd();
            __m256d c7 = _mm256_setzero_pd();

        
            for (int kk = k; kk < BLOCK_SIZE + k; kk++) { 

              // Load 4 contiguous elements from Matrix B 
              __m256d b_vec = _mm256_load_pd(&B[kk * stride + jj]);

              // Row 0 to 7: Clone A for partial compute
              __m256d a_vec0 = _mm256_set1_pd(A[(ii + 0) * stride + kk]);
              c0 = _mm256_add_pd(c0, _mm256_mul_pd(a_vec0, b_vec));

              __m256d a_vec1 = _mm256_set1_pd(A[(ii + 1) * stride + kk]);
              c1 = _mm256_add_pd(c1, _mm256_mul_pd(a_vec1, b_vec));

              __m256d a_vec2 = _mm256_set1_pd(A[(ii + 2) * stride + kk]);
              c2 = _mm256_add_pd(c2, _mm256_mul_pd(a_vec2, b_vec));

              __m256d a_vec3 = _mm256_set1_pd(A[(ii + 3) * stride + kk]);
              c3 = _mm256_add_pd(c3, _mm256_mul_pd(a_vec3, b_vec));

              __m256d a_vec4 = _mm256_set1_pd(A[(ii + 4) * stride + kk]);
              c4 = _mm256_add_pd(c4, _mm256_mul_pd(a_vec4, b_vec));

              __m256d a_vec5 = _mm256_set1_pd(A[(ii + 5) * stride + kk]);
              c5 = _mm256_add_pd(c5, _mm256_mul_pd(a_vec5, b_vec));

              __m256d a_vec6 = _mm256_set1_pd(A[(ii + 6) * stride + kk]);
              c6 = _mm256_add_pd(c6, _mm256_mul_pd(a_vec6, b_vec));

              __m256d a_vec7 = _mm256_set1_pd(A[(ii + 7) * stride + kk]);
              c7 = _mm256_add_pd(c7, _mm256_mul_pd(a_vec7, b_vec));
            }

            // 4. Store all 8 rows back to RAM
            _mm256_store_pd(
                &C[(ii + 0) * stride + jj],
                _mm256_add_pd(_mm256_load_pd(&C[(ii + 0) * stride + jj]), c0));
            _mm256_store_pd(
                &C[(ii + 1) * stride + jj],
                _mm256_add_pd(_mm256_load_pd(&C[(ii + 1) * stride + jj]), c1));
            _mm256_store_pd(
                &C[(ii + 2) * stride + jj],
                _mm256_add_pd(_mm256_load_pd(&C[(ii + 2) * stride + jj]), c2));
            _mm256_store_pd(
                &C[(ii + 3) * stride + jj],
                _mm256_add_pd(_mm256_load_pd(&C[(ii + 3) * stride + jj]), c3));
            _mm256_store_pd(
                &C[(ii + 4) * stride + jj],
                _mm256_add_pd(_mm256_load_pd(&C[(ii + 4) * stride + jj]), c4));
            _mm256_store_pd(
                &C[(ii + 5) * stride + jj],
                _mm256_add_pd(_mm256_load_pd(&C[(ii + 5) * stride + jj]), c5));
            _mm256_store_pd(
                &C[(ii + 6) * stride + jj],
                _mm256_add_pd(_mm256_load_pd(&C[(ii + 6) * stride + jj]), c6));
            _mm256_store_pd(
                &C[(ii + 7) * stride + jj],
                _mm256_add_pd(_mm256_load_pd(&C[(ii + 7) * stride + jj]), c7));
          }
        }
      }
    }
  }
}

int main() {

  struct timespec start, end;

  int n = 1024;
  int stride = 1032; //  padding offset
  size_t size = stride * n * sizeof(double);
  // 32-byte alignment is perfect for AVX 256-bit registers
  double *A = (double *)aligned_alloc(64, size);
  double *B = (double *)aligned_alloc(64, size);
  double *C = (double *)aligned_alloc(64, size);

  for (int i = 0; i < Total_SIZE; i++) {
    C[i] = 0; // auto initialize to 0
  }
  for (int i = 0; i < Total_SIZE; i++) {
    A[i] = 2.0;
    B[i] = 3.0;
  }
  printf("\nMatrix multiplication with AVX register 256-bit using 8 register for output with each having 4 value  \n");

  printf("\nStarting benchmark for %d x %d matrix...\n", n, n);
  clock_gettime(CLOCK_MONOTONIC, &start);
  multiply(n, stride, A, B, C);
  clock_gettime(CLOCK_MONOTONIC, &end);

  double time_taken =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("\nFunction took %f seconds to execute.\n", time_taken);

  // Calcuating GFLOPS for i3-2nd gen machine with theoratical DP GFLOPS as 52.8
  double operations = 2.0 * (double)n * (double)n * (double)n;

  double achived_gflpos =
      operations / (time_taken * 1e9); // 1e reprsents 10, thus e9 => 10^9

  double theoretical_gflops = 52.8;

  double efficiency = (achived_gflpos / theoretical_gflops) * 100.0;

  printf(
      "\n Achived GFLOPS: %f\n Theoretical DP GFLOPS: %f\n Efficiency %f%% \n ",
      achived_gflpos, theoretical_gflops, efficiency);
  printf("Verification: C[0] = %f\n\n", C[0]);
  free(A);
  free(B);
  free(C);
}
