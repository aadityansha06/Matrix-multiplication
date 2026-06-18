#define _POSIX_C_SOURCE 199309L // Required for CLOCK_MONOTONIC to measure precise timer without
          // moving backward
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define Total_SIZE 1024 * 1024

void multiply(int n, const double *restrict A, const double *restrict B,
              double *restrict C) {

  #pragma omp parallel for // from 1.2 sec to 0.607 sec
  for (int i = 0; i < n; i++) {
    for (int k = 0; k < n; k++) {
      for (int j = 0; j < n; j++) {
        C[i * n + j] += A[i * n + k] * B[k * n + j];
      }
    }
  }
}

int main() {

  struct timespec start, end;

  int n = 1024;
  size_t size = 1024 * 1024 * sizeof(double);
  double *A = (double *)malloc(size); // 64-bit or 8 byte float
  double *B = (double *)malloc(size);
  double *C =
      (double *)calloc(1024 * 1024, sizeof(double)); // auto initialize to 0
  for (int i = 0; i < Total_SIZE; i++) {
    A[i] = 2.0;
    B[i] = 3.0;
  }
    printf("\n\tNaive Matrix multiplication with max compiler optimization,Loop re-order and distributed across multiple core\n");

  printf("\nStarting benchmark for %d x %d matrix...\n", n, n);
  clock_gettime(CLOCK_MONOTONIC, &start);
  multiply(n, A, B, C);
  clock_gettime(CLOCK_MONOTONIC, &end);

  double time_taken =
      (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;

  printf("\nFunction took %f seconds to execute.\n", time_taken);

  // Calcuating GFLOPS for i3-2nd gen machine with theoratical DP GFLOPS as 52.8
    double operations = 2.0 * (double)n * (double)n * (double)n;

  double achived_gflpos = operations/ (time_taken * 1e9);
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
