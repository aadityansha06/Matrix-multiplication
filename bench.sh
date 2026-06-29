#!/bin/bash

# Default to 10 iterations if the user doesn't provide an argument
ITERS=${1:-10}

echo -e "\n[+] Compiling binaries..."
gcc -O3 -march=native -fopenmp matrix-tiling.c -o matrix-tiling
gcc -O3 -march=native -fopenmp matrix-simd.c -o matrix-simd
gcc -O3 matrix-openblas.c -o matrix-openblas -lopenblas

# Verify compilation
if [[ ! -f matrix-tiling || ! -f matrix-simd || ! -f matrix-openblas ]]; then
    echo "[-] Error: Compilation failed. Check your GCC flags and OpenBLAS install."
    exit 1
fi

echo -e "[+] Compilation successful. Running benchmark for $ITERS iterations each.\n"

# Function to run an executable N times, extract data, and calculate the median
run_benchmark() {
    local exe=$1
    local iters=$2
    local times=()
    local gflops=()
    local effs=()
    
    echo -n "Profiling ./$exe " >&2
    
    for (( i=0; i<iters; i++ )); do
        # Run the binary and capture the output
        output=$(./$exe)
        
        # Extract the numbers using grep and awk
        # Regex 'Achie?ved' handles both spellings in your C files
        t=$(echo "$output" | grep "Function took" | awk '{print $3}')
        g=$(echo "$output" | grep -iE "Achie?ved GFLOPS:" | awk '{print $3}')
        e=$(echo "$output" | grep "Efficiency" | awk '{print $2}' | tr -d '%')
        
        times+=($t)
        gflops+=($g)
        effs+=($e)
        
        # Print a dot to show progress
        echo -n "." >&2
    done
    echo "" >&2
    
    # Bash doesn't do float math natively. We pipe the arrays into sort and awk to find the median.
    med_t=$(printf "%s\n" "${times[@]}" | sort -n | awk '{a[i++]=$1} END {if(i%2==0) print (a[i/2-1]+a[i/2])/2; else print a[int(i/2)]}')
    med_g=$(printf "%s\n" "${gflops[@]}" | sort -n | awk '{a[i++]=$1} END {if(i%2==0) print (a[i/2-1]+a[i/2])/2; else print a[int(i/2)]}')
    med_e=$(printf "%s\n" "${effs[@]}" | sort -n | awk '{a[i++]=$1} END {if(i%2==0) print (a[i/2-1]+a[i/2])/2; else print a[int(i/2)]}')
    
    # Return the medians
    echo "$med_t $med_g $med_e"
}

# Run the benchmarks
tiling_data=$(run_benchmark "matrix-tiling" $ITERS)
simd_data=$(run_benchmark "matrix-simd" $ITERS)
blas_data=$(run_benchmark "matrix-openblas" $ITERS)

# Read the returned strings back into variables
read -r t_time t_gflops t_eff <<< "$tiling_data"
read -r s_time s_gflops s_eff <<< "$simd_data"
read -r b_time b_gflops b_eff <<< "$blas_data"

# Print the side-by-side comparison table
echo -e "\n=========================================================================="
echo -e "                   BENCHMARK RESULTS (MEDIAN OF $ITERS RUNS)\n Total GFLOPS->52.8    \n         "
echo -e "=========================================================================="
printf "%-20s | %-15s | %-15s | %-15s\n" "Implementation" "Time (seconds)" "Achieved GFLOPS" "Efficiency (%)"
echo -e "--------------------------------------------------------------------------"
printf "%-20s | %-15.6f | %-15.2f | %-15.2f\n" "Cache Tiling" "$t_time" "$t_gflops" "$t_eff"
printf "%-20s | %-15.6f | %-15.2f | %-15.2f\n" "Manual AVX (8x4)" "$s_time" "$s_gflops" "$s_eff"
printf "%-20s | %-15.6f | %-15.2f | %-15.2f\n" "OpenBLAS" "$b_time" "$b_gflops" "$b_eff"
echo -e "==========================================================================\n"
