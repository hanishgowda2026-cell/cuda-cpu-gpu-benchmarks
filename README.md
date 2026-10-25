# CUDA CPU vs GPU Performance Benchmarks

A CUDA C++ project exploring parallel computing algorithms and comparing CPU and GPU execution performance using an NVIDIA GPU.

## Project Overview

This project implements parallel algorithms using NVIDIA CUDA and evaluates their correctness and performance across different input sizes.

## Modules

1. **Vector Addition:** Compares CPU and GPU vector addition.
2. **Parallel Reduction:** Uses shared memory to calculate sums in parallel.
3. **Matrix Multiplication:** Compares basic and tiled CUDA kernels.
4. **Image Convolution:** Implements image convolution using basic and shared-memory approaches.
5. **Multi-Stage Parallel Reduction:** Performs reduction across multiple GPU kernel stages.
6. **Parallel Prefix Sum:** Computes inclusive prefix sums using CUDA.

## Technologies Used

- C++
- NVIDIA CUDA
- Python
- Matplotlib
- Kaggle GPU environment

## Development Environment

Developed and tested in Kaggle using an NVIDIA T4 GPU.

## Benchmarking

The project compares CPU and GPU execution times across multiple input sizes and checks GPU results against CPU implementations.

Kernel execution time and end-to-end execution time should be interpreted separately. GPU initialization and memory-transfer overhead can affect benchmark results.

## Key Learnings

- CUDA kernel development and thread indexing
- GPU memory management and shared memory
- Parallel reduction and prefix-sum algorithms
- Tiled matrix multiplication
- Performance measurement and correctness validation

## Repository Contents

- `.ipynb`: Notebook containing explanations, code, benchmark outputs, and visualizations.
- `.cu`: CUDA C++ source files for individual algorithms.

## Future Improvements

- Improve benchmark consistency using warm-up runs and repeated measurements.
- Add more parallel algorithms.
- Explore additional CUDA memory and kernel optimizations.
