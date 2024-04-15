# Heat Equation Solver

## Problem Statement
The heat equation is a partial differential equation that describes the propagation of heat in a region over time. The aim of our project is to develop and implement a parallelized simulation of heat diffusion using the explicit finite difference method on a three-dimensional grid.

## $\frac{∂U}{∂t}$ = $\frac{∂^2U}{∂x^2}$ + $\frac{∂^2U}{∂y^2}$ + $\frac{∂^2U}{∂z^2}$

## Test serial code
Type in your terminal to compile :\
`g++ heat3D_Serial.cpp -o heat3D_Serial.out`

To Run :\
`./heat3D_Serial.out`

## Test OpenMP code(Modify number of threads using omp_set_num_threads() as required)
Type in your terminal to compile :\
`g++ -fopenmp heat3D_OpenMP.cpp -o heat3D_OpenMP.out`

To Run :\
`./heat3D_OpenMP.out`

## Test CUDA code
Type in your terminal to compile :\
`nvcc heat3D_CUDA.cu -o heat3D_CUDA.out`

To Run :\
`./heat3D_CUDA.out`
