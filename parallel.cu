#include <algorithm>
#include <iostream>
#include <ctime>
#include <chrono>

// CUDA Kernel to convert 3D index layout to unrolled 1D layout
__global__ void getIndex(float *Un, float *Unp1, const int nx, const int ny, const int nz, const float a, const float dt, const float dx2, const float dy2, const float dz2)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= 1 && i < nx - 1 && j >= 1 && j < ny - 1 && k >= 1 && k < nz - 1)
    {
        int index = i * ny * nz + j * nz + k;
        float uijk = Un[index];
        float uim1jk = Un[(i - 1) * ny * nz + j * nz + k];
        float uip1jk = Un[(i + 1) * ny * nz + j * nz + k];
        float uijm1k = Un[i * ny * nz + (j - 1) * nz + k];
        float uijp1k = Un[i * ny * nz + (j + 1) * nz + k];
        float uijkm1 = Un[i * ny * nz + j * nz + (k - 1)];
        float uijkp1 = Un[i * ny * nz + j * nz + (k + 1)];

        Unp1[index] = uijk + a * dt * ((uim1jk - 2.0 * uijk + uip1jk) / dx2 + (uijm1k - 2.0 * uijk + uijp1k) / dy2 + (uijkm1 - 2.0 * uijk + uijkp1) / dz2);
    }
}

int main()
{
    // Width, Height and Depth of the area
    const int nx = 100;   
    const int ny = 100;   
    const int nz = 100; 

    // Thermal Conductivity constant
    const float a = 0.5;     

    // Horizontal, Vertical and Depth grid spacing
    const float dx = 0.01;    
    const float dy = 0.01;   
    const float dz = 0.01;  

    const float dx2 = dx * dx;
    const float dy2 = dy * dy;
    const float dz2 = dz * dz;

    // Number of time steps
    const int numSteps = 10000; 

    // Largest stable time step
    const float dt = dx2 * dy2 * dz2 / (2.0 * a * (dx2 * dy2 + dx2 * dz2 + dy2 * dz2)); 

    int numElements = nx * ny * nz;
    
    // Allocate two sets of data for current and next timesteps on GPU
    float *d_Un, *d_Unp1;
    cudaMalloc(&d_Un, numElements * sizeof(float));
    cudaMalloc(&d_Unp1, numElements * sizeof(float));

    // Initializing the data with a pattern of sphere of radius of 1/6 of the width
    float *Un = new float[numElements];
    float radius2 = (nx / 6.0) * (nx / 6.0);
    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            for (int k = 0; k < nz; k++)
            {
                int index = i * ny * nz + j * nz + k;
                // Distance of point from the origin
                float ds2 = (i - nx / 2) * (i - nx / 2) + (j - ny / 2) * (j - ny / 2) + (k - nz / 2) * (k - nz / 2);
                if (ds2 < radius2)
                {
                    Un[index] = 65.0;
                }
                else
                {
                    Un[index] = 5.0;
                }
            }
        }
    }
    cudaMemcpy(d_Un, Un, numElements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Unp1, Un, numElements * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(8, 8, 8);
    dim3 numBlocks((nx + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (ny + threadsPerBlock.y - 1) / threadsPerBlock.y,
                   (nz + threadsPerBlock.z - 1) / threadsPerBlock.z);

    // Time Start
    auto start = std::chrono::system_clock::now();

    // Main loop
    for (int n = 0; n <= numSteps; n++)
    {
        getIndex<<<numBlocks, threadsPerBlock>>>(d_Un, d_Unp1, nx, ny, nz, a, dt, dx2, dy2, dz2);
        cudaDeviceSynchronize();
        std::swap(d_Un, d_Unp1);
    }

    // Time End
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;

    // Release the memory
    delete[] Un;

    cudaFree(d_Un);
    cudaFree(d_Unp1);

    return 0;
}
