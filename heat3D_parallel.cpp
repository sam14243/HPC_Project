#include <algorithm>
#include <iostream>
#include <ctime>
#include <chrono>
#include <omp.h>

// Function to convert 3D index layout to unrolled 1D layout
int getIndex(const int i, const int j, const int k, const int width, const int height)
{
    return i * width * height + j * height + k;
}

int main()
{

    //Set number of threads
    // omp_set_num_threads(2);
    
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

    // Arrays for current and next timesteps
    float *Un = new float[numElements];
    float *Unp1 = new float[numElements];

    // Initializing the data with a pattern of sphere of radius of 1/6 of the width
    float radius2 = (nx / 6.0) * (nx / 6.0);

    for (int i = 0; i < nx; i++)
    {
        for (int j = 0; j < ny; j++)
        {
            for (int k = 0; k < nz; k++)
            {
                int index = getIndex(i, j, k, ny, nz);
                // Distance of point i, j, k from the origin
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

    // Fill in the data on the next step to ensure that the boundaries are identical.
    std::copy(Un, Un + numElements, Unp1);

    //Initialize clock with the system time
    auto start = std::chrono::system_clock::now();

    for (int n = 0; n <= numSteps; n++)
    {
        
// Parallelize the loop over i, j, k
#pragma omp parallel for collapse(3)


        for (int i = 1; i < nx - 1; i++)
        {
            for (int j = 1; j < ny - 1; j++)
            {
                for (int k = 1; k < nz - 1; k++)
                {
                    const int index = getIndex(i, j, k, ny, nz);
                    float uijk = Un[index];
                    float uim1jk = Un[getIndex(i - 1, j, k, ny, nz)];
                    float uip1jk = Un[getIndex(i + 1, j, k, ny, nz)];
                    float uijm1k = Un[getIndex(i, j - 1, k, ny, nz)];
                    float uijp1k = Un[getIndex(i, j + 1, k, ny, nz)];
                    float uijkm1 = Un[getIndex(i, j, k - 1, ny, nz)];
                    float uijkp1 = Un[getIndex(i, j, k + 1, ny, nz)];

                    Unp1[index] = uijk + a * dt * ((uim1jk - 2.0 * uijk + uip1jk) / dx2 + (uijm1k - 2.0 * uijk + uijp1k) / dy2 + (uijkm1 - 2.0 * uijk + uijkp1) / dz2);
                }
            }
        }
        // Swapping the pointers for the next timestep
        std::swap(Un, Unp1);
    }
    
    //Calculate elapsed time using the difference in the system time between starting and ending
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end-start;

    std::cout << "Elapsed time: " << elapsed_seconds.count() << "s" << std::endl;



    delete[] Un;
    delete[] Unp1;

    return 0;
}
