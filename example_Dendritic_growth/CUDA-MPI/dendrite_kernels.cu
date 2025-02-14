/*
 * Cahn-Hilliard Equaiton -- CUDA-Aware MPI version
 *
 * 10/28/2024
 * Ruoyao Zhang
 * Princeton Univeristy
 */

#include <cstdio>
#include <cmath>
#include <string>
#include <iostream>
#include <iomanip>
#include <fstream>

#ifdef HAVE_CUB
#include <cub/block/block_reduce.cuh>
#endif // HAVE_CUB

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus)                                                      \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
    }

// #ifdef USE_DOUBLE
typedef double real;
#define MPI_REAL_TYPE MPI_DOUBLE
// #else
// typedef float real;
// #define MPI_REAL_TYPE MPI_FLOAT
// #endif

#define BLKXSIZE 4
#define BLKYSIZE 4
#define BLKZSIZE 4

using namespace std;

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
__global__ void initialize_device(real* __restrict__ const phi, real* __restrict__ const T, \
                                real* __restrict__ const init_rand, \
                                const int offset, const int nx, const int my_ny, const int ny, const int nz){
    // Index follows iy*nx*nz + ix*nz + iz
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    
    real Radius;
    real R_phi = 5.0;
    real Delta = 0.25;

    unsigned int index;

    //real amp = 0.1;

    if ((iy < my_ny) && (ix < nx) && (iz < nz)) {
        index = iy*nx*nz + ix*nz + iz;

        Radius = sqrt((real) (ix - nx / 2) * (ix - nx / 2) \
                    + (iy+offset - ny / 2) * (iy+offset - ny / 2) \
                    + (iz - nz / 2) * (iz - nz / 2));
        
        phi[index] = -0.5 * (tanh((Radius - R_phi) * 1.0)) + 0.5;
        //c[index] = c_init + amp * (init_rand[index] - 0.5); // noise of 0.05 around c_init
        T[index] = -1.0 * Delta;
    }
}

void launch_initialize(real* __restrict__ const phi, real* __restrict__ const T, \
                    real* __restrict__ const init_rand,\
                    const int offset, const int nx, const int my_ny, const int ny, const int nz) {

    dim3 dim_grid((nx + BLKXSIZE - 1) / BLKXSIZE,
                (my_ny + BLKYSIZE - 1) / BLKYSIZE, // was ny instead of my_ny
                (nz + BLKZSIZE - 1) / BLKZSIZE);

    initialize_device<BLKXSIZE, BLKYSIZE, BLKZSIZE><<<dim_grid, {BLKXSIZE, BLKYSIZE, BLKZSIZE}>>>(
        phi, T, init_rand, offset, nx, my_ny, ny, nz);
    CUDA_RT_CALL(cudaGetLastError());
}

/////////////////////////////////////////////////////////////////////////////////////////////////
// Initialization for DBA
/////////////////////////////////////////////////////////////////////////////////////////////////
template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
__global__ void initialize_DBA_device(int *__restrict__ const wakeup_phi, int *__restrict__ const wakeup_phi_next, \
                                int* __restrict__ const wakeup_T, int* __restrict__ const wakeup_T_next, \
                                real* __restrict__ const phi, real* __restrict__ const T, \
                                real *__restrict__ const init_rand, \
                                const int offset, const int nx, const int my_ny, const int ny, const int nz)
{
    // int t_tot_x = blockDim.x * gridDim.x; // total number of threads in x direction
    // int t_tot_y = blockDim.y * gridDim.y;
    // int t_tot_z = blockDim.z * gridDim.z;
    int block_index_x = blockIdx.x;
    int block_index_y = blockIdx.y;
    int block_index_z = blockIdx.z;

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        // printf("blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
        // printf("block_index_x = %d, block_index_y = %d, block_index_z = %d\n", block_index_x, block_index_y, block_index_z);

        int block_index = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;

        wakeup_phi[block_index] = 1;
        wakeup_T[block_index] = 1;
        wakeup_phi_next[block_index] = 0;
        wakeup_T_next[block_index] = 0;
    }

    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int index;

    real Radius;
    real R_phi = 5.0;
    real Delta = 0.25;
    int num_nuclei_z = 10;
    int num_nuclei_y = num_nuclei_z * ny / nz;

    if ((iy < my_ny) && (ix < nx) && (iz < nz))
    {
        index = iy * nx * nz + ix * nz + iz;

        //Radius = sqrt((real) (ix - nx / 2) * (ix - nx / 2) \
                    + (iy+offset - ny / 2) * (iy+offset - ny / 2) \
                    + (iz - nz / 2) * (iz - nz / 2));
        
        //phi[index] = -0.5 * (tanh((Radius - R_phi) * 1.0)) + 0.5;

        // new initialization for phi
        phi[index] = 0.;

        for (int num_y = 0; num_y < num_nuclei_y; num_y++){
            for (int num_z = 0; num_z < num_nuclei_z; num_z++){   
                Radius = sqrt((real) (ix) * (ix) \
                    + (iy+offset - ny / (num_nuclei_y+1)*(num_y+1)) * (iy+offset - ny / (num_nuclei_y+1)*(num_y+1)) \
                    + (iz - nz / (num_nuclei_z+1)*(num_z+1)) * (iz - nz / (num_nuclei_z+1)*(num_z+1)));

                phi[index] += -0.5 * (tanh((Radius - R_phi) * 1.0)) + 0.5;
            }
        }

        T[index] = -1.0 * Delta;
    }
}


void launch_initialize_DBA(int *__restrict__ const wakeup_phi, int *__restrict__ const wakeup_phi_next, \
                        int* __restrict__ const wakeup_T, int* __restrict__ const wakeup_T_next, \
                        real* __restrict__ const phi, real* __restrict__ const T, \
                        real *__restrict__ const init_rand, \
                        const int offset, const int nx, const int my_ny, const int ny, const int nz)
{

    dim3 dim_grid((nx + BLKXSIZE - 1) / BLKXSIZE,
                  (my_ny + BLKYSIZE - 1) / BLKYSIZE,
                  (nz + BLKZSIZE - 1) / BLKZSIZE);

    printf("Launching initialize_DBA_device with grid (%d, %d, %d)\n", dim_grid.x, dim_grid.y, dim_grid.z);
    initialize_DBA_device<BLKXSIZE, BLKYSIZE, BLKZSIZE><<<dim_grid, {BLKXSIZE, BLKYSIZE, BLKZSIZE}>>>(
        wakeup_phi, wakeup_phi_next, wakeup_T, wakeup_T_next, \
        phi, T, init_rand, offset, nx, my_ny, ny, nz);
    CUDA_RT_CALL(cudaGetLastError());
}

// Gradient in x and z needs to be manually set for periodic BC
// while in y it is taken care of by boundary communication
__device__ real GradientX(real* __restrict__ const a, const int ix, const int iy, const int iz, \
                            const int nx, const int nz, const real dx){
    int xp = ix + 1;
    int xn = ix - 1;

    /*
    // periodic BC
    if (xp > nx - 1){
        xp = 0;
    }
    if (xn < 0){
        xn = nx - 1;
    }*/

    // No flux BC
    if (xp > nx - 1){
        xp = nx - 1;
    }
    if (xn < 0){
        xn = 0;
    }

    real ax = (a[iy*nx*nz + xp*nz + iz] - a[iy*nx*nz + xn*nz + iz]) / (2.0 * dx);

    return ax;
}

__device__ real GradientY(real* __restrict__ const a, const int ix, const int iy, const int iz, \
                            const int nx, const int nz, const real dx){
    int yp = iy + 1;
    int yn = iy - 1;

    real ay = (a[yp*nx*nz + ix*nz + iz] - a[yn*nx*nz + ix*nz + iz]) / (2.0 * dx);

    return ay;
}

__device__ real GradientZ(real* __restrict__ const a, const int ix, const int iy, const int iz, \
                            const int nx, const int nz, const real dx){
    int zp = iz + 1;
    int zn = iz - 1;

    /*
    // periodic BC
    if (zp > nz - 1){
        zp = 0;
    }
    if (zn < 0){
        zn = nz - 1;
    }*/

    // No flux BC
    if (zp > nz - 1){
        zp = nz - 1;
    }
    if (zn < 0){
        zn = 0;
    }

    real az = (a[iy*nx*nz + ix*nz + zp] - a[iy*nx*nz + ix*nz + zn]) / (2.0 * dx);

    return az;
}

// 3D 27-point Laplacian
__device__ real iso_Laplacian(real* __restrict__ const a, const int ix, const int iy, const int iz,\
                            const int nx, const int nz, const real dx){

    // Index follows iy*nx*nz + ix*nz + iz
    int xp, xn, yp, yn, zp, zn;

    xp = ix + 1;
    xn = ix - 1;
    yp = iy + 1;
    yn = iy - 1;
    zp = iz + 1;
    zn = iz - 1;

    /*
    // periodic BC
    if (xp > nx-1){
        xp = 0;
    }
    if (xn < 0){
        xn = nx-1;
    }
    if (zp > nz-1){
        zp = 0;
    }
    if (zn < 0){
        zn = nz-1;
    }*/

    // No flux BC
    if (xp > nx-1){
        xp = nx-1;
    }
    if (xn < 0){
        xn = 0;
    }
    if (zp > nz-1){
        zp = nz-1;
    }
    if (zn < 0){
        zn = 0;
    }

    real result = 1.0 / (dx * dx) * ( \
                - 64. / 15. * a[iy*nx*nz + ix*nz + iz] \
                + 7. / 15. * (a[iy*nx*nz + xp*nz + iz] + a[iy*nx*nz + xn*nz + iz] \
                                 + a[yp*nx*nz + ix*nz + iz] + a[yn*nx*nz + ix*nz + iz] \
                                 + a[iy*nx*nz + ix*nz + zp] + a[iy*nx*nz + ix*nz + zn] ) \
                + 0.1 * (a[yp*nx*nz + xp*nz + iz] + a[yn*nx*nz + xp*nz + iz] + a[yp*nx*nz + xn*nz + iz] + a[yn*nx*nz + xn*nz + iz] \
                            + a[iy*nx*nz + xp*nz + zp] + a[yp*nx*nz + ix*nz + zp] + a[yn*nx*nz + ix*nz + zp] + a[iy*nx*nz + xn*nz + zp] \
                            + a[iy*nx*nz + xp*nz + zn] + a[yp*nx*nz + ix*nz + zn] + a[yn*nx*nz + ix*nz + zn] + a[iy*nx*nz + xn*nz + zn] \
                            ) \
                + 1. / 30. * (a[yp*nx*nz + xp*nz + zp] + a[yp*nx*nz + xn*nz + zp] + a[yn*nx*nz + xn*nz + zp] + a[yn*nx*nz + xp*nz + zp] \
                            + a[yp*nx*nz + xp*nz + zn] + a[yp*nx*nz + xn*nz + zn] + a[yn*nx*nz + xn*nz + zn] + a[yn*nx*nz + xp*nz + zn] \
                            ) \
                );
    return result;
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
__global__ void phi_kernel(real *__restrict__ const phi, real *__restrict__ const T, \
                           real *__restrict__ const phi_new, \
                           const int iy_start, const int iy_end, const int nx, const int nz, const real dx)
{

    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    real epsilon = 0.01;
    real delta = 0.5;
    real dt = 5e-5;
    real alpha = 0.9;
    real gamma = 40.0;
    real tau = 3e-4;

    real dpx, dpy, dpz;
    real lap_phi;
    real v2, sigma, m, phi_change;

    unsigned int index;

    // calculating phi field
    if ((iy < iy_end) && (ix < nx) && (iz < nz))
    {
        index = iy * nx * nz + ix * nz + iz;

        dpx = GradientX(phi, ix, iy, iz, nx, nz, dx);
        dpy = GradientY(phi, ix, iy, iz, nx, nz, dx);
        dpz = GradientZ(phi, ix, iy, iz, nx, nz, dx);

        lap_phi = iso_Laplacian(phi, ix, iy, iz, nx, nz, dx);

        v2 = dpx*dpx + dpy*dpy + dpz*dpz;

        if (v2 <= 1e-6){
            sigma = 1.0;
        } else{
            sigma = 1.0 - delta * (1.0 - (pow(dpx, 4) + pow(dpy, 4) + pow(dpz, 4)) / (v2 * v2));
        }

        m = -1.0 * alpha / M_PI * atan(gamma * sigma * T[index]);

        // Calculate dp/dt
        phi_change = 1.0 / tau * (epsilon * epsilon * lap_phi \
                                + phi[index] * (1.0 - phi[index]) * (phi[index] - 0.5 + m));

        // Update p and T
        phi_new[index] = phi[index] + dt * phi_change;
    }
}

void launch_phi_kernel(real *__restrict__ const phi, real *__restrict__ const T, \
                    real *__restrict__ const phi_new, \
                    const int iy_start, const int iy_end, const int nx, const int nz, const real dx, \
                    cudaStream_t stream)
{

    dim3 dim_grid((nx + BLKXSIZE - 1) / BLKXSIZE,
                  ((iy_end - iy_start) + BLKYSIZE - 1) / BLKYSIZE,
                  (nz + BLKZSIZE - 1) / BLKZSIZE);

    // printf("Launching mu_kernel with grid (%d, %d, %d)\n", dim_grid.x, dim_grid.y, dim_grid.z);
    phi_kernel<BLKXSIZE, BLKYSIZE, BLKZSIZE><<<dim_grid, {BLKXSIZE, BLKYSIZE, BLKZSIZE}, 0, stream>>>(
        phi, T, phi_new, iy_start, iy_end, nx, nz, dx);
    CUDA_RT_CALL(cudaGetLastError());
}


template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
__global__ void phi_DBA_kernel(int *__restrict__ const wakeup_phi,
                            real *__restrict__ const phi, real *__restrict__ const T, \
                            real *__restrict__ const phi_new, real *__restrict__ const metric, \
                            const int iy_start, const int iy_end, const int nx, const int nz, const real dx)
{

    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ int wakeup_phi_s;

    real epsilon = 0.01;
    real delta = 0.5;
    real dt = 5e-5;
    real alpha = 0.9;
    real gamma = 40.0;
    real tau = 3e-4;

    real dpx, dpy, dpz;
    real lap_phi;
    real v2, sigma, m, phi_change;

    unsigned int index;

    // original
    //int block_index_x = blockIdx.x;
    //int block_index_y = blockIdx.y;
    //int block_index_z = blockIdx.z;

    // recalculating block_index
    int block_index_x = ix / BLKXSIZE;
    int block_index_y = iy / BLKYSIZE;
    int block_index_z = iz / BLKZSIZE;

    // Read the wakeup status into wake_up_s
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        int block_index = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;

        wakeup_phi_s = wakeup_phi[block_index];
    }

    // Synchronize threads in threadblock before using the shared memory
    __syncthreads();

    // calculating mu for CH
    if (wakeup_phi_s && (iy < iy_end) && (ix < nx) && (iz < nz)){

        index = iy * nx * nz + ix * nz + iz;

        dpx = GradientX(phi, ix, iy, iz, nx, nz, dx);
        dpy = GradientY(phi, ix, iy, iz, nx, nz, dx);
        dpz = GradientZ(phi, ix, iy, iz, nx, nz, dx);

        lap_phi = iso_Laplacian(phi, ix, iy, iz, nx, nz, dx);

        v2 = dpx*dpx + dpy*dpy + dpz*dpz;

        if (v2 <= 1e-6){
            sigma = 1.0;
        } else{
            sigma = 1.0 - delta * (1.0 - (pow(dpx, 4) + pow(dpy, 4) + pow(dpz, 4)) / (v2 * v2));
        }

        m = -1.0 * alpha / M_PI * atan(gamma * sigma * T[index]);

        // Calculate dp/dt
        phi_change = 1.0 / tau * (epsilon * epsilon * lap_phi \
                                + phi[index] * (1.0 - phi[index]) * (phi[index] - 0.5 + m));

        // Update phi
        phi_new[index] = phi[index] + dt * phi_change;
    }
}

void launch_phi_DBA_kernel(int *__restrict__ const wakeup_phi,
                        real *__restrict__ const phi, real *__restrict__ const T, \
                        real *__restrict__ const phi_new, real *__restrict__ const metric, \
                        const int iy_start, const int iy_end, const int nx, const int nz, const real dx, \
                        cudaStream_t stream)
{

    dim3 dim_grid((nx + BLKXSIZE - 1) / BLKXSIZE,
                  ((iy_end - iy_start) + BLKYSIZE - 1) / BLKYSIZE,
                  (nz + BLKZSIZE - 1) / BLKZSIZE);

    //printf("Launching mu_kernel with grid (%d, %d, %d)\n", dim_grid.x, dim_grid.y, dim_grid.z);
    phi_DBA_kernel<BLKXSIZE, BLKYSIZE, BLKZSIZE><<<dim_grid, {BLKXSIZE, BLKYSIZE, BLKZSIZE}, 0, stream>>>(
        wakeup_phi, phi, T, phi_new, metric, iy_start, iy_end, nx, nz, dx);
    CUDA_RT_CALL(cudaGetLastError());
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
__global__ void T_kernel(real *__restrict__ const T, real *__restrict__ const T_new, \
                        real* __restrict__ const phi, real* __restrict__ const phi_new, \
                        real *__restrict__ const sum_c, \
                        const int iy_start, const int iy_end, const int nx, const int nz, const real dx, \
                        const bool calculate_sum)
{
#ifdef HAVE_CUB
    typedef cub::BlockReduce<real, BLOCK_DIM_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_DIM_Y, BLOCK_DIM_Z>
        BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
#endif // HAVE_CUB
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    real local_sum_c = 0.;

    real dt = 5e-5;

    real lap_T;

    unsigned int index;
    // calculating
    if ((iy < iy_end) && (ix < nx) && (iz < nz)){

        index = iy * nx * nz + ix * nz + iz;

        lap_T = iso_Laplacian(T, ix, iy, iz, nx, nz, dx);

        // Update T
        T_new[index] = T[index] + dt * lap_T + phi_new[index] - phi[index];

        if (calculate_sum)
        {
            local_sum_c += T_new[index];
        }
    }

    if (calculate_sum)
    {
#ifdef HAVE_CUB
        real block_sum_c = BlockReduce(temp_storage).Sum(local_sum_c);
        if (0 == threadIdx.y && 0 == threadIdx.x && 0 == threadIdx.z)
        {
            atomicAdd(sum_c, block_sum_c);
        }
#else
        atomicAdd(sum_c, local_sum_c);
#endif // HAVE_CUB
    }
}

void launch_T_kernel(real *__restrict__ const T, real *__restrict__ const T_new, \
                    real* __restrict__ const phi, real* __restrict__ const phi_new, \
                    real *__restrict__ const sum_c, \
                    const int iy_start, const int iy_end, const int nx, const int nz, const real dx, \
                    const bool calculate_sum, cudaStream_t stream)
{

    dim3 dim_grid((nx + BLKXSIZE - 1) / BLKXSIZE,
                  ((iy_end - iy_start) + BLKYSIZE - 1) / BLKYSIZE,
                  (nz + BLKZSIZE - 1) / BLKZSIZE);

    T_kernel<BLKXSIZE, BLKYSIZE, BLKZSIZE><<<dim_grid, {BLKXSIZE, BLKYSIZE, BLKZSIZE}, 0, stream>>>(
        T, T_new, phi, phi_new, sum_c,
        iy_start, iy_end, nx, nz, dx, calculate_sum);
    CUDA_RT_CALL(cudaGetLastError());
}


template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
__global__ void T_DBA_kernel(int *__restrict__ const wakeup_T, int *__restrict__ const wakeup_phi, \
                            real *__restrict__ const T, real *__restrict__ const T_new, \
                            real* __restrict__ const phi, real* __restrict__ const phi_new, \
                            real* __restrict__ const metric_T, \
                            real *__restrict__ const sum_c, int *__restrict__ const sum_wakeup, \
                            const int iy_start, const int iy_end, const int nx, const int nz, const real dx, \
                            const bool calculate_sum)
{
#ifdef HAVE_CUB
    typedef cub::BlockReduce<real, BLOCK_DIM_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_DIM_Y, BLOCK_DIM_Z>
        BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage_c;
    __shared__ typename BlockReduce::TempStorage temp_storage_wakeup;
#endif // HAVE_CUB
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    real local_sum_c = 0.;
    int local_sum_wakeup = 0;
    real dt = 5e-5;
    real lap_T;
    unsigned int index;

    __shared__ int wakeup_T_s, wakeup_phi_s;

    // original
    //int block_index_x = blockIdx.x;
    //int block_index_y = blockIdx.y;
    //int block_index_z = blockIdx.z;

    // recalculating block_index
    int block_index_x = ix / BLKXSIZE;
    int block_index_y = iy / BLKYSIZE;
    int block_index_z = iz / BLKZSIZE;

    // Read the wakeup status into wake_up_s
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {
        int block_index = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;

        wakeup_T_s = wakeup_T[block_index];
        wakeup_phi_s = wakeup_phi[block_index];

        if (calculate_sum)
        {
            local_sum_wakeup += wakeup_T_s;
        }
    }

    // Synchronize threads in threadblock before using the shared memory
    __syncthreads();

    // calculating update c

    if ((iy < iy_end) && (ix < nx) && (iz < nz)){

        index = iy * nx * nz + ix * nz + iz;

        if (wakeup_T_s){
            lap_T = iso_Laplacian(T, ix, iy, iz, nx, nz, dx);
            //metric_T[index] = lap_T;

            // Update T
            T_new[index] = T[index] + dt * lap_T;
            metric_T[index] = fabs(T_new[index] - T[index]);
        }

        if (wakeup_phi_s){
                
            // Update T
            T_new[index] += phi_new[index] - phi[index];
            metric_T[index] = fabs(T_new[index] - T[index]);
        }

        if (calculate_sum){
            local_sum_c += T_new[index];
        }
    }

    if (calculate_sum)
    {
#ifdef HAVE_CUB
        real block_sum_c = BlockReduce(temp_storage_c).Sum(local_sum_c);
        // int block_sum_wakeup = BlockReduce(temp_storage_wakeup).Sum(local_sum_wakeup);
        if (0 == threadIdx.y && 0 == threadIdx.x && 0 == threadIdx.z)
        {
            atomicAdd(sum_c, block_sum_c);
            atomicAdd(sum_wakeup, local_sum_wakeup);
            // printf("have CUB\n");
        }
#else
        atomicAdd(sum_c, local_sum_c);
        atomicAdd(sum_wakeup, local_sum_wakeup);
#endif // HAVE_CUB
    }
}

void launch_T_DBA_kernel(int *__restrict__ const wakeup_T, int *__restrict__ const wakeup_phi, \
                        real *__restrict__ const T, real *__restrict__ const T_new, \
                        real* __restrict__ const phi, real* __restrict__ const phi_new, \
                        real* __restrict__ const metric_T, \
                        real *__restrict__ const sum_c, int *__restrict__ const sum_wakeup, \
                        const int iy_start, const int iy_end, const int nx, const int nz, const real dx, \
                        const bool calculate_sum, cudaStream_t stream)
{

    dim3 dim_grid((nx + BLKXSIZE - 1) / BLKXSIZE,
                  ((iy_end - iy_start) + BLKYSIZE - 1) / BLKYSIZE,
                  (nz + BLKZSIZE - 1) / BLKZSIZE);

    T_DBA_kernel<BLKXSIZE, BLKYSIZE, BLKZSIZE><<<dim_grid, {BLKXSIZE, BLKYSIZE, BLKZSIZE}, 0, stream>>>(
        wakeup_T, wakeup_phi, T, T_new, phi, phi_new, metric_T, sum_c, sum_wakeup,
        iy_start, iy_end, nx, nz, dx, calculate_sum);
    CUDA_RT_CALL(cudaGetLastError());
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
__global__ void wakeup_next_kernel(real* __restrict__ const phi_new, real* __restrict__ const T_new, \
                                int *__restrict__ const wakeup_phi, int *__restrict__ const wakeup_phi_next, \
                                int *__restrict__ const wakeup_T, int *__restrict__ const wakeup_T_next, \
                                real *__restrict__ const metric_phi, real *__restrict__ const metric_T, \
                                const int my_ny, const int nx, const int nz)
{

    //int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // we want to include the halo blocks
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ int wakeup_phi_s, wakeup_phi_s_next, wakeup_T_s, wakeup_T_s_next;

    real metric_eps_T = 1e-6;
    real metric_eps_phi = 1e-3;

    unsigned int index;

    int block_index_x = blockIdx.x;
    int block_index_y = blockIdx.y;
    int block_index_z = blockIdx.z;
    int block_index;

    // Read the wakeup status into wake_up_s
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
    {

        block_index = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;

        wakeup_phi_s = wakeup_phi[block_index];
        wakeup_T_s = wakeup_T[block_index];
        wakeup_phi[block_index] = 0;
        wakeup_T[block_index] = 0;
        wakeup_phi_s_next = 0;
        wakeup_T_s_next = 0;
    }

    // Synchronize threads in threadblock before using the shared memory
    __syncthreads();

    // Evaluating activation criteria for phi
    //if (wakeup_phi_s && (iy < my_ny) && (ix < nx) && (iz < nz))
    if ((iy < my_ny) && (ix < nx) && (iz < nz))
    {
        index = iy * nx * nz + ix * nz + iz;

        if ( (phi_new[index] >= metric_eps_phi) && (phi_new[index] <= (1.0-metric_eps_phi)) \
            && (wakeup_phi_s_next == 0))
        {   // additional criterion: || (timestep < 10000)
            // printf("metric[%d] = %f\n", index, metric[index]);
            wakeup_phi_s_next = 1;
        }

        // Synchronize threads in threadblock before using the shared memory
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        {
            block_index = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;
            wakeup_phi_next[block_index] = wakeup_phi_s_next;
        }
    }

    // Evaluating activation criteria for T
    //if (wakeup_phi_s && (iy < my_ny) && (ix < nx) && (iz < nz))
    if ((iy < my_ny) && (ix < nx) && (iz < nz))
    {
        index = iy * nx * nz + ix * nz + iz;

        //if ((metric_T[index] > metric_eps_T && wakeup_T_s_next == 0) || (wakeup_phi_s_next == 1))
        if ( (metric_T[index] > metric_eps_T) && (wakeup_T_s_next == 0)){
            // printf("metric[%d] = %f\n", index, metric[index]);
            wakeup_T_s_next = 1;
        }

        // Synchronize threads in threadblock before using the shared memory
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
        {
            block_index = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;
            wakeup_T_next[block_index] = wakeup_T_s_next;
        }
    }
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
__global__ void wakeup_neighbor_kernel(int *__restrict__ const wakeup_phi, int *__restrict__ const wakeup_phi_next, \
                                int *__restrict__ const wakeup_T, int *__restrict__ const wakeup_T_next, \
                                const int my_ny, const int nx, const int nz)
{
    int block_index_x = blockIdx.x;
    int block_index_y = blockIdx.y;
    int block_index_z = blockIdx.z;
    int block_index = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;

    //if (wakeup_phi_next[block_index] || wakeup_T_next[block_index]){
    if (1==1){
        /*
        // periodic BC
        int block_xm = (block_index_x - 1 >= 0) ? block_index_x - 1 : gridDim.x - 1;
        int block_ym = (block_index_y - 1 >= 0) ? block_index_y - 1 : gridDim.y - 1;
        int block_zm = (block_index_z - 1 >= 0) ? block_index_z - 1 : gridDim.z - 1;
        int block_xp = (block_index_x + 1 < gridDim.x) ? block_index_x + 1 : 0;
        int block_yp = (block_index_y + 1 < gridDim.y) ? block_index_y + 1 : 0;
        int block_zp = (block_index_z + 1 < gridDim.z) ? block_index_z + 1 : 0;
        */

        // no periodic BC for y; no-flux BC for x and z
        int block_xm = (block_index_x - 1 >= 0) ? block_index_x - 1 : block_index_x;
        int block_ym = (block_index_y - 1 >= 0) ? block_index_y - 1 : block_index_y;
        int block_zm = (block_index_z - 1 >= 0) ? block_index_z - 1 : block_index_z;
        int block_xp = (block_index_x + 1 < gridDim.x) ? block_index_x + 1 : block_index_x;
        int block_yp = (block_index_y + 1 < gridDim.y) ? block_index_y + 1 : block_index_y;
        int block_zp = (block_index_z + 1 < gridDim.z) ? block_index_z + 1 : block_index_z;

        int block_index_xm = block_index_y * gridDim.x * gridDim.z + block_xm * gridDim.z + block_index_z;
        int block_index_xp = block_index_y * gridDim.x * gridDim.z + block_xp * gridDim.z + block_index_z;
        int block_index_ym = block_ym * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;
        int block_index_yp = block_yp * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;
        int block_index_zm = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_zm;
        int block_index_zp = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_zp;

        if (wakeup_phi_next[block_index]){
            wakeup_phi[block_index] = 1;
            wakeup_phi[block_index_xm] = 1;
            wakeup_phi[block_index_xp] = 1;
            wakeup_phi[block_index_ym] = 1;
            wakeup_phi[block_index_yp] = 1;
            wakeup_phi[block_index_zm] = 1;
            wakeup_phi[block_index_zp] = 1;
        }
        
        if (wakeup_T_next[block_index]){
            wakeup_T[block_index] = 1;
            wakeup_T[block_index_xm] = 1;
            wakeup_T[block_index_xp] = 1;
            wakeup_T[block_index_ym] = 1;
            wakeup_T[block_index_yp] = 1;
            wakeup_T[block_index_zm] = 1;
            wakeup_T[block_index_zp] = 1;
        }
    }
}

void launch_wakeup_DBA_kernel(real* __restrict__ const phi_new, real* __restrict__ const T_new, \
                            int *__restrict__ const wakeup_phi, int *__restrict__ const wakeup_phi_next, \
                            int *__restrict__ const wakeup_T, int *__restrict__ const wakeup_T_next, \
                            real *__restrict__ const metric_phi, real *__restrict__ const metric_T, \
                            const int my_ny, const int nx, const int nz, cudaStream_t stream)
{

    dim3 dim_grid((nx + BLKXSIZE - 1) / BLKXSIZE,
                  (my_ny + BLKYSIZE - 1) / BLKYSIZE,
                  (nz + BLKZSIZE - 1) / BLKZSIZE);

    //printf("Launching mu_kernel with grid (%d, %d, %d)\n", dim_grid.x, dim_grid.y, dim_grid.z);
    wakeup_next_kernel<BLKXSIZE, BLKYSIZE, BLKZSIZE><<<dim_grid, {BLKXSIZE, BLKYSIZE, BLKZSIZE}, 0, stream>>>(
        phi_new, T_new, wakeup_phi, wakeup_phi_next, wakeup_T, wakeup_T_next, \
        metric_phi, metric_T, my_ny, nx, nz);
    CUDA_RT_CALL(cudaGetLastError());
}

void launch_neighbor_DBA_kernel(int *__restrict__ const wakeup_phi, int *__restrict__ const wakeup_phi_next, \
                            int *__restrict__ const wakeup_T, int *__restrict__ const wakeup_T_next, \
                            const int my_ny, const int nx, const int nz, cudaStream_t stream)
{
    dim3 dim_grid((nx + BLKXSIZE - 1) / BLKXSIZE,
                  (my_ny + BLKYSIZE - 1) / BLKYSIZE,
                  (nz + BLKZSIZE - 1) / BLKZSIZE);

    wakeup_neighbor_kernel<BLKXSIZE, BLKYSIZE, BLKZSIZE><<<dim_grid, {BLKXSIZE, BLKYSIZE, BLKZSIZE}, 0, stream>>>(
        wakeup_phi, wakeup_phi_next, wakeup_T, wakeup_T_next, \
        my_ny, nx, nz);
    CUDA_RT_CALL(cudaGetLastError());
}

void integral_c(real *__restrict__ const c, int nx, int iy_start, int iy_end, int nz)
{
    real sum = 0.;

    for (int j = iy_start; j < iy_end; j++)
    {
        for (int i = 0; i < nx; i++)
        {
            for (int k = 0; k < nz; k++)
            {
                sum += c[j * nx * nz + i * nz + k];
            }
        }
    }
    printf("Sum is %8.5f\n", sum);
}

void integral_wakeup(int *__restrict__ const wakeup, int nx, int iy_start, int iy_end, int nz)
{
    unsigned int sum = 0;

    for (int k = 0; k < nz; k++)
    {
        for (int j = 0; j <= iy_end - iy_start; j++)
        {
            // for (int j = iy_start; j <= iy_end; j++) {
            for (int i = 0; i < nx; i++)
            {
                int block_x = i / BLKXSIZE;
                int block_y = j / BLKYSIZE;
                int block_z = k / BLKZSIZE;

                int index_block = block_y * ((nx + BLKXSIZE - 1) / BLKXSIZE) * ((nz + BLKZSIZE - 1) / BLKZSIZE) + block_x * ((nz + BLKZSIZE - 1) / BLKZSIZE) + block_z;

                sum += wakeup[index_block];
            }
        }
    }

    printf("Total number of active blocks: %d\n", sum / 64);
}

void write_output_pvtr(int total_rank, int t, int nx, int ny, int nz)
{
    string name = "output_" + to_string(t) + ".pvtr";
    ofstream ofile(name);

    int chunk_size;
    int chunk_size_low = (ny - 2) / total_rank;
    int chunk_size_high = chunk_size_low + 1;
    int num_ranks_low = total_rank * chunk_size_low + total_rank - (ny - 2); // Number of ranks with chunk_size = chunk_size_low
    int iy_start_global, iy_end_global;

    // pvtr preamble
    ofile << "<VTKFile type=\"PRectilinearGrid\" version=\"2.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">" << endl;
    ofile << "<PRectilinearGrid WholeExtent=\"0 " << nx - 1 << " 1 " << ny - 2 << " 0 " << nz - 1 << "\" GhostLevel=\"0\">" << endl;

    // write names of arrays
    ofile << "<PPointData>" << endl;
    ofile << "<PDataArray type=\"Float32\" Name=\"phi\"/>" << endl;
    ofile << "<PDataArray type=\"Float32\" Name=\"T\"/>" << endl;
    ofile << "<PDataArray type=\"Int8\" Name=\"active_phi\"/>" << endl;
    ofile << "<PDataArray type=\"Int8\" Name=\"active_T\"/>" << endl;
    ofile << "</PPointData>" << endl;

    // write coordinates
    ofile << "<PCoordinates>" << endl;
    ofile << "<PDataArray type=\"Float32\"/>" << endl;
    ofile << "<PDataArray type=\"Float32\"/>" << endl;
    ofile << "<PDataArray type=\"Float32\"/>" << endl;
    ofile << "</PCoordinates>" << endl;
    for (int rank = 0; rank < total_rank; rank++){
        if (rank < num_ranks_low){
            chunk_size = chunk_size_low;
        }
        else{
            chunk_size = chunk_size_high;
        }

        if (rank < num_ranks_low){
            iy_start_global = rank * chunk_size_low + 1;
        }
        else{
            iy_start_global =
                num_ranks_low * chunk_size_low + (rank - num_ranks_low) * chunk_size_high + 1;
        }
        iy_end_global = iy_start_global + chunk_size - 1;

        ofile << "<Piece Extent=\"0 " << nx - 1 << " " << iy_start_global - 1
              << " " << iy_end_global + 1
              << " 0 " << nz - 1 << "\" Source=\"output_" << t << "_" << rank << ".vtr\"/>" << endl;
    }

    ofile << "</PRectilinearGrid>" << endl;
    ofile << "</VTKFile>" << endl;

    ofile.close();
}

void write_output_vtr(real *__restrict__ const phi, real *__restrict__ const T, \
                    int *__restrict__ const wakeup_phi, int *__restrict__ const wakeup_T, \
                    int t, int local_rank, int nx, int iy_start, int iy_end, int nz)
{
    string name = "output_" + to_string(t) + "_" + to_string(local_rank) + ".vtr";
    ofstream ofile(name);
    int FlattenedID;
    int block_x, block_y, block_z;
    int index_block;

    // vtr preamble
    ofile << "<VTKFile type=\"RectilinearGrid\" version=\"2.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">" << endl;
    ofile << "<RectilinearGrid WholeExtent=\"0 " << nx - 1 << " " << iy_start << " " << iy_end << " 0 " << nz - 1 << "\">" << endl;
    ofile << "<Piece Extent=\"0 " << nx - 1 << " " << iy_start << " " << iy_end << " 0 " << nz - 1 << "\">" << endl;
    
    //write field data
    ofile << "<PointData>" << endl;

    // phi data
    ofile << "<DataArray type=\"Float32\" Name=\"phi\" format=\"ascii\" RangeMin=\"-1.0\" RangeMax=\"1.00\">" << endl;
    for (int k = 0; k < nz; k++){
        for (int j = 0; j <= iy_end - iy_start; j++){
            for (int i = 0; i < nx; i++){
                FlattenedID = j * nx * nz + i * nz + k;
                ofile << std::fixed << phi[FlattenedID] << " ";
            }
        }
    }
    ofile << endl;
    ofile << "</DataArray>" << endl;

    // write T data
    ofile << "<DataArray type=\"Float32\" Name=\"T\" format=\"ascii\" RangeMin=\"-1.0\" RangeMax=\"1.00\">" << endl;
    for (int k = 0; k < nz; k++){
        for (int j = 0; j <= iy_end - iy_start; j++){
            for (int i = 0; i < nx; i++){
                FlattenedID = j * nx * nz + i * nz + k;
                ofile << std::fixed << T[FlattenedID] << " ";
            }
        }
    }
    ofile << endl;
    ofile << "</DataArray>" << endl;

    // write wakeup_phi block data
    ofile << "<DataArray type=\"Int8\" Name=\"active_phi\" format=\"ascii\" RangeMin=\"-1.0\" RangeMax=\"1.00\">" << endl;
    for (int k = 0; k < nz; k++){
        for (int j = 0; j <= iy_end - iy_start; j++){
            for (int i = 0; i < nx; i++){
                block_x = i / BLKXSIZE;
                block_y = j / BLKYSIZE;
                block_z = k / BLKZSIZE;

                index_block = block_y * ((nx + BLKXSIZE - 1) / BLKXSIZE) * ((nz + BLKZSIZE - 1) / BLKZSIZE) \
                            + block_x * ((nz + BLKZSIZE - 1) / BLKZSIZE) + block_z;
                ofile << wakeup_phi[index_block] << " ";
                /*
                if ( (j == 0) || (j == iy_end-iy_start)){
                    ofile << 0 << " ";
                } else{
                    block_x = i / BLKXSIZE;
                    block_y = (j-1) / BLKYSIZE;
                    block_z = k / BLKZSIZE;

                    index_block = block_y * ((nx + BLKXSIZE - 1) / BLKXSIZE) * ((nz + BLKZSIZE - 1) / BLKZSIZE) \
                            + block_x * ((nz + BLKZSIZE - 1) / BLKZSIZE) + block_z;
                    ofile << wakeup_phi[index_block] << " ";
                }*/
            }
        }
    }
    ofile << endl;
    ofile << "</DataArray>" << endl;

    // write wakeup_T block data
    ofile << "<DataArray type=\"Int8\" Name=\"active_T\" format=\"ascii\" RangeMin=\"-1.0\" RangeMax=\"1.00\">" << endl;
    for (int k = 0; k < nz; k++){
        for (int j = 0; j <= iy_end - iy_start; j++){
            for (int i = 0; i < nx; i++){
                block_x = i / BLKXSIZE;
                block_y = j / BLKYSIZE;
                block_z = k / BLKZSIZE;

                index_block = block_y * ((nx + BLKXSIZE - 1) / BLKXSIZE) * ((nz + BLKZSIZE - 1) / BLKZSIZE) \
                            + block_x * ((nz + BLKZSIZE - 1) / BLKZSIZE) + block_z;
                ofile << wakeup_T[index_block] << " ";
            }
        }
    }
    ofile << endl;
    ofile << "</DataArray>" << endl;

    // end of data
    ofile << "</PointData>" << endl;
    ofile << "<CellData>" << endl;
    ofile << "</CellData>" << endl;

    // write coordinates
    ofile << "<Coordinates>" << endl;
    ofile << "<DataArray type=\"Float32\" Name=\"Array x\" format=\"ascii\" RangeMin=\"0\" RangeMax=\"" << nx - 1 << "\">" << endl;
    for (int i = 0; i < nx; i++)
    {
        ofile << i << " ";
    }
    ofile << endl;
    ofile << "</DataArray>" << endl;

    ofile << "<DataArray type=\"Float32\" Name=\"Array y\" format=\"ascii\" RangeMin=\"" << iy_start << "\" RangeMax=\"" << iy_end << "\">" << endl;
    for (int i = iy_start; i <= iy_end; i++)
    {
        ofile << i << " ";
    }
    ofile << endl;
    ofile << "</DataArray>" << endl;

    ofile << "<DataArray type=\"Float32\" Name=\"Array z\" format=\"ascii\" RangeMin=\"0\" RangeMax=\"" << nz - 1 << "\">" << endl;
    for (int i = 0; i < nz; i++)
    {
        ofile << i << " ";
    }
    ofile << endl;
    ofile << "</DataArray>" << endl;

    ofile << "</Coordinates>" << endl;
    ofile << "</Piece>" << endl;
    ofile << "</RectilinearGrid>" << endl;
    ofile << "</VTKFile>" << endl;

    ofile.close();
}