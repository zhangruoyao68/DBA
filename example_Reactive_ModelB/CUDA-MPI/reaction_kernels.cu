/* 
 * Model B with chemical reactions -- CUDA-Aware MPI version
 * 
 * 11/28/2024
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
#endif  // HAVE_CUB

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

//#ifdef USE_DOUBLE
typedef double real;
#define MPI_REAL_TYPE MPI_DOUBLE
//#else
//typedef float real;
//#define MPI_REAL_TYPE MPI_FLOAT
//#endif

#define BLKXSIZE 4
#define BLKYSIZE 4
#define BLKZSIZE 4

using namespace std;

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
__global__ void initialize_device(real* __restrict__ const x, real* __restrict__ const y, \
                            real* __restrict__ const z, real* __restrict__ const b, \
                            const real x_init, const real y_init, const real z_init,\
                            const int offset, const int nx, const int my_ny, const int ny, const int nz){
    // Index follows iy*nx*nz + ix*nz + iz
    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    
    //real Radius;
    unsigned int index;

    real amp = 0.05;
    real z_amp = 0.001;

    if ((iy < my_ny) && (ix < nx) && (iz < nz)) {
        index = iy*nx*nz + ix*nz + iz;
        
        // noise of 0.05 around initial values
        x[index] = x_init + amp * (x[index]-0.5); 
		y[index] = y_init + amp * (y[index]-0.5);
		z[index] = z_init + z_amp * (z[index]-0.5);
        b[index] = 1.0 - x[index] - y[index] - z[index];
    }
}

void launch_initialize(real* __restrict__ const x, real* __restrict__ const y, \
                    real* __restrict__ const z, real* __restrict__ const b, \
                    const real x_init, const real y_init, const real z_init,\
                    const int offset, const int nx, const int my_ny, const int ny, const int nz) {

    dim3 dim_grid((nx + BLKXSIZE - 1) / BLKXSIZE,
                (my_ny + BLKYSIZE - 1) / BLKYSIZE, // was ny instead of my_ny
                (nz + BLKZSIZE - 1) / BLKZSIZE);

    initialize_device<BLKXSIZE, BLKYSIZE, BLKZSIZE><<<dim_grid, {BLKXSIZE, BLKYSIZE, BLKZSIZE}>>>(
        x, y, z, b, x_init, y_init, z_init, offset, nx, my_ny, ny, nz);
    CUDA_RT_CALL(cudaGetLastError());
}


/////////////////////////////////////////////////////////////////////////////////////////////////
// Initialization for DBA
/////////////////////////////////////////////////////////////////////////////////////////////////
template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
__global__ void initialize_DBA_device(int* __restrict__ const wakeup, int* __restrict__ const wakeup_next, \
                            real* __restrict__ const x, real* __restrict__ const y, \
                            real* __restrict__ const z, real* __restrict__ const b, \
                            const real x_init, const real y_init, const real z_init,\
                            const int offset, const int nx, const int my_ny, const int ny, const int nz){
    
    int block_index_x = blockIdx.x;
    int block_index_y = blockIdx.y;
    int block_index_z = blockIdx.z;

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
    
        int block_index = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;
        
        wakeup[block_index] = 1;
        wakeup_next[block_index] = 0;
    }

    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;
    
    unsigned int index;

    real amp = 0.05;
    real z_amp = 0.001;

    if ((iy < my_ny) && (ix < nx) && (iz < nz)) {
        index = iy*nx*nz + ix*nz + iz;

        // noise of 0.05 around initial values
        x[index] = x_init + amp * (x[index]-0.5); 
		y[index] = y_init + amp * (y[index]-0.5);
		z[index] = z_init + z_amp * (z[index]-0.5);
        b[index] = 1.0 - x[index] - y[index] - z[index];
    }
}

void launch_initialize_DBA(int* __restrict__ const wakeup, int* __restrict__ const wakeup_next, \
                        real* __restrict__ const x, real* __restrict__ const y, \
                        real* __restrict__ const z, real* __restrict__ const b, \
                        const real x_init, const real y_init, const real z_init,\
                        const int offset, const int nx, const int my_ny, const int ny, const int nz) {

    dim3 dim_grid((nx + BLKXSIZE - 1) / BLKXSIZE,
                (my_ny + BLKYSIZE - 1) / BLKYSIZE, // was ny instead of my_ny
                (nz + BLKZSIZE - 1) / BLKZSIZE);

    printf("Launching initialize_DBA_device with grid (%d, %d, %d)\n", dim_grid.x, dim_grid.y, dim_grid.z);
    initialize_DBA_device<BLKXSIZE, BLKYSIZE, BLKZSIZE><<<dim_grid, {BLKXSIZE, BLKYSIZE, BLKZSIZE}>>>(
        wakeup, wakeup_next, x, y, z, b, x_init, y_init, z_init, \
        offset, nx, my_ny, ny, nz);
    CUDA_RT_CALL(cudaGetLastError());
}


// Gradient in x and z needs to be manually set for periodic BC
// while in y it is taken care of by boundary communication
__device__ real GradientX(real* __restrict__ const a, const int ix, const int iy, const int iz, \
                            const int nx, const int nz){
    int xp = ix + 1;
    int xn = ix - 1;

    if (xp > nx - 1){
        xp = 0;
    }
    if (xn < 0){
        xn = nx - 1;
    }

    real ax = (a[iy*nx*nz + xp*nz + iz] - a[iy*nx*nz + xn*nz + iz]) / (2.0 * 1);

    return ax;
}

__device__ real GradientY(real* __restrict__ const a, const int ix, const int iy, const int iz, \
                            const int nx, const int nz){
    int yp = iy + 1;
    int yn = iy - 1;

    real ay = (a[yp*nx*nz + ix*nz + iz] - a[yn*nx*nz + ix*nz + iz]) / (2.0 * 1);

    return ay;
}

__device__ real GradientZ(real* __restrict__ const a, const int ix, const int iy, const int iz, \
                            const int nx, const int nz){
    int zp = iz + 1;
    int zn = iz - 1;

    if (zp > nz - 1){
        zp = 0;
    }
    if (zn < 0){
        zn = nz - 1;
    }

    real az = (a[iy*nx*nz + ix*nz + zp] - a[iy*nx*nz + ix*nz + zn]) / (2.0 * 1);

    return az;
}

// 2D 9-point Laplacian
__device__ real nine_point_Laplacian(real* __restrict__ const a, const int ix, const int iy, const int iz,\
                            const int nx){
    int xp, xn, yp, yn;

    xp = ix + 1;
    xn = ix - 1;
    yp = iy + 1;
    yn = iy - 1;


    if (xp > nx-1){
        xp = 0;
    }
    if (xn < 0){
        xn = nx-1;
    }


    // for dx = 1
    real result = 1. / 6. * (4. * a[iy * nx + xp] + 4. * a[iy * nx + xn] + \
                            4. * a[yp * nx + ix] + 4. * a[yn * nx + ix] + \
                            a[yp * nx + xp] + a[yn * nx + xn] + \
                            a[yp * nx + xn] + a[yn * nx + xp] + \
                                     - 20.0 * a[iy * nx + ix]);
    return result;
}

// 3D 27-point Laplacian
__device__ real iso_Laplacian(real* __restrict__ const a, const int ix, const int iy, const int iz,\
                            const int nx, const int nz){

    // Index follows iy*nx*nz + ix*nz + iz
    int xp, xn, yp, yn, zp, zn;

    xp = ix + 1;
    xn = ix - 1;
    yp = iy + 1;
    yn = iy - 1;
    zp = iz + 1;
    zn = iz - 1;

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
    }

    // for dx = 1
    real result = 1. * (\
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
__global__ void mu_kernel(real* __restrict__ const x, real* __restrict__ const y, \
                        real* __restrict__ const z, real* __restrict__ const b, \
                        real* __restrict__ const dfdx, real* __restrict__ const dfdy, \
                        real* __restrict__ const dfdz, \
                        const int iy_start, const int iy_end, const int nx, const int nz) {

    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int index;

    real chi_xy = 0;
    real chi_xz = 0;
    real chi_xb = 0;
    real chi_yz = 0;
    real chi_yb = 0;
    real chi_zb = 4.0;

    real rx = 1.0;
    real ry = 1.0;
    real rz = 2.0;

    real epsilonx_sq = 4.;
    real epsilony_sq = 4.;
    real epsilonz_sq = 4.;

    real K = 100.0;

    // calculating mu for CH
    if ((iy < iy_end) && (ix < nx) && (iz < nz)) {
        index = iy*nx*nz + ix*nz + iz;

        //b[index] = 1.0 - x[index] - y[index] - z[index];
        
		dfdx[index] = -1.0 + 1.0/rx + chi_xy * y[index] + chi_xz * z[index] \
					+ chi_xb * b[index] - chi_xb * x[index] - chi_yb * y[index] \
					- chi_zb * z[index] + log(x[index])/rx - log(b[index]) \
					- epsilonx_sq * iso_Laplacian(x,ix,iy,iz,nx,nz);

	    dfdy[index] = -1.0 + 1.0/ry + chi_xy * x[index] + chi_yz * z[index] \
					+ chi_yb * b[index] - chi_xb * x[index] - chi_yb * y[index] \
					- chi_zb * z[index] + log(y[index])/ry - log(b[index]) \
					- epsilony_sq * iso_Laplacian(y,ix,iy,iz,nx,nz);

		dfdz[index] = -1.0 + 1.0/rz + chi_xz * x[index] + chi_yz * y[index] \
					+ chi_zb * b[index] - chi_xb * x[index] - chi_yb * y[index] \
					- chi_zb * z[index] + log(z[index])/rz - log(b[index]) \
					- epsilonz_sq * iso_Laplacian(z,ix,iy,iz,nx,nz) \
					- log(K)/rz;
    }
}

void launch_mu_kernel(real* __restrict__ const x, real* __restrict__ const y, \
                    real* __restrict__ const z, real* __restrict__ const b, \
                    real* __restrict__ const dfdx, real* __restrict__ const dfdy, \
                    real* __restrict__ const dfdz, \
                    const int iy_start, const int iy_end, const int nx, const int nz, cudaStream_t stream) {
    
    dim3 dim_grid((nx + BLKXSIZE - 1) / BLKXSIZE,
                  ((iy_end - iy_start) + BLKYSIZE - 1) / BLKYSIZE,
                  (nz + BLKZSIZE - 1) / BLKZSIZE);
    
    //printf("Launching mu_kernel with grid (%d, %d, %d)\n", dim_grid.x, dim_grid.y, dim_grid.z);
    mu_kernel<BLKXSIZE, BLKYSIZE, BLKZSIZE><<<dim_grid, {BLKXSIZE, BLKYSIZE, BLKZSIZE}, 0, stream>>>(
        x, y, z, b, dfdx, dfdy, dfdz, iy_start, iy_end, nx, nz);
    CUDA_RT_CALL(cudaGetLastError());
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
__global__ void mu_DBA_kernel(int* __restrict__ const wakeup, \
                        real* __restrict__ const x, real* __restrict__ const y, \
                        real* __restrict__ const z, real* __restrict__ const b, \
                        real* __restrict__ const dfdx, real* __restrict__ const dfdy, \
                        real* __restrict__ const dfdz, \
                        real* __restrict__ const metric, \
                        const int iy_start, const int iy_end, const int nx, const int nz) {

    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ int wake_up_s;

    real chi_xy = 0;
    real chi_xz = 0;
    real chi_xb = 0;
    real chi_yz = 0;
    real chi_yb = 0;
    real chi_zb = 4.0;

    real rx = 1.0;
    real ry = 1.0;
    real rz = 2.0;

    real epsilonx_sq = 4.;
    real epsilony_sq = 4.;
    real epsilonz_sq = 4.;

    real K = 100.0;

    unsigned int index;
    real lap_z;

    int block_index_x = blockIdx.x;
    int block_index_y = blockIdx.y;
    int block_index_z = blockIdx.z;

    // Read the wakeup status into wake_up_s
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
        //printf("blockIdx.x = %d, blockIdx.y = %d, blockIdx.z = %d\n", blockIdx.x, blockIdx.y, blockIdx.z);
        //printf("block_index_x = %d, block_index_y = %d, block_index_z = %d\n", block_index_x, block_index_y, block_index_z);
    
        int block_index = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;
        
        wake_up_s = wakeup[block_index];
        //wakeup_next[block_index] = 0;
    }

    // Synchronize threads in threadblock before using the shared memory
	__syncthreads();

    // calculating mu for CH
    if (wake_up_s && (iy < iy_end) && (ix < nx) && (iz < nz)) {
        index = iy*nx*nz + ix*nz + iz;
        
        //b[index] = 1.0 - x[index] - y[index] - z[index];
        
		dfdx[index] = -1.0 + 1.0/rx + chi_xy * y[index] + chi_xz * z[index] \
					+ chi_xb * b[index] - chi_xb * x[index] - chi_yb * y[index] \
					- chi_zb * z[index] + log(x[index])/rx - log(b[index]) \
					- epsilonx_sq * iso_Laplacian(x,ix,iy,iz,nx,nz);

	    dfdy[index] = -1.0 + 1.0/ry + chi_xy * x[index] + chi_yz * z[index] \
					+ chi_yb * b[index] - chi_xb * x[index] - chi_yb * y[index] \
					- chi_zb * z[index] + log(y[index])/ry - log(b[index]) \
					- epsilony_sq * iso_Laplacian(y,ix,iy,iz,nx,nz);
        
        lap_z = iso_Laplacian(z,ix,iy,iz,nx,nz);

		dfdz[index] = -1.0 + 1.0/rz + chi_xz * x[index] + chi_yz * y[index] \
					+ chi_zb * b[index] - chi_xb * x[index] - chi_yb * y[index] \
					- chi_zb * z[index] + log(z[index])/rz - log(b[index]) \
					- epsilonz_sq * lap_z \
					- log(K)/rz;

        // Update metric
        metric[index] = fabs(lap_z);
    }
}

void launch_mu_DBA_kernel(int* __restrict__ const wakeup, \
                        real* __restrict__ const x, real* __restrict__ const y, \
                        real* __restrict__ const z, real* __restrict__ const b, \
                        real* __restrict__ const dfdx, real* __restrict__ const dfdy, \
                        real* __restrict__ const dfdz, \
                        real* __restrict__ const metric, \
                        const int iy_start, const int iy_end, const int nx, const int nz, cudaStream_t stream) {
    
    dim3 dim_grid((nx + BLKXSIZE - 1) / BLKXSIZE,
                  ((iy_end - iy_start) + BLKYSIZE - 1) / BLKYSIZE,
                  (nz + BLKZSIZE - 1) / BLKZSIZE);
    
    //printf("Launching mu_kernel with grid (%d, %d, %d)\n", dim_grid.x, dim_grid.y, dim_grid.z);
    mu_DBA_kernel<BLKXSIZE, BLKYSIZE, BLKZSIZE><<<dim_grid, {BLKXSIZE, BLKYSIZE, BLKZSIZE}, 0, stream>>>(
        wakeup, x, y, z, b, dfdx, dfdy, dfdz, metric, iy_start, iy_end, nx, nz);
    CUDA_RT_CALL(cudaGetLastError());
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
__global__ void update_kernel(real* __restrict__ const x_old, real* __restrict__ const y_old, \
                            real* __restrict__ const z_old, real* __restrict__ const b, \
                            real* __restrict__ const x_new, real* __restrict__ const y_new, \
                            real* __restrict__ const z_new, \
                            real* __restrict__ const dfdx, real* __restrict__ const dfdy, \
                            real* __restrict__ const dfdz, \
                            real* __restrict__ const sum_c, \
                            const int iy_start, const int iy_end, const int nx, const int nz, \
                            const bool calculate_sum) {
#ifdef HAVE_CUB
    typedef cub::BlockReduce<real, BLOCK_DIM_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_DIM_Y, BLOCK_DIM_Z>
        BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
#endif  // HAVE_CUB
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    real local_sum_c = 0.;

    real dt = 0.0005;

    real k_0 = 1.0;

    real n = 1.0;
    real m = 1.0;
    real vx = 1.0;
    real vy = 1.0;
    real vz = 2.0;

    real Mobility_x = 1.0;
    real Mobility_y = 1.0;
    real Mobility_z = 1.0;

    real R;

    unsigned int index;
    // calculating
    if ((iy < iy_end) && (ix < nx) && (iz < nz)) {

        index = iy*nx*nz + ix*nz + iz;

        // Update fields
        R = k_0 * (exp(n * vx * dfdx[index] + m * vy * dfdy[index]) \
                  -exp(vz * dfdz[index]));

        x_new[index] = x_old[index] + dt * vx * (Mobility_x * iso_Laplacian(dfdx,ix,iy,iz,nx,nz) - n * R );
        
        y_new[index] = y_old[index] + dt * vy * (Mobility_y * iso_Laplacian(dfdy,ix,iy,iz,nx,nz) - m * R );
        
        z_new[index] = z_old[index] + dt * vz * (Mobility_z * iso_Laplacian(dfdz,ix,iy,iz,nx,nz) + R );

		b[index] = 1.0 - x_new[index] - y_new[index] - z_new[index];

        if (calculate_sum) {
            local_sum_c += b[index];
        }
    }

    if (calculate_sum) {
#ifdef HAVE_CUB
        real block_sum_c = BlockReduce(temp_storage).Sum(local_sum_c);
        if (0 == threadIdx.y && 0 == threadIdx.x && 0 == threadIdx.z){
            atomicAdd(sum_c, block_sum_c);
        }
#else
        atomicAdd(sum_c, local_sum_c);
#endif  // HAVE_CUB

    }
}

void launch_update_kernel(real* __restrict__ const x_old, real* __restrict__ const y_old, \
                        real* __restrict__ const z_old, real* __restrict__ const b, \
                        real* __restrict__ const x_new, real* __restrict__ const y_new, \
                        real* __restrict__ const z_new, \
                        real* __restrict__ const dfdx, real* __restrict__ const dfdy, \
                        real* __restrict__ const dfdz, \
                        real* __restrict__ const sum_c, \
                        const int iy_start, const int iy_end, const int nx, const int nz, \
                        const bool calculate_sum, cudaStream_t stream) {

    dim3 dim_grid((nx + BLKXSIZE - 1) / BLKXSIZE,
                ((iy_end - iy_start) + BLKYSIZE - 1) / BLKYSIZE,
                (nz + BLKZSIZE - 1) / BLKZSIZE);

    update_kernel<BLKXSIZE, BLKYSIZE, BLKZSIZE><<<dim_grid, {BLKXSIZE, BLKYSIZE, BLKZSIZE}, 0, stream>>>(
                x_old, y_old, z_old, b, x_new, y_new, z_new, \
                dfdx, dfdy, dfdz, sum_c, \
                iy_start, iy_end, nx, nz, calculate_sum);
    CUDA_RT_CALL(cudaGetLastError());
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
__global__ void update_DBA_kernel(int* __restrict__ const wakeup, \
                                real* __restrict__ const x_old, real* __restrict__ const y_old, \
                                real* __restrict__ const z_old, real* __restrict__ const b, \
                                real* __restrict__ const x_new, real* __restrict__ const y_new, \
                                real* __restrict__ const z_new, \
                                real* __restrict__ const dfdx, real* __restrict__ const dfdy, \
                                real* __restrict__ const dfdz, \
                                real* __restrict__ const sum_c, int* __restrict__ const sum_wakeup, \
                                const int iy_start, const int iy_end, const int nx, const int nz, \
                                const bool calculate_sum) {
#ifdef HAVE_CUB
    typedef cub::BlockReduce<real, BLOCK_DIM_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_DIM_Y, BLOCK_DIM_Z>
        BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage_c;
    __shared__ typename BlockReduce::TempStorage temp_storage_wakeup;
#endif  // HAVE_CUB
    int iy = blockIdx.y * blockDim.y + threadIdx.y + iy_start;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    unsigned int index;
    __shared__ int wake_up_s;

    real local_sum_c = 0.;
    int local_sum_wakeup = 0;

    real dt = 0.0005;
    real k_0 = 1.0;
    real n = 1.0;
    real m = 1.0;
    real vx = 1.0;
    real vy = 1.0;
    real vz = 2.0;
    real Mobility_x = 1.0;
    real Mobility_y = 1.0;
    real Mobility_z = 1.0;
    real R;

    int block_index_x = blockIdx.x;
    int block_index_y = blockIdx.y;
    int block_index_z = blockIdx.z;

    // Read the wakeup status into wake_up_s
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
        int block_index = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;
        
        wake_up_s = wakeup[block_index];

        if (calculate_sum) {
            local_sum_wakeup += wake_up_s;
        }
    }

    // Synchronize threads in threadblock before using the shared memory
	__syncthreads();

    // calculating update c
    if (wake_up_s && (iy < iy_end) && (ix < nx) && (iz < nz)) {

        index = iy*nx*nz + ix*nz + iz;

        R = k_0 * (exp(n * vx * dfdx[index] + m * vy * dfdy[index]) \
                        -exp(vz * dfdz[index]));

        x_new[index] = x_old[index] + dt * vx * (Mobility_x * iso_Laplacian(dfdx,ix,iy,iz,nx,nz) - n * R );
        
        y_new[index] = y_old[index] + dt * vy * (Mobility_y * iso_Laplacian(dfdy,ix,iy,iz,nx,nz) - m * R );
        
        z_new[index] = z_old[index] + dt * vz * (Mobility_z * iso_Laplacian(dfdz,ix,iy,iz,nx,nz) + R );

		b[index] = 1.0 - x_new[index] - y_new[index] - z_new[index];

        if (calculate_sum) {
            local_sum_c += b[index];
        }
    }

    if (calculate_sum) {
#ifdef HAVE_CUB
        real block_sum_c = BlockReduce(temp_storage_c).Sum(local_sum_c);
        //int block_sum_wakeup = BlockReduce(temp_storage_wakeup).Sum(local_sum_wakeup);
        if (0 == threadIdx.y && 0 == threadIdx.x && 0 == threadIdx.z){
            atomicAdd(sum_c, block_sum_c);
            atomicAdd(sum_wakeup, local_sum_wakeup);
            //printf("have CUB\n");
        }
#else
        atomicAdd(sum_c, local_sum_c);
        atomicAdd(sum_wakeup, local_sum_wakeup);
#endif  // HAVE_CUB

    }
}

void launch_update_DBA_kernel(int* __restrict__ const wakeup, \
                            real* __restrict__ const x_old, real* __restrict__ const y_old, \
                            real* __restrict__ const z_old, real* __restrict__ const b, \
                            real* __restrict__ const x_new, real* __restrict__ const y_new, \
                            real* __restrict__ const z_new, \
                            real* __restrict__ const dfdx, real* __restrict__ const dfdy, \
                            real* __restrict__ const dfdz, \
                            real* __restrict__ const sum_c, int* __restrict__ const sum_wakeup, \
                            const int iy_start, const int iy_end, const int nx, const int nz, \
                            const bool calculate_sum, cudaStream_t stream) {

    dim3 dim_grid((nx + BLKXSIZE - 1) / BLKXSIZE,
                ((iy_end - iy_start) + BLKYSIZE - 1) / BLKYSIZE,
                (nz + BLKZSIZE - 1) / BLKZSIZE);

    update_DBA_kernel<BLKXSIZE, BLKYSIZE, BLKZSIZE><<<dim_grid, {BLKXSIZE, BLKYSIZE, BLKZSIZE}, 0, stream>>>(
                wakeup, x_old, y_old, z_old, b, x_new, y_new, z_new, \
                dfdx, dfdy, dfdz, sum_c, sum_wakeup, \
                iy_start, iy_end, nx, nz, calculate_sum);
    CUDA_RT_CALL(cudaGetLastError());
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
__global__ void wakeup_next_kernel(int* __restrict__ const wakeup, int* __restrict__ const wakeup_next,\
                                real* __restrict__ const metric, \
                                const int nx, const int my_ny, const int nz) {

    int iy = blockIdx.y * blockDim.y + threadIdx.y;
    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int iz = blockIdx.z * blockDim.z + threadIdx.z;

    __shared__ int wake_up_s, wake_up_s_next;

    real metric_eps = 1e-10;

    unsigned int index;

    int block_index_x = blockIdx.x;
    int block_index_y = blockIdx.y;
    int block_index_z = blockIdx.z;
    int block_index;

    // Read the wakeup status into wake_up_s
    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
    
        block_index = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;
        
        wake_up_s = wakeup[block_index];
        wakeup[block_index] = 0;
        wake_up_s_next = 0;
    }

    // Synchronize threads in threadblock before using the shared memory
	__syncthreads();

    // Evaluating activation criteria
    if (wake_up_s && (iy < my_ny) && (ix < nx) && (iz < nz)) {

        index = iy*nx*nz + ix*nz + iz;
        if (metric[index] > metric_eps){ // additional criterion: || (timestep < 10000)
            //printf("metric[%d] = %f\n", index, metric[index]);
			wake_up_s_next = 1;
	    }

        // Synchronize threads in threadblock before using the shared memory
        __syncthreads();

        if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
            block_index = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;
            wakeup_next[block_index] = wake_up_s_next;
        }
    }

    
}

template <int BLOCK_DIM_X, int BLOCK_DIM_Y, int BLOCK_DIM_Z>
__global__ void wakeup_neighbor_kernel(int* __restrict__ const wakeup, int* __restrict__ const wakeup_next){

    int block_index_x = blockIdx.x;
    int block_index_y = blockIdx.y;
    int block_index_z = blockIdx.z;
    int block_index = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;

    if (wakeup_next[block_index]){
        int block_xm = (block_index_x - 1 >= 0) ? block_index_x - 1 : gridDim.x - 1;
		//int block_ym = (block_index_y - 1 >= 0) ? block_index_y - 1 : gridDim.y - 1;
		int block_zm = (block_index_z - 1 >= 0) ? block_index_z - 1 : gridDim.z - 1;
		int block_xp = (block_index_x + 1 < gridDim.x) ? block_index_x + 1 : 0;
		//int block_yp = (block_index_y + 1 < gridDim.y) ? block_index_y + 1 : 0;
		int block_zp = (block_index_z + 1 < gridDim.z) ? block_index_z + 1 : 0;

        // do not apply periodic BC in y direction
        int block_ym = (block_index_y - 1 >= 0) ? block_index_y - 1 : block_index_y;
        int block_yp = (block_index_y + 1 < gridDim.y) ? block_index_y + 1 : block_index_y;

		int block_index_xm = block_index_y * gridDim.x * gridDim.z + block_xm * gridDim.z + block_index_z;
		int block_index_xp = block_index_y * gridDim.x * gridDim.z + block_xp * gridDim.z + block_index_z;
		int block_index_ym = block_ym * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;
		int block_index_yp = block_yp * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_index_z;
		int block_index_zm = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_zm;
		int block_index_zp = block_index_y * gridDim.x * gridDim.z + block_index_x * gridDim.z + block_zp;

        wakeup[block_index] = 1;
		wakeup[block_index_xm] = 1;
		wakeup[block_index_xp] = 1;
		wakeup[block_index_ym] = 1;
		wakeup[block_index_yp] = 1;
		wakeup[block_index_zm] = 1;
		wakeup[block_index_zp] = 1;
    }
}

void launch_wake_DBA_kernel(int* __restrict__ const wakeup, int* __restrict__ const wakeup_next,\
                real* __restrict__ const metric, \
                const int nx, const int my_ny, const int nz, cudaStream_t stream)
{
    
    dim3 dim_grid((nx + BLKXSIZE - 1) / BLKXSIZE,
                  (my_ny + BLKYSIZE - 1) / BLKYSIZE,
                  (nz + BLKZSIZE - 1) / BLKZSIZE);
    
    //printf("Launching mu_kernel with grid (%d, %d, %d)\n", dim_grid.x, dim_grid.y, dim_grid.z);
    wakeup_next_kernel<BLKXSIZE, BLKYSIZE, BLKZSIZE><<<dim_grid, {BLKXSIZE, BLKYSIZE, BLKZSIZE}, 0, stream>>>(
        wakeup, wakeup_next, metric, nx, my_ny, nz);
    CUDA_RT_CALL(cudaGetLastError());

    // wake up neighbors
    wakeup_neighbor_kernel<BLKXSIZE, BLKYSIZE, BLKZSIZE><<<dim_grid, {BLKXSIZE, BLKYSIZE, BLKZSIZE}, 0, stream>>>(
        wakeup, wakeup_next);
    CUDA_RT_CALL(cudaGetLastError());
}

void integral_c(real* __restrict__ const c, int nx, int iy_start, int iy_end, int nz){
    real sum = 0.;

    for (int j = iy_start; j < iy_end; j++) {
        for (int i = 0; i < nx; i++) {
            for (int k = 0; k < nz; k++) {
                sum += c[j*nx*nz + i*nz + k];
            }
        }
    }
    printf("Sum is %8.5f\n", sum);
}

void integral_wakeup(int* __restrict__ const wakeup, int nx, int iy_start, int iy_end, int nz){
    unsigned int sum = 0;

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j <= iy_end-iy_start; j++) {
        //for (int j = iy_start; j <= iy_end; j++) {
            for (int i = 0; i < nx; i++) {
                int block_x = i / BLKXSIZE;
                int block_y = j / BLKYSIZE;
                int block_z = k / BLKZSIZE;

                int index_block = block_y * ((nx + BLKXSIZE - 1) / BLKXSIZE) * ((nz + BLKZSIZE - 1) / BLKZSIZE) + block_x * ((nz + BLKZSIZE - 1) / BLKZSIZE) + block_z;

                sum += wakeup[index_block];
            }
        }
    }

    printf("Total number of active blocks: %d\n", sum/64);
}


void write_output_pvtr(int total_rank, int t, int nx, int ny, int nz)
{
    string name = "output_" + to_string(t) + ".pvtr";
    ofstream ofile(name);

    int chunk_size;
    int chunk_size_low = (ny - 2) / total_rank;
    int chunk_size_high = chunk_size_low + 1;
    int num_ranks_low = total_rank * chunk_size_low + total_rank - (ny - 2);  // Number of ranks with chunk_size = chunk_size_low
    int iy_start_global, iy_end_global;
    
    // pvtr preamble
    ofile << "<VTKFile type=\"PRectilinearGrid\" version=\"2.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">" << endl;
    ofile << "<PRectilinearGrid WholeExtent=\"0 " << nx-1 << " 1 " << ny-2 << " 0 " << nz-1 << "\" GhostLevel=\"0\">" << endl;

    // write names of arrays
    ofile << "<PPointData>" << endl;
    ofile << "<PDataArray type=\"Float32\" Name=\"X\"/>" << endl;
    ofile << "<PDataArray type=\"Float32\" Name=\"Y\"/>" << endl;
    ofile << "<PDataArray type=\"Float32\" Name=\"Z\"/>" << endl;
    ofile << "<PDataArray type=\"Float32\" Name=\"B\"/>" << endl;
    ofile << "<PDataArray type=\"Int8\" Name=\"active\"/>" << endl;
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
        } else{
            chunk_size = chunk_size_high;
        }

        if (rank < num_ranks_low) {
            iy_start_global = rank * chunk_size_low + 1;
        } else {
            iy_start_global =
                num_ranks_low * chunk_size_low + (rank - num_ranks_low) * chunk_size_high + 1;
        }
        iy_end_global = iy_start_global + chunk_size - 1; 
        
        ofile << "<Piece Extent=\"0 " << nx-1 << " " << iy_start_global - 1\
                << " " << iy_end_global + 1\
                << " 0 " << nz-1 << "\" Source=\"output_"<< t << "_" << rank << ".vtr\"/>" << endl;
    }

    ofile << "</PRectilinearGrid>" << endl;
    ofile << "</VTKFile>" << endl;

    ofile.close();
}

void write_output_vtr(real* __restrict__ const x, real* __restrict__ const y, \
                    real* __restrict__ const z, real* __restrict__ const b, \
                    int* __restrict__ const wakeup, int t, int local_rank, \
                    int nx, int iy_start, int iy_end, int nz){
    string name = "output_" + to_string(t) + "_" + to_string(local_rank) + ".vtr";
    ofstream ofile(name);
    int FlattenedID;
    int block_x, block_y, block_z;
    int index_block;

    // vtr preamble
    ofile << "<VTKFile type=\"RectilinearGrid\" version=\"2.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">" << endl;
    ofile << "<RectilinearGrid WholeExtent=\"0 " << nx-1 << " " << iy_start << " " << iy_end << " 0 " << nz-1 << "\">" << endl;
    ofile << "<Piece Extent=\"0 " << nx-1 << " " << iy_start << " " << iy_end << " 0 " << nz-1 << "\">" << endl;

    //write field data
    ofile << "<PointData>" << endl;

    // write x data
    ofile << "<DataArray type=\"Float32\" Name=\"X\" format=\"ascii\" RangeMin=\"-1.0\" RangeMax=\"1.00\">" << endl;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j <= iy_end-iy_start; j++) {
        //for (int j = iy_start; j <= iy_end; j++) {
            for (int i = 0; i < nx; i++) {
                FlattenedID = j*nx*nz + i*nz + k;
                ofile << std::fixed << x[FlattenedID] << " ";
            }
        }
    }
    ofile << endl;
    ofile << "</DataArray>" << endl;

    // write y data
    ofile << "<DataArray type=\"Float32\" Name=\"Y\" format=\"ascii\" RangeMin=\"-1.0\" RangeMax=\"1.00\">" << endl;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j <= iy_end-iy_start; j++) {
        //for (int j = iy_start; j <= iy_end; j++) {
            for (int i = 0; i < nx; i++) {
                FlattenedID = j*nx*nz + i*nz + k;
                ofile << std::fixed << y[FlattenedID] << " ";
            }
        }
    }
    ofile << endl;
    ofile << "</DataArray>" << endl;

    // write z data
    ofile << "<DataArray type=\"Float32\" Name=\"Z\" format=\"ascii\" RangeMin=\"-1.0\" RangeMax=\"1.00\">" << endl;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j <= iy_end-iy_start; j++) {
        //for (int j = iy_start; j <= iy_end; j++) {
            for (int i = 0; i < nx; i++) {
                FlattenedID = j*nx*nz + i*nz + k;
                ofile << std::fixed << z[FlattenedID] << " ";
            }
        }
    }
    ofile << endl;
    ofile << "</DataArray>" << endl;

    // write b data
    ofile << "<DataArray type=\"Float32\" Name=\"B\" format=\"ascii\" RangeMin=\"-1.0\" RangeMax=\"1.00\">" << endl;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j <= iy_end-iy_start; j++) {
        //for (int j = iy_start; j <= iy_end; j++) {
            for (int i = 0; i < nx; i++) {
                FlattenedID = j*nx*nz + i*nz + k;
                ofile << std::fixed << b[FlattenedID] << " ";
            }
        }
    }
    ofile << endl;
    ofile << "</DataArray>" << endl;

    // write wakeup block data
    ofile << "<DataArray type=\"Int8\" Name=\"active\" format=\"ascii\" RangeMin=\"-1.0\" RangeMax=\"1.00\">" << endl;
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j <= iy_end-iy_start; j++) {
        //for (int j = iy_start; j <= iy_end; j++) {
            for (int i = 0; i < nx; i++) {
                block_x = i / BLKXSIZE;
                block_y = j / BLKYSIZE;
                block_z = k / BLKZSIZE;

                index_block = block_y * ((nx + BLKXSIZE - 1) / BLKXSIZE) * ((nz + BLKZSIZE - 1) / BLKZSIZE) + block_x * ((nz + BLKZSIZE - 1) / BLKZSIZE) + block_z;

                ofile << wakeup[index_block] << " ";
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
    ofile << "<DataArray type=\"Float32\" Name=\"Array x\" format=\"ascii\" RangeMin=\"0\" RangeMax=\"" << nx-1 <<"\">" << endl;
    for (int i = 0; i < nx; i++) {
        ofile << i << " ";
    }
    ofile << endl;
    ofile << "</DataArray>" << endl;

    ofile << "<DataArray type=\"Float32\" Name=\"Array y\" format=\"ascii\" RangeMin=\"" << iy_start << "\" RangeMax=\"" << iy_end <<"\">" << endl;
    for (int i = iy_start; i <= iy_end; i++) {
        ofile << i << " ";
    }
    ofile << endl;
    ofile << "</DataArray>" << endl;

    ofile << "<DataArray type=\"Float32\" Name=\"Array z\" format=\"ascii\" RangeMin=\"0\" RangeMax=\"" << nz-1 <<"\">" << endl;
    for (int i = 0; i < nz; i++) {
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