/*
 * Compressible Euler equation: KH instability
 * 2D Version, Parallel with CUDA
 * 
 */

#include "cuda_runtime.h"
#include <curand.h>
#include "device_launch_parameters.h"
#include <iostream>
#include <random>
#include <fstream>

#include <thrust/extrema.h>
#include <thrust/device_ptr.h>

//#include <cmath>
//#define _USE_MATH_DEFINES
#include <math.h>
#include <ctime>
#include <cstdlib>
#include <string>
#include <algorithm>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>

dim3 dimGrid3D(16, 16, 1); // adjust to 2d grid
dim3 dimBlock3D(8, 8, 1);

// for cuda error checking
#define CUDACHECK(x) cudacheck((x), __FILE__, __LINE__)
inline void cudacheck(cudaError_t err, std::string const file, int const line) {
	if (err != cudaSuccess) {
		std::cout << "Cuda error: " << cudaGetErrorString(err) << " at line " << line << " in file " << file << std::endl;
		exit(1);
	}
}

// For cuda error check
#define CheckLastCudaError() checklastcudaerror(__FILE__, __LINE__)
inline void checklastcudaerror(std::string const file, int const line)
{
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		std::cout << "Cuda error: " << cudaGetErrorString(err) << " at line " << line << " in file " << file << std::endl;
		exit(2);
	}
}

// check error for cuRand calls
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

class Variable
{
	public:
		double* rho, *p, *vx, *vy;
        double* Mass, *Momx, *Momy, *Energy;
        double* rho_dx, *rho_dy, *p_dx, *p_dy, *vx_dx, *vx_dy, *vy_dx, *vy_dy;
        double* rho_prime, *vx_prime, *vy_prime, *p_prime;
		double* flux_Mass_x, *flux_Momx_x, *flux_Momy_x, *flux_Energy_x;
        double* flux_Mass_y, *flux_Momx_y, *flux_Momy_y, *flux_Energy_y;
        double* metric1, *metric2, *metric3, *metric4;
		int* wake_1, * wake_1_next, * wake_up_T, * wake_up_T_next;		//parameter controlling mode of blocks
	double* change_Mass, *change_Momx, *change_Momy, *change_Energy;
};

class InputPara
{
	public:
        int Nx, Ny, Nz;
		int t_total, t_freq;
		double dx, dy, dz, dt;
        int mode;
        double Metric_eps;
        int blockNum_x, blockNum_y, blockNum_z;

		void print_input()
		{
			printf("--------------Input parameters--------------\n");
			printf("Size of system: %d, %d, %d\n", Nx, Ny, Nz);
			printf("totalTime = %d,	printFreq = %d, dt = %f\n", t_total, t_freq, dt);
			printf("dx = %lf, dy = %lf, dz =%lf\n", dx, dy, dz);
            printf("Mode = %d (0 for regular while 1 for wake-up mode)\n", mode);
            printf("Metric_eps = %lf\n", Metric_eps);
		}
};

//Read input parameters
void ReadInput(std::string InputFileName, InputPara& InputP)
{
	std::ifstream infile;

	/* Open input file */
	infile.open(InputFileName);
	if (!infile.is_open())
	{
		std::cout << "!!!!!Can not open" << InputFileName << "!! Exit!!!!!!!!" << std::endl;
		exit(1);
	}

	std::string space;
	/* Read all input */
	std::cout << "Reading input from " << InputFileName << std::endl;
    infile >> InputP.Nx >> InputP.Ny >> InputP.Nz;
	std::getline(infile, space);
	infile >> InputP.t_total >> InputP.t_freq >> InputP.dt;
	std::getline(infile, space);
	infile >> InputP.dx >> InputP.dy >> InputP.dz;
    std::getline(infile, space);
    infile >> InputP.mode;
    std::getline(infile, space);
	infile >> InputP.Metric_eps;

	/* Close input file */
	infile.close();
	std::cout << "Done with input reading." << std::endl;

    InputP.blockNum_x = (InputP.Nx - 1) / dimBlock3D.x + 1;
	InputP.blockNum_y = (InputP.Ny - 1) / dimBlock3D.y + 1;
	InputP.blockNum_z = (InputP.Nz - 1) / dimBlock3D.z + 1;

	InputP.print_input();
}

//Allocate memory
void Allocate(Variable & Var, InputPara InputP)
{
	size_t size = InputP.Nx * InputP.Ny * InputP.Nz * sizeof(double);
	size_t block_size = InputP.blockNum_x * InputP.blockNum_y * InputP.blockNum_z * sizeof(int);

    // Primitive variables
	CUDACHECK(cudaMallocManaged(&Var.rho, size));
    CUDACHECK(cudaMallocManaged(&Var.p, size));
    CUDACHECK(cudaMallocManaged(&Var.vx, size));
    CUDACHECK(cudaMallocManaged(&Var.vy, size));

    // Conserved variables
    CUDACHECK(cudaMallocManaged(&Var.Mass, size));
    CUDACHECK(cudaMallocManaged(&Var.Momx, size));
    CUDACHECK(cudaMallocManaged(&Var.Momy, size));
    CUDACHECK(cudaMallocManaged(&Var.Energy, size));

    // Gradients
    CUDACHECK(cudaMallocManaged(&Var.rho_dx, size));
    CUDACHECK(cudaMallocManaged(&Var.rho_dy, size));
    CUDACHECK(cudaMallocManaged(&Var.p_dx, size));
    CUDACHECK(cudaMallocManaged(&Var.p_dy, size));
    CUDACHECK(cudaMallocManaged(&Var.vx_dx, size));
    CUDACHECK(cudaMallocManaged(&Var.vx_dy, size));
    CUDACHECK(cudaMallocManaged(&Var.vy_dx, size));
    CUDACHECK(cudaMallocManaged(&Var.vy_dy, size));

    // Extrapolated variables
    CUDACHECK(cudaMallocManaged(&Var.rho_prime, size));
    CUDACHECK(cudaMallocManaged(&Var.vx_prime, size));
    CUDACHECK(cudaMallocManaged(&Var.vy_prime, size));
    CUDACHECK(cudaMallocManaged(&Var.p_prime, size));

    // Fluxes
    CUDACHECK(cudaMallocManaged(&Var.flux_Mass_x, size));
    CUDACHECK(cudaMallocManaged(&Var.flux_Momx_x, size));
    CUDACHECK(cudaMallocManaged(&Var.flux_Momy_x, size));
    CUDACHECK(cudaMallocManaged(&Var.flux_Energy_x, size));
    CUDACHECK(cudaMallocManaged(&Var.flux_Mass_y, size));
    CUDACHECK(cudaMallocManaged(&Var.flux_Momx_y, size));
    CUDACHECK(cudaMallocManaged(&Var.flux_Momy_y, size));
    CUDACHECK(cudaMallocManaged(&Var.flux_Energy_y, size));

    // Metric
    CUDACHECK(cudaMallocManaged(&Var.metric1, size));
    CUDACHECK(cudaMallocManaged(&Var.metric2, size));
    CUDACHECK(cudaMallocManaged(&Var.metric3, size));
    CUDACHECK(cudaMallocManaged(&Var.metric4, size));


    // Wake-up mode
	CUDACHECK(cudaMallocManaged(&Var.wake_1, block_size));
	CUDACHECK(cudaMallocManaged(&Var.wake_1_next, block_size));
    CUDACHECK(cudaMallocManaged(&Var.wake_up_T, block_size));
    CUDACHECK(cudaMallocManaged(&Var.wake_up_T_next, block_size));
    
    // Enforced conservation
    CUDACHECK(cudaMallocManaged(&Var.change_Mass, size));
    CUDACHECK(cudaMallocManaged(&Var.change_Energy, size));
    CUDACHECK(cudaMallocManaged(&Var.change_Momx, size));
    CUDACHECK(cudaMallocManaged(&Var.change_Momy, size));
}

//Free memory
void FreeMemory(Variable & Var)
{
	CUDACHECK(cudaFree(Var.rho));
    CUDACHECK(cudaFree(Var.p));
    CUDACHECK(cudaFree(Var.vx));
    CUDACHECK(cudaFree(Var.vy));
    CUDACHECK(cudaFree(Var.Mass));
    CUDACHECK(cudaFree(Var.Energy));
    CUDACHECK(cudaFree(Var.Momx));
    CUDACHECK(cudaFree(Var.Momy));
    CUDACHECK(cudaFree(Var.rho_dx));
    CUDACHECK(cudaFree(Var.rho_dy));
    CUDACHECK(cudaFree(Var.p_dx));
    CUDACHECK(cudaFree(Var.p_dy));
    CUDACHECK(cudaFree(Var.vx_dx));
    CUDACHECK(cudaFree(Var.vx_dy));
    CUDACHECK(cudaFree(Var.vy_dx));
    CUDACHECK(cudaFree(Var.vy_dy));
    CUDACHECK(cudaFree(Var.rho_prime));
    CUDACHECK(cudaFree(Var.vx_prime));
    CUDACHECK(cudaFree(Var.vy_prime));
    CUDACHECK(cudaFree(Var.p_prime));
    CUDACHECK(cudaFree(Var.flux_Mass_x));
    CUDACHECK(cudaFree(Var.flux_Energy_x));
    CUDACHECK(cudaFree(Var.flux_Momx_x));
    CUDACHECK(cudaFree(Var.flux_Momy_x));
    CUDACHECK(cudaFree(Var.flux_Mass_y));
    CUDACHECK(cudaFree(Var.flux_Energy_y));
    CUDACHECK(cudaFree(Var.flux_Momx_y));
    CUDACHECK(cudaFree(Var.flux_Momy_y));
    CUDACHECK(cudaFree(Var.metric1));
    CUDACHECK(cudaFree(Var.metric2));
    CUDACHECK(cudaFree(Var.metric3));
    CUDACHECK(cudaFree(Var.metric4));
    CUDACHECK(cudaFree(Var.wake_1));
    CUDACHECK(cudaFree(Var.wake_1_next));
    CUDACHECK(cudaFree(Var.wake_up_T));
    CUDACHECK(cudaFree(Var.wake_up_T_next));
    CUDACHECK(cudaFree(Var.change_Mass));
    CUDACHECK(cudaFree(Var.change_Energy));
    CUDACHECK(cudaFree(Var.change_Momx));
    CUDACHECK(cudaFree(Var.change_Momy));
}

__device__ double Grad_sq(double* c, int x, int y, int z, InputPara InputP)
{
	int xm, xp, ym, yp, zm, zp;
	xm = (x - 1 >= 0) ? x - 1 : InputP.Nx - 1;
	xp = (x + 1 < InputP.Nx) ? x + 1 : 0;
	ym = (y - 1 >= 0) ? y - 1 : InputP.Ny - 1;
	yp = (y + 1 < InputP.Ny) ? y + 1 : 0;
	zm = (z - 1 >= 0) ? z - 1 : InputP.Nz - 1;
	zp = (z + 1 < InputP.Nz) ? z + 1 : 0;
    
    // Coalesced memory access
	//int index_xyz = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;	
	int index_xm = xm + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_xp = xp + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_ym = x + ym * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_yp = x + yp * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_zm = x + y * InputP.Nx + zm * InputP.Nx * InputP.Ny;
	int index_zp = x + y * InputP.Nx + zp * InputP.Nx * InputP.Ny;

	double dpx = (c[index_xp] - c[index_xm]) / 2.0 / InputP.dx;
	double dpy = (c[index_yp] - c[index_ym]) / 2.0 / InputP.dy;
	double dpz = (c[index_zp] - c[index_zm]) / 2.0 / InputP.dz;
	
	return dpx*dpx + dpy*dpy + dpz*dpz;
}

__device__ double Grad_x(double*c, int x, int y, int z, InputPara InputP)
{
    int xm, xp;
    xm = (x - 1 >= 0) ? x - 1 : InputP.Nx - 1;
    xp = (x + 1 < InputP.Nx) ? x + 1 : 0;

    int index_xm = xm + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
    int index_xp = xp + y * InputP.Nx + z * InputP.Nx * InputP.Ny;

    double dpx = (c[index_xp] - c[index_xm]) / 2.0 / InputP.dx;

    return dpx;
}

__device__ double Grad_y(double*c, int x, int y, int z, InputPara InputP)
{
    int ym, yp;
    ym = (y - 1 >= 0) ? y - 1 : InputP.Ny - 1;
    yp = (y + 1 < InputP.Ny) ? y + 1 : 0;

    int index_ym = x + ym * InputP.Nx + z * InputP.Nx * InputP.Ny;
    int index_yp = x + yp * InputP.Nx + z * InputP.Nx * InputP.Ny;

    double dpy = (c[index_yp] - c[index_ym]) / 2.0 / InputP.dy;

    return dpy;
}

__device__ double Laplacian(double* c, int x, int y, int z, InputPara InputP)
{
	int xm, xp, ym, yp, zm, zp;

	//Finite Difference
	xm = (x - 1 >= 0) ? x - 1 : InputP.Nx - 1;
	xp = (x + 1 < InputP.Nx) ? x + 1 : 0;
	ym = (y - 1 >= 0) ? y - 1 : InputP.Ny - 1;
	yp = (y + 1 < InputP.Ny) ? y + 1 : 0;
	zm = (z - 1 >= 0) ? z - 1 : InputP.Nz - 1;
	zp = (z + 1 < InputP.Nz) ? z + 1 : 0;

	int index_xyz = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;	// Coalesced memory access
	int index_xmyz = xm + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_xpyz = xp + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_xymz = x + ym * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_xypz = x + yp * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_xyzm = x + y * InputP.Nx + zm * InputP.Nx * InputP.Ny;
	int index_xyzp = x + y * InputP.Nx + zp * InputP.Nx * InputP.Ny;

	int index_xpypz = xp + yp * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_xpynz = xp + ym * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_xnypz = xm + yp * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_xnynz = xm + ym * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_xypzp = x + yp * InputP.Nx + zp * InputP.Nx * InputP.Ny;
	int index_xynzp = x + ym * InputP.Nx + zp * InputP.Nx * InputP.Ny;
	int index_xypzn = x + yp * InputP.Nx + zm * InputP.Nx * InputP.Ny;
	int index_xynzn = x + ym * InputP.Nx + zm * InputP.Nx * InputP.Ny;
	int index_xpyzp = xp + y * InputP.Nx + zp * InputP.Nx * InputP.Ny;
	int index_xnyzp = xm + y * InputP.Nx + zp * InputP.Nx * InputP.Ny;
	int index_xpyzn = xp + y * InputP.Nx + zm * InputP.Nx * InputP.Ny;
	int index_xnyzn = xm + y * InputP.Nx + zm * InputP.Nx * InputP.Ny;

	int index_xpypzp = xp + yp * InputP.Nx + zp * InputP.Nx * InputP.Ny;
	int index_xnypzp = xm + yp * InputP.Nx + zp * InputP.Nx * InputP.Ny;
	int index_xnynzp = xm + ym * InputP.Nx + zp * InputP.Nx * InputP.Ny;
	int index_xpynzp = xp + ym * InputP.Nx + zp * InputP.Nx * InputP.Ny;
	int index_xpypzn = xp + yp * InputP.Nx + zm * InputP.Nx * InputP.Ny;
	int index_xnypzn = xm + yp * InputP.Nx + zm * InputP.Nx * InputP.Ny;
	int index_xnynzn = xm + ym * InputP.Nx + zm * InputP.Nx * InputP.Ny;
	int index_xpynzn = xp + ym * InputP.Nx + zm * InputP.Nx * InputP.Ny;

    // 27-point Laplacian
    double result = 1.0 / (InputP.dx * InputP.dx) * (\
                    - 64.0 / 15.0 * c[index_xyz] \
                    + 7.0 / 15.0 * (c[index_xpyz] + c[index_xmyz] + c[index_xypz] \
								  + c[index_xymz] + c[index_xyzp] + c[index_xyzm] ) \
                    + 0.1 * (c[index_xpypz] + c[index_xpynz] + c[index_xnypz] + c[index_xnynz] \
                            + c[index_xpyzp] + c[index_xypzp] + c[index_xynzp] + c[index_xnyzp] \
                            + c[index_xpyzn] + c[index_xypzn] + c[index_xynzn] + c[index_xnyzn] \
                            ) \
                    + 1.0 / 30.0 * (c[index_xpypzp] + c[index_xnypzp] + c[index_xnynzp] + c[index_xpynzp] \
                            + c[index_xpypzn] + c[index_xnypzn] + c[index_xnynzn] + c[index_xpynzn] \
                            ) \
                    );
    return result;
}

__global__ void getConserved(double* Mass, double* Energy, double* Momx, double* Momy, \
                            double* rho, double* p, double* vx, double* vy, \
                            InputPara InputP)
{   
    int x, y, z, index;
    double gamma = 5.0/3.0;
    double vol = InputP.dx * InputP.dy;

	int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;
    
    for (int k = 0; k < (InputP.Nz - 1) / t_tot_z + 1; k++){
		for (int j = 0; j < (InputP.Ny - 1) / t_tot_y + 1; j++){
			for (int i = 0; i < (InputP.Nx - 1) / t_tot_x + 1; i++){
				z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
				y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
				x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

				if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz)
				{
					index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
                    
                    double rho_value = rho[index];
                    double vx_value = vx[index];
                    double vy_value = vy[index];
                    double p_value = p[index];
    
                    Mass[index] = rho_value * vol;
                    Momx[index] = rho_value * vx_value * vol;
                    Momy[index] = rho_value * vy_value * vol;
                    Energy[index] = (p_value / (gamma - 1.0) + 0.5 * rho_value * \
                                            (vx_value*vx_value + vy_value*vy_value)) * vol;
				}
			}
        }
    }
}

__global__ void getPrimitive(double* rho, double* p, double* vx, double* vy, \
                            double* Mass, double* Energy, double* Momx, double* Momy, \
                            InputPara InputP)
{   
    int x, y, z, index;
    double gamma = 5.0/3.0;
    double vol = InputP.dx * InputP.dy;

	int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;
    
    for (int k = 0; k < (InputP.Nz - 1) / t_tot_z + 1; k++){
		for (int j = 0; j < (InputP.Ny - 1) / t_tot_y + 1; j++){
			for (int i = 0; i < (InputP.Nx - 1) / t_tot_x + 1; i++){
				z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
				y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
				x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

				if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz)
				{
					index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
                    double Mass_value = Mass[index];
                    double Momx_value = Momx[index];
                    double Momy_value = Momy[index];
                    double Energy_value = Energy[index];

                    double rho_value = Mass_value / vol;
                    double vx_value = Momx_value / rho_value / vol;
                    double vy_value = Momy_value / rho_value / vol;
                    double p_value = (Energy_value / vol - 0.5 * rho_value * \
                      (vx_value*vx_value + vy_value*vy_value)) * (gamma - 1.0);

                    rho[index] = rho_value;
                    vx[index] = vx_value;
                    vy[index] = vy_value;
                    p[index] = p_value;
                }
            }
        }

    }
}

__global__ void getPrimitive_wakeup(double* rho, double* p, double* vx, double* vy, \
                            double* Mass, double* Energy, double* Momx, double* Momy, \
                            double* metric, int* wake_up, InputPara InputP)
{   
    int x, y, z, index;
    int block_index, block_index_x, block_index_y, block_index_z;
    __shared__ int wake_up_s;

    double gamma = 5.0/3.0;
    double vol = InputP.dx * InputP.dy;

	int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;
    
    for (int k = 0; k < (InputP.Nz - 1) / t_tot_z + 1; k++){
		for (int j = 0; j < (InputP.Ny - 1) / t_tot_y + 1; j++){
			for (int i = 0; i < (InputP.Nx - 1) / t_tot_x + 1; i++){
                
                // Read from the first thread in each block
                if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
					block_index_x = blockIdx.x + i * gridDim.x;
					block_index_y = blockIdx.y + j * gridDim.y;
					block_index_z = blockIdx.z + k * gridDim.z;
					if (block_index_x < InputP.blockNum_x && block_index_y < InputP.blockNum_y \
                        && block_index_z < InputP.blockNum_z){
						block_index = block_index_x + block_index_y * InputP.blockNum_x \
                                                    + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
						wake_up_s = wake_up[block_index];
					}
				}

                // Synchronize threads in threadblock before using the shared memory
				__syncthreads();

                // Execute the following only if the block is waken up
                if (wake_up_s){
				    z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
				    y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
				    x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

				    if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz){
					    index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
                        double Mass_value = Mass[index];
                        double Momx_value = Momx[index];
                        double Momy_value = Momy[index];
                        double Energy_value = Energy[index];

                        double rho_value = Mass_value / vol;
                        double vx_value = Momx_value / rho_value / vol;
                        double vy_value = Momy_value / rho_value / vol;
                        double p_value = (Energy_value / vol - 0.5 * rho_value * \
                                        (vx_value*vx_value + vy_value*vy_value)) * (gamma - 1.0);

                        rho[index] = rho_value;
                        vx[index] = vx_value;
                        vy[index] = vy_value;
                        p[index] = p_value;
                    }
                }
            }
        }

    }
}

__global__ void extrap_in_time(double* rho, double* p, double* vx, double* vy, \
                            double* rho_dx, double* rho_dy, double* p_dx, double* p_dy, \
                            double* vx_dx, double* vx_dy, double* vy_dx, double* vy_dy, \
                            double* rho_prime, double* p_prime, double* vx_prime, double* vy_prime, \
                            InputPara InputP)
{
    int x, y, z, index;
    double gamma = 5.0/3.0;

    int t_tot_x = blockDim.x * gridDim.x;
    int t_tot_y = blockDim.y * gridDim.y;
    int t_tot_z = blockDim.z * gridDim.z;
    
    for (int k = 0; k < (InputP.Nz - 1) / t_tot_z + 1; k++){
        for (int j = 0; j < (InputP.Ny - 1) / t_tot_y + 1; j++){
            for (int i = 0; i < (InputP.Nx - 1) / t_tot_x + 1; i++){
                z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
                y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
                x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

                if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz)
                {
                    index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
                    
                    double rho_value = rho[index];
                    double vx_value = vx[index];
                    double vy_value = vy[index];
                    double p_value = p[index];
        
                    // calculate gradients -- periodic boundary
                    rho_dx[index] = Grad_x(rho, x, y, z, InputP);
                    rho_dy[index] = Grad_y(rho, x, y, z, InputP);
                    p_dx[index] = Grad_x(p, x, y, z, InputP);
                    p_dy[index] = Grad_y(p, x, y, z, InputP);
                    vx_dx[index] = Grad_x(vx, x, y, z, InputP);
                    vx_dy[index] = Grad_y(vx, x, y, z, InputP);
                    vy_dx[index] = Grad_x(vy, x, y, z, InputP);
                    vy_dy[index] = Grad_y(vy, x, y, z, InputP);
                    
                    // extrapolate half-step in time
                    rho_prime[index] = rho_value - 0.5*InputP.dt * ( vx_value * rho_dx[index] + rho_value * vx_dx[index] \
                                        + vy_value * rho_dy[index] + rho_value * vy_dy[index]);
                    vx_prime[index]  = vx_value  - 0.5*InputP.dt * ( vx_value * vx_dx[index] + vy_value * vx_dy[index] \
                                        + (1.0/rho_value) * p_dx[index] );
                    vy_prime[index]  = vy_value  - 0.5*InputP.dt * ( vx_value * vy_dx[index] + vy_value * vy_dy[index] \
                                        + (1.0/rho_value) * p_dy[index] );
                    p_prime[index]   = p_value   - 0.5*InputP.dt * ( gamma * p_value * (vx_dx[index] + vy_dy[index]) \
                                        + vx_value * p_dx[index] + vy_value * p_dy[index] );
                }
            }
        }
    }
}

__global__ void extrap_in_time_wakeup(double* rho, double* p, double* vx, double* vy, \
                            double* rho_dx, double* rho_dy, double* p_dx, double* p_dy, \
                            double* vx_dx, double* vx_dy, double* vy_dx, double* vy_dy, \
                            double* rho_prime, double* p_prime, double* vx_prime, double* vy_prime, \
                            double* metric1, double* metric2, double* metric3, double* metric4, \
                            int* wake_up, InputPara InputP)
{
    int x, y, z, index;
    double gamma = 5.0/3.0;

    int block_index, block_index_x, block_index_y, block_index_z;
    __shared__ int wake_up_s;

    int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;
    
    for (int k = 0; k < (InputP.Nz - 1) / t_tot_z + 1; k++){
		for (int j = 0; j < (InputP.Ny - 1) / t_tot_y + 1; j++){
			for (int i = 0; i < (InputP.Nx - 1) / t_tot_x + 1; i++){
                
                // Read from the first thread in each block
                if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
					block_index_x = blockIdx.x + i * gridDim.x;
					block_index_y = blockIdx.y + j * gridDim.y;
					block_index_z = blockIdx.z + k * gridDim.z;
					if (block_index_x < InputP.blockNum_x && block_index_y < InputP.blockNum_y \
                        && block_index_z < InputP.blockNum_z){
						block_index = block_index_x + block_index_y * InputP.blockNum_x \
                                                    + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
						wake_up_s = wake_up[block_index];
					}
				}

                // Synchronize threads in threadblock before using the shared memory
				__syncthreads();

                // Execute the following only if the block is waken up
                if (wake_up_s){
				    z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
				    y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
				    x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

                    if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz){
                        index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
                    
                        double rho_value = rho[index];
                        double vx_value = vx[index];
                        double vy_value = vy[index];
                        double p_value = p[index];
        
                        // calculate gradients -- periodic boundary
                        rho_dx[index] = Grad_x(rho, x, y, z, InputP);
                        rho_dy[index] = Grad_y(rho, x, y, z, InputP);
                        p_dx[index] = Grad_x(p, x, y, z, InputP);
                        p_dy[index] = Grad_y(p, x, y, z, InputP);
                        vx_dx[index] = Grad_x(vx, x, y, z, InputP);
                        vx_dy[index] = Grad_y(vx, x, y, z, InputP);
                        vy_dx[index] = Grad_x(vy, x, y, z, InputP);
                        vy_dy[index] = Grad_y(vy, x, y, z, InputP);
                        
                        // extrapolate half-step in time
                        rho_prime[index] = rho_value - 0.5*InputP.dt \
                                                    * ( vx_value * rho_dx[index] + rho_value * vx_dx[index] \
                                                      + vy_value * rho_dy[index] + rho_value * vy_dy[index]);
                        vx_prime[index]  = vx_value  - 0.5*InputP.dt \
                                                    * ( vx_value * vx_dx[index] + vy_value * vx_dy[index] \
                                                    + (1.0/rho_value) * p_dx[index] );
                        vy_prime[index]  = vy_value  - 0.5*InputP.dt \
                                                    * ( vx_value * vy_dx[index] + vy_value * vy_dy[index] \
                                                    + (1.0/rho_value) * p_dy[index] );
                        p_prime[index]   = p_value   - 0.5*InputP.dt \
                                                    * ( gamma * p_value * (vx_dx[index] + vy_dy[index]) \
                                                    + vx_value * p_dx[index] + vy_value * p_dy[index] );

                        // set a metric for wake-up mode
                        metric1[index] = fabs(rho_dx[index]) + fabs(rho_dy[index]) \
                                        + fabs(p_dx[index]) + fabs(p_dy[index]) \
                                        + fabs(vx_dx[index]) + fabs(vx_dy[index]) \
                                        + fabs(vy_dx[index]) + fabs(vy_dy[index]);
                        //metric1[index] = fabs(rho_dx[index]) + fabs(rho_dy[index]);
                        //printf("fabs(rho) = %f\n", fabs(rho_dx[index])+fabs(rho_dy[index]));
                        //printf("fabs(p) = %f\n", fabs(p_dx[index])+fabs(p_dy[index]));
                        //printf("fabs(vx) = %f\n", fabs(vx_dx[index])+fabs(vx_dy[index]));
                        //printf("fabs(vy) = %f\n", fabs(vy_dx[index])+fabs(vy_dy[index]));
                    }
                }
            }
        }
    }
}

__device__ void extrapolateInSpaceToFace(double face[4], double* f, double* f_dx, double* f_dy,
                                        int x, int y, int z, InputPara InputP)
{   
    // index for periodic boundary
	//int xm = (x - 1 >= 0) ? x - 1 : InputP.Nx - 1;
	int xp = (x + 1 < InputP.Nx) ? x + 1 : 0;
	//int ym = (y - 1 >= 0) ? y - 1 : InputP.Ny - 1;
	int yp = (y + 1 < InputP.Ny) ? y + 1 : 0;
	//int zm = (z - 1 >= 0) ? z - 1 : InputP.Nz - 1;
	//int zp = (z + 1 < InputP.Nz) ? z + 1 : 0;
    
    // Coalesced memory access
    // xm and ym are not used
	int index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;	
	//int index_xm = xm + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_xp = xp + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
	//int index_ym = x + ym * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_yp = x + yp * InputP.Nx + z * InputP.Nx * InputP.Ny;
	//int index_zm = x + y * InputP.Nx + zm * InputP.Nx * InputP.Ny;
	//int index_zp = x + y * InputP.Nx + zp * InputP.Nx * InputP.Ny;

    //  +--3--+
    //  |     |
    //  0     1
    //  |     |
    //  +--2--+
    face[0] = f[index_xp] - f_dx[index_xp] * InputP.dx / 2.0;
    face[1] = f[index] + f_dx[index] * InputP.dx / 2.0;
    face[2] = f[index_yp] - f_dy[index_yp] * InputP.dy / 2.0;
    face[3] = f[index] + f_dy[index] * InputP.dy / 2.0;
}

__device__ void getFlux(double* flux_Mass, double* flux_Energy, double* flux_Momx, double* flux_Momy, \
                        double rho_L, double rho_R, double p_L, double p_R, \
                        double vx_L, double vx_R, double vy_L, double vy_R, \
                        int x, int y, int z, InputPara InputP)
{
    int index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
    double gamma = 5.0/3.0;

    // left and right energies
	double en_L = p_L/(gamma-1.0)+0.5*rho_L * (vx_L*vx_L+vy_L*vy_L);
	double en_R = p_R/(gamma-1.0)+0.5*rho_R * (vx_R*vx_R+vy_R*vy_R);

	// compute star (averaged) states
	double rho_star  = 0.5*(rho_L + rho_R);
	double momx_star = 0.5*(rho_L * vx_L + rho_R * vx_R);
	double momy_star = 0.5*(rho_L * vy_L + rho_R * vy_R);
	double en_star   = 0.5*(en_L + en_R);
	double p_star = (gamma-1.0)*(en_star-0.5*(momx_star*momx_star+momy_star*momy_star)/rho_star);

	// compute fluxes (local Lax-Friedrichs/Rusanov)
	double flux_Mass_value   = momx_star;
	double flux_Momx_value   = momx_star*momx_star/rho_star + p_star;
	double flux_Momy_value   = momx_star*momy_star/rho_star;
	double flux_Energy_value = (en_star+p_star) * momx_star/rho_star;

	// find wavespeeds
	double C_L = sqrt(gamma*p_L/rho_L) + fabs(vx_L);
	double C_R = sqrt(gamma*p_R/rho_R) + fabs(vx_R);
    double C = (C_L >= C_R) ? C_L : C_R; // use the maximum wavespeed

	// add stabilizing diffusive term
	flux_Mass[index] = flux_Mass_value - C * 0.5 * (rho_L - rho_R);
	flux_Momx[index] = flux_Momx_value - C * 0.5 * (rho_L * vx_L - rho_R * vx_R);
	flux_Momy[index] = flux_Momy_value - C * 0.5 * (rho_L * vy_L - rho_R * vy_R);
	flux_Energy[index] = flux_Energy_value - C * 0.5 * (en_L - en_R);
}

__global__ void calc_flux(double* rho, double* p, double* vx, double* vy, \
                        double* rho_dx, double* rho_dy, double* p_dx, double* p_dy, \
                        double* vx_dx, double* vx_dy, double* vy_dx, double* vy_dy, \
                        double* rho_prime, double* p_prime, double* vx_prime, double* vy_prime, \
                        double* flux_Mass_x, double* flux_Energy_x, double* flux_Momx_x, double* flux_Momy_x, \
                        double* flux_Mass_y, double* flux_Energy_y, double* flux_Momx_y, double* flux_Momy_y, \
                        InputPara InputP)
{
    int x, y, z;

    int t_tot_x = blockDim.x * gridDim.x;
    int t_tot_y = blockDim.y * gridDim.y;
    int t_tot_z = blockDim.z * gridDim.z;
    
    for (int k = 0; k < (InputP.Nz - 1) / t_tot_z + 1; k++){
        for (int j = 0; j < (InputP.Ny - 1) / t_tot_y + 1; j++){
            for (int i = 0; i < (InputP.Nx - 1) / t_tot_x + 1; i++){
                z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
                y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
                x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

                if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz)
                {                    
                    // extrapolate in space to face centers
                    // 0: left, 1: right, 2: down, 3: up
                    double rho_face[4];
                    double vx_face[4];
                    double vy_face[4];
                    double p_face[4];

                    extrapolateInSpaceToFace(rho_face, rho_prime, rho_dx, rho_dy, x, y, z, InputP);
                    extrapolateInSpaceToFace(vx_face, vx_prime, vx_dx, vx_dy, x, y, z, InputP);
		            extrapolateInSpaceToFace(vy_face, vy_prime, vy_dx, vy_dy, x, y, z, InputP);
                    extrapolateInSpaceToFace(p_face, p_prime, p_dx, p_dy, x, y, z, InputP);

                    // compute fluxes (local Lax-Friedrichs/Rusanov)
                    getFlux(flux_Mass_x, flux_Energy_x, flux_Momx_x, flux_Momy_x, \
                            rho_face[0], rho_face[1], p_face[0], p_face[1], \
                            vx_face[0], vx_face[1], vy_face[0], vy_face[1], x, y, z, InputP);
                    getFlux(flux_Mass_y, flux_Energy_y, flux_Momy_y, flux_Momx_y, \
                            rho_face[2], rho_face[3], p_face[2], p_face[3], \
                            vy_face[2], vy_face[3], vx_face[2], vx_face[3], x, y, z, InputP);
                }
            }
        }
    }
}

__global__ void calc_flux_wakeup(double* rho, double* p, double* vx, double* vy, \
                        double* rho_dx, double* rho_dy, double* p_dx, double* p_dy, \
                        double* vx_dx, double* vx_dy, double* vy_dx, double* vy_dy, \
                        double* rho_prime, double* p_prime, double* vx_prime, double* vy_prime, \
                        double* flux_Mass_x, double* flux_Energy_x, double* flux_Momx_x, double* flux_Momy_x, \
                        double* flux_Mass_y, double* flux_Energy_y, double* flux_Momx_y, double* flux_Momy_y, \
                        double* metric1, double* metric2, double* metric3, double* metric4, \
                        int* wake_up, InputPara InputP)
{
    int x, y, z;

    int block_index, block_index_x, block_index_y, block_index_z;
    __shared__ int wake_up_s;

    int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;
    
    for (int k = 0; k < (InputP.Nz - 1) / t_tot_z + 1; k++){
		for (int j = 0; j < (InputP.Ny - 1) / t_tot_y + 1; j++){
			for (int i = 0; i < (InputP.Nx - 1) / t_tot_x + 1; i++){
                
                // Read from the first thread in each block
                if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
					block_index_x = blockIdx.x + i * gridDim.x;
					block_index_y = blockIdx.y + j * gridDim.y;
					block_index_z = blockIdx.z + k * gridDim.z;
					if (block_index_x < InputP.blockNum_x && block_index_y < InputP.blockNum_y \
                        && block_index_z < InputP.blockNum_z){
						block_index = block_index_x + block_index_y * InputP.blockNum_x \
                                                    + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
						wake_up_s = wake_up[block_index];
					}
				}

                // Synchronize threads in threadblock before using the shared memory
				__syncthreads();

                // Execute the following only if the block is waken up
                if (wake_up_s){
				    z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
				    y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
				    x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

                    if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz){
                        // extrapolate in space to face centers
                        // 0: left, 1: right, 2: down, 3: up
                        double rho_face[4];
                        double vx_face[4];
                        double vy_face[4];
                        double p_face[4];

                        extrapolateInSpaceToFace(rho_face, rho_prime, rho_dx, rho_dy, x, y, z, InputP);
                        extrapolateInSpaceToFace(vx_face, vx_prime, vx_dx, vx_dy, x, y, z, InputP);
                        extrapolateInSpaceToFace(vy_face, vy_prime, vy_dx, vy_dy, x, y, z, InputP);
                        extrapolateInSpaceToFace(p_face, p_prime, p_dx, p_dy, x, y, z, InputP);

                        // compute fluxes (local Lax-Friedrichs/Rusanov)
                        getFlux(flux_Mass_x, flux_Energy_x, flux_Momx_x, flux_Momy_x, \
                                rho_face[0], rho_face[1], p_face[0], p_face[1], \
                                vx_face[0], vx_face[1], vy_face[0], vy_face[1], x, y, z, InputP);
                        getFlux(flux_Mass_y, flux_Energy_y, flux_Momy_y, flux_Momx_y, \
                                rho_face[2], rho_face[3], p_face[2], p_face[3], \
                                vy_face[2], vy_face[3], vx_face[2], vx_face[3], x, y, z, InputP);
                    }
                }
            }
        }
    }
}

__device__ void applyFluxes(double* f, double* f_change, double* flux_x, double* flux_y, \
                                int x, int y, int z, InputPara InputP)
{
    // index for periodic boundary
	int xm = (x - 1 >= 0) ? x - 1 : InputP.Nx - 1;
	//int xp = (x + 1 < InputP.Nx) ? x + 1 : 0;
	int ym = (y - 1 >= 0) ? y - 1 : InputP.Ny - 1;
	//int yp = (y + 1 < InputP.Ny) ? y + 1 : 0;
	//int zm = (z - 1 >= 0) ? z - 1 : InputP.Nz - 1;
	//int zp = (z + 1 < InputP.Nz) ? z + 1 : 0;
    
    // Coalesced memory access
    // xp and yp are not used
	int index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;	
	int index_xm = xm + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
	//int index_xp = xp + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_ym = x + ym * InputP.Nx + z * InputP.Nx * InputP.Ny;
	//int index_yp = x + yp * InputP.Nx + z * InputP.Nx * InputP.Ny;
	//int index_zm = x + y * InputP.Nx + zm * InputP.Nx * InputP.Ny;
	//int index_zp = x + y * InputP.Nx + zp * InputP.Nx * InputP.Ny;

	double f_change_temp = InputP.dt * InputP.dx * (-1.0*flux_x[index] + flux_x[index_xm] \
                                        -1.0*flux_y[index] + flux_y[index_ym]);
        f_change[index] = f_change_temp;
    	f[index] += f_change_temp;
}

__global__ void apply_flux(double* Mass, double* Energy, double* Momx, double* Momy, double* change_Mass, double* change_Energy, double* change_Momx, double* change_Momy, \
                        double* flux_Mass_x, double* flux_Energy_x, double* flux_Momx_x, double* flux_Momy_x, \
                        double* flux_Mass_y, double* flux_Energy_y, double* flux_Momx_y, double* flux_Momy_y, \
                        InputPara InputP)
{
    int x, y, z;

    int t_tot_x = blockDim.x * gridDim.x;
    int t_tot_y = blockDim.y * gridDim.y;
    int t_tot_z = blockDim.z * gridDim.z;
    
    for (int k = 0; k < (InputP.Nz - 1) / t_tot_z + 1; k++){
        for (int j = 0; j < (InputP.Ny - 1) / t_tot_y + 1; j++){
            for (int i = 0; i < (InputP.Nx - 1) / t_tot_x + 1; i++){
                z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
                y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
                x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

                if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz){     
                    applyFluxes(Mass, change_Mass, flux_Mass_x, flux_Mass_y, x, y, z, InputP);
                    applyFluxes(Momx, change_Momx, flux_Momx_x, flux_Momx_y, x, y, z, InputP);
                    applyFluxes(Momy, change_Momy, flux_Momy_x, flux_Momy_y, x, y, z, InputP);
                    applyFluxes(Energy, change_Energy, flux_Energy_x, flux_Energy_y, x, y, z, InputP);
                }
            }
        }
    }
}

__global__ void apply_flux_wakeup(double* Mass, double* Energy, double* Momx, double* Momy, double* change_Mass, double* change_Energy, double* change_Momx, double* change_Momy, \
                        double* flux_Mass_x, double* flux_Energy_x, double* flux_Momx_x, double* flux_Momy_x, \
                        double* flux_Mass_y, double* flux_Energy_y, double* flux_Momx_y, double* flux_Momy_y, \
                        double* metric1, double* metric2, double* metric3, double* metric4, \
                        int* wake_up, InputPara InputP)
{
    int x, y, z;

        int block_index, block_index_x, block_index_y, block_index_z;
    __shared__ int wake_up_s;

    int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;
    
    for (int k = 0; k < (InputP.Nz - 1) / t_tot_z + 1; k++){
		for (int j = 0; j < (InputP.Ny - 1) / t_tot_y + 1; j++){
			for (int i = 0; i < (InputP.Nx - 1) / t_tot_x + 1; i++){
                
                // Read from the first thread in each block
                if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
					block_index_x = blockIdx.x + i * gridDim.x;
					block_index_y = blockIdx.y + j * gridDim.y;
					block_index_z = blockIdx.z + k * gridDim.z;
					if (block_index_x < InputP.blockNum_x && block_index_y < InputP.blockNum_y \
                        && block_index_z < InputP.blockNum_z){
						block_index = block_index_x + block_index_y * InputP.blockNum_x \
                                                    + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
						wake_up_s = wake_up[block_index];
					}
				}

                // Synchronize threads in threadblock before using the shared memory
				__syncthreads();

                // Execute the following only if the block is waken up
                if (wake_up_s){
				    z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
				    y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
				    x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

                    if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz){
                        applyFluxes(Mass, change_Mass, flux_Mass_x, flux_Mass_y, x, y, z, InputP);
                    	applyFluxes(Momx, change_Momx, flux_Momx_x, flux_Momx_y, x, y, z, InputP);
                    	applyFluxes(Momy, change_Momy, flux_Momy_x, flux_Momy_y, x, y, z, InputP);
                    	applyFluxes(Energy, change_Energy, flux_Energy_x, flux_Energy_y, x, y, z, InputP);
                    }
                }
                else
                {
                	z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
			y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
			x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;
			if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz)
			{
				int index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;	// Coalsced memory access
				change_Mass[index] = 0;
				change_Energy[index] = 0;
				change_Momx[index] = 0;
				change_Momy[index] = 0;
			}
                }
            }
        }
    }
}

//Wakeup blocks for next step
__global__ void Wakeup_Next(double* c, double* c_new, \
                            double* metric1, double* metric2, double* metric3, double* metric4, \
                            int* wake_1, int* wake_1_next, InputPara InputP, int timestep)
{
	int x, y, z, index;
	int block_index, block_index_x, block_index_y, block_index_z;
	__shared__ int wake_1_s, wake_1_s_next;

	int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;

	for (int k = 0; k < (InputP.Nz - 1) / t_tot_z + 1; k++)
		for (int j = 0; j < (InputP.Ny - 1) / t_tot_y + 1; j++)
			for (int i = 0; i < (InputP.Nx - 1) / t_tot_x + 1; i++)
			{
				z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
				y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
				x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

				if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz)
				{
					index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;	// Coalsced memory access
					if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)	// First thread in each block
					{
						block_index_x = blockIdx.x + i * gridDim.x;
						block_index_y = blockIdx.y + j * gridDim.y;
						block_index_z = blockIdx.z + k * gridDim.z;
						if (block_index_x < InputP.blockNum_x && block_index_y < InputP.blockNum_y \
                            && block_index_z < InputP.blockNum_z)
						{
							block_index = block_index_x + block_index_y * InputP.blockNum_x + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
							wake_1_s = wake_1[block_index];
							wake_1[block_index] = 0;
							wake_1_s_next = 0;
						}
					}
					// Synchronize threads in threadblock before using the shared memory
					__syncthreads();

					if (wake_1_s)		// Check if this block is awake
					{	
						//Check if this block awake in next step
						if ((metric1[index] > InputP.Metric_eps)){ // additional criterion: || (timestep < 10000)
							wake_1_s_next = 1;
							//std::printf("Metric = %f\n", metric[index]);
						}
						// Synchronize threads in threadblock before using the shared memory
						__syncthreads();
						if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)	// First thread in each block
						{
							if (block_index_x < InputP.blockNum_x && block_index_y < InputP.blockNum_y && block_index_z < InputP.blockNum_z)
							{
								wake_1_next[block_index] = wake_1_s_next;
							}
						}
					}
				}
			}
}

//Wakeup neighbor blocks
__global__ void Wakeup_Neighbor(int* wake_1, int* wake_1_next, InputPara InputP)
{
	int block_index, block_index_x, block_index_y, block_index_z;
	int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;
	for (int k = 0; k < (InputP.blockNum_z - 1) / t_tot_z + 1; k++)
		for (int j = 0; j < (InputP.blockNum_y - 1) / t_tot_y + 1; j++)
			for (int i = 0; i < (InputP.blockNum_x - 1) / t_tot_x + 1; i++)
			{
				block_index_z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
				block_index_y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
				block_index_x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;
				
				if(block_index_x < InputP.blockNum_x && block_index_y < InputP.blockNum_y && block_index_z < InputP.blockNum_z)
				{
					block_index = block_index_x + block_index_y * InputP.blockNum_x + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
					if (wake_1_next[block_index])
					{
						int block_xm = (block_index_x - 1 >= 0) ? block_index_x - 1 : InputP.blockNum_x - 1;
						int block_ym = (block_index_y - 1 >= 0) ? block_index_y - 1 : InputP.blockNum_y - 1;
						int block_zm = (block_index_z - 1 >= 0) ? block_index_z - 1 : InputP.blockNum_z - 1;
						int block_xp = (block_index_x + 1 < InputP.blockNum_x) ? block_index_x + 1 : 0;
						int block_yp = (block_index_y + 1 < InputP.blockNum_y) ? block_index_y + 1 : 0;
						int block_zp = (block_index_z + 1 < InputP.blockNum_z) ? block_index_z + 1 : 0;

						int block_index_xm = block_xm + block_index_y * InputP.blockNum_x + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
						int block_index_xp = block_xp + block_index_y * InputP.blockNum_x + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
						int block_index_ym = block_index_x + block_ym * InputP.blockNum_x + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
						int block_index_yp = block_index_x + block_yp * InputP.blockNum_x + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
						int block_index_zm = block_index_x + block_index_y * InputP.blockNum_x + block_zm * InputP.blockNum_x * InputP.blockNum_y;
						int block_index_zp = block_index_x + block_index_y * InputP.blockNum_x + block_zp * InputP.blockNum_x * InputP.blockNum_y;

                        int block_index_xm_ym = block_xm + block_ym * InputP.blockNum_x + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
                        int block_index_xm_yp = block_xm + block_yp * InputP.blockNum_x + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
                        int block_index_xp_ym = block_xp + block_ym * InputP.blockNum_x + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
                        int block_index_xp_yp = block_xp + block_yp * InputP.blockNum_x + block_index_z * InputP.blockNum_x * InputP.blockNum_y;

						if (wake_1_next[block_index])
						{
							wake_1[block_index] = 1;
							wake_1[block_index_xm] = 1;
							wake_1[block_index_xp] = 1;
							wake_1[block_index_ym] = 1;
							wake_1[block_index_yp] = 1;
							wake_1[block_index_zm] = 1;
							wake_1[block_index_zp] = 1;
                            wake_1[block_index_xm_ym] = 1;
                            wake_1[block_index_xm_yp] = 1;
                            wake_1[block_index_xp_ym] = 1;
                            wake_1[block_index_xp_yp] = 1;
						}
					}
					
				}
			}		
}

//Initialize system
__global__ void Initialize(double* rho, double* p, double* vx, double* vy, \
                            int* wake_1, int* wake_1_next, InputPara InputP)
{
	int x, y, z, index;
	int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;

	for (int k = 0; k < (InputP.Nz - 1) / t_tot_z + 1; k++){
		for (int j = 0; j < (InputP.Ny - 1) / t_tot_y + 1; j++){
			for (int i = 0; i < (InputP.Nx - 1) / t_tot_x + 1; i++){
				if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)	// First thread in each block
				{
					int block_index_x = blockIdx.x + i * gridDim.x;
					int block_index_y = blockIdx.y + j * gridDim.y;
					int block_index_z = blockIdx.z + k * gridDim.z;
					if (block_index_x < InputP.blockNum_x && block_index_y < InputP.blockNum_y && block_index_z < InputP.blockNum_z)
					{
						int block_index = block_index_x + block_index_y * InputP.blockNum_x \
										 + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
						wake_1[block_index] = 1;
						wake_1_next[block_index] = 0;
					}
				}
				z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
				y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
				x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

				if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz)
				{
					index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;	// Coalesced memory access

					// constant pressure everywhere
                    p[index] = 2.5;
                    
                    /*
                    // Shape density interface
                    // differnt density and opposite velocity in x direction
                    if ((y>=InputP.Ny/4) && (y<=InputP.Ny*3/4)){
                        rho[index] = 2.0;
                        vx[index] = 0.5;
                    } else {
                        rho[index] = 1.0;
                        vx[index] = -0.5;
                    }

                    // perturbation of velocity in y direction at the interfaces
                    double w0 = 0.1;
                    double sigma = 0.05*M_SQRT1_2;
                    int freq = 4;
                    vy[index] = w0*sin(freq*M_PI*x/InputP.Nx) \
                                * ( exp(-(1.0*y/InputP.Ny-0.25)*(1.0*y/InputP.Ny-0.25)/(2*sigma*sigma)) \
                                    + exp(-(1.0*y/InputP.Ny-0.75)*(1.0*y/InputP.Ny-0.75)/(2*sigma*sigma)) );
                    */

                    // Smooth density interface
                    // Reference Paper: A well-posed Kelvin-Helmholtz instability test and comparison
                    double rho1 = 1.0;
                    double rho2 = 2.0;
                    double rho_m = 0.5*(rho1-rho2);
                    double vx1 = 0.5;
                    double vx2 = -0.5;
                    double vx_m = 0.5*(vx1-vx2);

                    if (y >= 0 && y < InputP.Ny/4){
                        rho[index] = rho1 - rho_m*exp((1.0*y/InputP.Ny - 0.25)/0.025);
                        vx[index] = vx1 - vx_m*exp((1.0*y/InputP.Ny - 0.25)/0.025);
                    } else if (y >= InputP.Ny/4 && y < InputP.Ny/2){
                        rho[index] = rho2 + rho_m*exp((-1.0*y/InputP.Ny + 0.25)/0.025);
                        vx[index] = vx2 + vx_m*exp((-1.0*y/InputP.Ny + 0.25)/0.025);
                    } else if (y >= InputP.Ny/2 && y < 3*InputP.Ny/4){
                        rho[index] = rho2 + rho_m*exp((1.0*y/InputP.Ny - 0.75)/0.025);
                        vx[index] = vx2 + vx_m*exp((1.0*y/InputP.Ny - 0.75)/0.025);
                    } else {
                        rho[index] = rho1 - rho_m*exp((-1.0*y/InputP.Ny + 0.75)/0.025);
                        vx[index] = vx1 - vx_m*exp((-1.0*y/InputP.Ny + 0.75)/0.025);
                    }

                    vy[index] = 0.01*sin(4.0*M_PI*x/InputP.Nx);
				}
			}
        }
    }
}

void write_output_vtk(double* rho, double* p, double* vx, double*vy, InputPara InputP, int t)
{
    int index;

    std::string name = "output_" + std::to_string(InputP.mode)+ "_" + std::to_string(t) + ".vtk";
    std::ofstream ofile(name);

    // vtk preamble
    ofile << "# vtk DataFile Version 2.0" << std::endl;
    ofile << "OUTPUT by Roy Zhang\n";
    ofile << "ASCII" << std::endl;

    // write grid
    ofile << "DATASET RECTILINEAR_GRID" << std::endl;
    ofile << "DIMENSIONS " << InputP.Nx << " " << InputP.Ny << " " << InputP.Nz << std::endl;
    ofile << "X_COORDINATES " << InputP.Nx << " int" << std::endl;
    for (int i = 0; i < InputP.Nx; i++)
        ofile << i << "\t";
    ofile << std::endl;
    ofile << "Y_COORDINATES " << InputP.Ny << " int" << std::endl;
    for (int i = 0; i < InputP.Ny; i++)
        ofile << i << "\t";
    ofile << std::endl;
    ofile << "Z_COORDINATES " << InputP.Nz << " int" << std::endl;
    for (int i = 0; i < InputP.Nz; i++)
        ofile << i << "\t";
    ofile << std::endl;

    // point data
    ofile << "POINT_DATA " << InputP.Nx * InputP.Ny * InputP.Nz << std::endl;

    // write rho
    ofile << "SCALARS rho double" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int k = 0; k < InputP.Nz; k++){
        for (int j = 0; j < InputP.Ny; j++){
            for (int i = 0; i < InputP.Nx; i++){
                index = i + j * InputP.Nx + k * InputP.Nx * InputP.Ny;	// Coalesced memory access
                ofile << rho[index] << " ";
            }
        }
    }

    // write p
    ofile << std::endl;
    ofile << "SCALARS p double" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int k = 0; k < InputP.Nz; k++){
        for (int j = 0; j < InputP.Ny; j++){
            for (int i = 0; i < InputP.Nx; i++){
                index = i + j * InputP.Nx + k * InputP.Nx * InputP.Ny;	// Coalesced memory access
                ofile << p[index] << " ";
            }
        }
    }

    // Vector field point data
    // write vec
    ofile << std::endl;
    ofile << "VECTORS vec double" << std::endl;
    for (int k = 0; k < InputP.Nz; k++){
        for (int j = 0; j < InputP.Ny; j++){
            for (int i = 0; i < InputP.Nx; i++){
                index = i + j * InputP.Nx + k * InputP.Nx * InputP.Ny;	// Coalesced memory access
                ofile << vx[index] << " " << vy[index] << " 0.0 " << std::endl;
            }
        }
    }

    ofile.close();
}

//Output wakeup parameter in .vtk form, as points rather than blocks
void Output_wakeup(std::string prefix, int* data, InputPara InputP, int time)
{
	//std::cout << "--------------Writting Output for Wakeup Parameter Now--------------" << std::endl;
	std::string OutputFileName;
	OutputFileName = prefix + "_" + std::to_string(time) + ".vtk";

	std::ofstream outf;
	outf.open(OutputFileName);
	if (!outf.is_open())
	{
		std::cout << "!!!!!Can not open" << OutputFileName << "!! Exit!!!!!!!!" << std::endl;
		exit(1);
	}

	/* Writting output */
	//std::cout << "Writting output into " << OutputFileName << std::endl;

	outf << "# vtk DataFile Version 2.0" << std::endl;
	outf << "wakeup" << std::endl << "ASCII" << std::endl << "DATASET STRUCTURED_POINTS" << std::endl;
	outf << "DIMENSIONS " << InputP.Nx << " " << InputP.Ny << " " << InputP.Nz << std::endl;
	outf << "ASPECT_RATIO 1 1 1" << std::endl << "ORIGIN 0 0 0" << std::endl << "POINT_DATA " << InputP.Nx * InputP.Ny * InputP.Nz << std::endl;
	outf << "SCALARS wakeup int" << std::endl << "LOOKUP_TABLE default" << std::endl;
	for (int z = 0; z < InputP.Nz; z++)
		for (int y = 0; y < InputP.Ny; y++)
			for (int x = 0; x < InputP.Nx; x++)
			{
				int block_x = x / dimBlock3D.x;
				int block_y = y / dimBlock3D.y;
				int block_z = z / dimBlock3D.z;
				int index_block = block_x + block_y * InputP.blockNum_x + block_z * InputP.blockNum_x * InputP.blockNum_y;
				outf << data[index_block] << " ";
			}
	outf << std::endl;
	outf.close();
	//std::cout << "--------------Output Done--------------" << std::endl;
}

void Calc_RealTime_Values(int* wake_1, InputPara InputP, int time, int RT_freq, time_t sim_time)
{	
	//std::cout << "--------------Writting Output of Wakeup Portion Now--------------" << std::endl;
	std::string OutputFileName;
	OutputFileName = "Wakeup_portion.out";

	std::ofstream outf;
	outf.open(OutputFileName, std::fstream::app);
	if (!outf.is_open())
	{
		std::cout << "!!!!!Can not open" << OutputFileName << "!! Exit!!!!!!!!" << std::endl;
		exit(1);
	}
	//Calculate wakeup portion
	int awake_num_p = 0;
	for (int i = 0; i < InputP.blockNum_x * InputP.blockNum_y * InputP.blockNum_z; i++)
	{
		if (wake_1[i])
			awake_num_p++;
	}
	double awake_portion_p = (double)awake_num_p / (InputP.blockNum_x * InputP.blockNum_y * InputP.blockNum_z);
	std::cout << "Wakeup Portion is " << awake_portion_p << std::endl;

	//Calculate steps per second
	double steps_per_second = RT_freq / ((double)sim_time / CLOCKS_PER_SEC);

	outf << time << " " << awake_portion_p << " " << steps_per_second << std::endl;
	outf.close();
	//std::cout << "--------------Output Done--------------" << std::endl;
}

void Integral(double* Mass, double* Energy, double* Momx, double* Momy, int time, InputPara InputP)
{
    	std::string OutputFileName;
	OutputFileName = "Integral.out";
	std::ofstream outf;
	outf.open(OutputFileName, std::fstream::app);

	double Mass_sum = 0.0;
	double Energy_sum = 0.0;
	double Momx_sum = 0.0;
	double Momy_sum = 0.0;
    	for (int index = 0; index < InputP.Nx*InputP.Ny*InputP.Nz; index++){
		Mass_sum += Mass[index];
		Energy_sum += Energy[index];
		Momx_sum += Momx[index];
		Momy_sum += Momy[index];
	}
	outf << time << " " << Mass_sum << " " << Energy_sum << " " << Momx_sum << " " << Momy_sum << std::endl;
	std::cout << "Integral of Mass = " << Mass_sum << "	Integral of Energy = " << Energy_sum << "	Integral of Momx = " << Momx_sum << "	Integral of Momy = " << Momy_sum << std::endl;

	outf.close();
}

// calculate the amplitude of the y-velocity mode of the instability
void Calc_Vy_mode(double* vy, int time, InputPara InputP){
    std::string OutputFileName;
	OutputFileName = "VY_mode.out";
	std::ofstream outf;
	outf.open(OutputFileName, std::fstream::app);

    // normalized grid points
    double sum_s = 0.0;
    double sum_c = 0.0;
    double sum_d = 0.0;
    for (int z = 0; z < InputP.Nz; z++){
        for (int y = 0; y < InputP.Ny; y++){
            for (int x = 0; x < InputP.Nx; x++){
                int index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
                double x_s = (double)x/InputP.Nx; // scaled x
                double y_s = (double)y/InputP.Ny;
                double z_s = (double)z/InputP.Nz;

                if (y_s < 0.5){
                    sum_s += vy[index] * sin(4.0*M_PI*x_s)*exp(-4.0*M_PI*abs(y_s-0.25));
                    sum_c += vy[index] * cos(4.0*M_PI*x_s)*exp(-4.0*M_PI*abs(y_s-0.25));
                    sum_d += exp(-4.0*M_PI*abs(y_s-0.25));
                } else{
                    sum_s += vy[index] * sin(4.0*M_PI*x_s)*exp(-4.0*M_PI*abs(y_s-0.75));
                    sum_c += vy[index] * cos(4.0*M_PI*x_s)*exp(-4.0*M_PI*abs(y_s-0.75));
                    sum_d += exp(-4.0*M_PI*abs(y_s-0.75));
                }
                
            }
        }
    }
    
    double M = 2.0 * sqrt(sum_s*sum_s/(sum_d*sum_d) + sum_c*sum_c/(sum_d*sum_d));
    outf << time << " " << M << std::endl;
	std::cout << "Growth mode amplitude = " << M << std::endl;
    outf.close();
}

//Force the conservation, apply to both active and deactive blocks as reduced revolution
__global__ void Conservation(double* Mass, double* Energy, double* Momx, double* Momy, double change_Mass_avg, double change_Energy_avg, double change_Momx_avg, double change_Momy_avg, InputPara InputP)
{
	int x, y, z, index;

	int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;
	
	for (int k = 0; k < (InputP.Nz - 1) / t_tot_z + 1; k++)
		for (int j = 0; j < (InputP.Ny - 1) / t_tot_y + 1; j++)
			for (int i = 0; i < (InputP.Nx - 1) / t_tot_x + 1; i++)
			{
				z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
				y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
				x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;
				if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz)
				{
					index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;	// Coalesced memory access
					
					Mass[index] -= change_Mass_avg;
					Energy[index] -= change_Energy_avg;
					Momx[index] -= change_Momx_avg;
					Momy[index] -= change_Momy_avg;
				}
						
			}
}

int main()
{   
    /*-------------------For performance test-------------------*/
	time_t t_start_tot, t_end_tot;
	float kernel_time[10] = { 0 };
	float kernel_time_tot[10] = { 0 };
	cudaEvent_t kernel_start, kernel_end;
	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_end);
	t_start_tot = clock();

    InputPara InputP;
	ReadInput("Input.txt", InputP);

    cudaSetDevice(0);

    Variable Var;
	Allocate(Var, InputP);

    size_t total_size = InputP.Nx * InputP.Ny * InputP.Nz * sizeof(double);
    std::cout << "Total managed memory allocated: " \
             << 32 * total_size/(1024.*1024.*1024.) << " Gb\n" << std::endl;
    
    // Initialization and write file
    Initialize << <dimGrid3D, dimBlock3D >> > (Var.rho, Var.p, Var.vx, Var.vy, \
                                        Var.wake_1, Var.wake_1_next, InputP);
	CheckLastCudaError();
    std::cout << "Initialization finished." << std::endl;
    

    CUDACHECK(cudaDeviceSynchronize()); // this is necessary to ensure the kernel is finished
    write_output_vtk(Var.rho, Var.p, Var.vx, Var.vy, InputP, 0);
    std::cout << "Write initial output file finished." << std::endl;

    // print max value of d_rho using thrust
    //thrust::device_ptr<double> d_ptr = thrust::device_pointer_cast(*d_rho);
    //double max = *(thrust::max_element(d_ptr, d_ptr + system_size));
    //std::cout << max << std::endl;
    
    /*-------------------For performance test-------------------*/
    time_t t_start_loop, t_end_loop, sim_time_start, sim_time_end;
	t_start_loop = clock();
	sim_time_start = clock();

    // calculate conserved variables
    getConserved <<<dimGrid3D, dimBlock3D>>> (Var.Mass, Var.Energy, Var.Momx, Var.Momy, \
                                            Var.rho, Var.p, Var.vx, Var.vy, InputP);
    CheckLastCudaError();

    // test conservation laws
    CUDACHECK(cudaDeviceSynchronize()); // this is necessary
    Integral(Var.Mass, Var.Energy, Var.Momx, Var.Momy, 0, InputP);

    // Main loop
    int t = 1;
    printf("Start main loop...\n\n");
    while (t <= InputP.t_total) {
        //Check the mode of computing
		if (!InputP.mode)
		{   
            // calculate primitive variables
            cudaEventRecord(kernel_start, 0);
            getPrimitive <<<dimGrid3D, dimBlock3D>>> (Var.rho, Var.p, Var.vx, Var.vy, \
                                                Var.Mass, Var.Energy, Var.Momx, Var.Momy, \
                                                InputP);
            CheckLastCudaError();
            cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[0], kernel_start, kernel_end);
            
            // extrapolate primitive variables in time
            cudaEventRecord(kernel_start, 0);
            extrap_in_time <<<dimGrid3D, dimBlock3D>>> (Var.rho, Var.p, Var.vx, Var.vy, \
                                                Var.rho_dx, Var.rho_dy, Var.p_dx, Var.p_dy, \
                                                Var.vx_dx, Var.vx_dy, Var.vy_dx, Var.vy_dy, \
                                                Var.rho_prime, Var.p_prime, Var.vx_prime, Var.vy_prime, \
                                                InputP);
            CheckLastCudaError();
            cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[1], kernel_start, kernel_end);
            
            // calculate fluxes to conserved variables
            cudaEventRecord(kernel_start, 0);
            calc_flux <<<dimGrid3D, dimBlock3D>>> (Var.rho, Var.p, Var.vx, Var.vy, \
                                                Var.rho_dx, Var.rho_dy, Var.p_dx, Var.p_dy, \
                                                Var.vx_dx, Var.vx_dy, Var.vy_dx, Var.vy_dy, \
                                                Var.rho_prime, Var.p_prime, Var.vx_prime, Var.vy_prime, \
                                                Var.flux_Mass_x, Var.flux_Energy_x, Var.flux_Momx_x, Var.flux_Momy_x, \
                                                Var.flux_Mass_y, Var.flux_Energy_y, Var.flux_Momx_y, Var.flux_Momy_y, \
                                                InputP);
            CheckLastCudaError();
            cudaEventRecord(kernel_end, 0);
            cudaEventSynchronize(kernel_end);
            cudaEventElapsedTime(&kernel_time[2], kernel_start, kernel_end);
            
            // apply fluxes to conserved variables
            cudaEventRecord(kernel_start, 0);
            apply_flux <<<dimGrid3D, dimBlock3D>>> (Var.Mass, Var.Energy, Var.Momx, Var.Momy, Var.change_Mass, Var.change_Energy, Var.change_Momx, Var.change_Momy,\
                                                Var.flux_Mass_x, Var.flux_Energy_x, Var.flux_Momx_x, Var.flux_Momy_x, \
                                                Var.flux_Mass_y, Var.flux_Energy_y, Var.flux_Momx_y, Var.flux_Momy_y, \
                                                InputP);
            CheckLastCudaError();
            cudaEventRecord(kernel_end, 0);
            cudaEventSynchronize(kernel_end);
            cudaEventElapsedTime(&kernel_time[3], kernel_start, kernel_end);
        }
        else {
            // calculate primitive variables
            cudaEventRecord(kernel_start, 0);
            getPrimitive_wakeup <<<dimGrid3D, dimBlock3D>>> (Var.rho, Var.p, Var.vx, Var.vy, \
                                                Var.Mass, Var.Energy, Var.Momx, Var.Momy, \
                                                Var.metric1, \
                                                Var.wake_1, InputP);
            CheckLastCudaError();
            cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[4], kernel_start, kernel_end);
            
            // extrapolate primitive variables in time
            cudaEventRecord(kernel_start, 0);
            extrap_in_time_wakeup <<<dimGrid3D, dimBlock3D>>> (Var.rho, Var.p, Var.vx, Var.vy, \
                                                Var.rho_dx, Var.rho_dy, Var.p_dx, Var.p_dy, \
                                                Var.vx_dx, Var.vx_dy, Var.vy_dx, Var.vy_dy, \
                                                Var.rho_prime, Var.p_prime, Var.vx_prime, Var.vy_prime, \
                                                Var.metric1, Var.metric2, Var.metric3, Var.metric4, \
                                                Var.wake_1, InputP);
            CheckLastCudaError();
            cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[5], kernel_start, kernel_end);
            
            // calculate fluxes to conserved variables
            cudaEventRecord(kernel_start, 0);
            calc_flux_wakeup <<<dimGrid3D, dimBlock3D>>> (Var.rho, Var.p, Var.vx, Var.vy, \
                                                Var.rho_dx, Var.rho_dy, Var.p_dx, Var.p_dy, \
                                                Var.vx_dx, Var.vx_dy, Var.vy_dx, Var.vy_dy, \
                                                Var.rho_prime, Var.p_prime, Var.vx_prime, Var.vy_prime, \
                                                Var.flux_Mass_x, Var.flux_Energy_x, Var.flux_Momx_x, Var.flux_Momy_x, \
                                                Var.flux_Mass_y, Var.flux_Energy_y, Var.flux_Momx_y, Var.flux_Momy_y, \
                                                Var.metric1, Var.metric2, Var.metric3, Var.metric4, \
                                                Var.wake_1, InputP);
            CheckLastCudaError();
            cudaEventRecord(kernel_end, 0);
            cudaEventSynchronize(kernel_end);
            cudaEventElapsedTime(&kernel_time[6], kernel_start, kernel_end);
            
            // apply fluxes to conserved variables
            cudaEventRecord(kernel_start, 0);
            apply_flux_wakeup <<<dimGrid3D, dimBlock3D>>> (Var.Mass, Var.Energy, Var.Momx, Var.Momy,  Var.change_Mass, Var.change_Energy, Var.change_Momx, Var.change_Momy, \
                                                Var.flux_Mass_x, Var.flux_Energy_x, Var.flux_Momx_x, Var.flux_Momy_x, \
                                                Var.flux_Mass_y, Var.flux_Energy_y, Var.flux_Momx_y, Var.flux_Momy_y, \
                                                Var.metric1, Var.metric2, Var.metric3, Var.metric4, \
                                                Var.wake_1, InputP);
            CheckLastCudaError();
            cudaEventRecord(kernel_end, 0);
            cudaEventSynchronize(kernel_end);
            cudaEventElapsedTime(&kernel_time[7], kernel_start, kernel_end);
            
            /*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_start, 0);
			// Use Thrust to perform the reduction
    			// We wrap the managed memory in a thrust::device_vector
    			thrust::device_ptr<double> dev_ptr_Mass(Var.change_Mass);
    			thrust::device_ptr<double> dev_ptr_Energy(Var.change_Energy);
    			thrust::device_ptr<double> dev_ptr_Momx(Var.change_Momx);
    			thrust::device_ptr<double> dev_ptr_Momy(Var.change_Momy);
    			double change_Mass_sum = thrust::reduce(dev_ptr_Mass, dev_ptr_Mass + InputP.Nx * InputP.Ny * InputP.Nz, 0.0, thrust::plus<double>());
    			double change_Energy_sum = thrust::reduce(dev_ptr_Energy, dev_ptr_Energy + InputP.Nx * InputP.Ny * InputP.Nz, 0.0, thrust::plus<double>());
    			double change_Momx_sum = thrust::reduce(dev_ptr_Momx, dev_ptr_Momx + InputP.Nx * InputP.Ny * InputP.Nz, 0.0, thrust::plus<double>());
    			double change_Momy_sum = thrust::reduce(dev_ptr_Momy, dev_ptr_Momy + InputP.Nx * InputP.Ny * InputP.Nz, 0.0, thrust::plus<double>());
    			double change_Mass_avg = change_Mass_sum/(InputP.Nx * InputP.Ny * InputP.Nz);
    			double change_Energy_avg = change_Energy_sum/(InputP.Nx * InputP.Ny * InputP.Nz);
    			double change_Momx_avg = change_Momx_sum/(InputP.Nx * InputP.Ny * InputP.Nz);
    			double change_Momy_avg = change_Momy_sum/(InputP.Nx * InputP.Ny * InputP.Nz);
    			
    			Conservation << <dimGrid3D, dimBlock3D >> > (Var.Mass, Var.Energy, Var.Momx, Var.Momy, change_Mass_avg, change_Energy_avg, change_Momx_avg, change_Momy_avg, InputP);
    			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[8], kernel_start, kernel_end);
            
            /*-------------------Wake up-------------------*/
            cudaEventRecord(kernel_start, 0);
			//Wakeup blocks for next step
			Wakeup_Next << <dimGrid3D, dimBlock3D >> > (Var.rho, Var.Mass, \
                                                    Var.metric1, Var.metric2, Var.metric3, Var.metric4, \
                                                    Var.wake_1, Var.wake_1_next, InputP, t);
			//Wakeup neighboring blocks for next step
			Wakeup_Neighbor << <dimGrid3D, dimBlock3D >> > (Var.wake_1, Var.wake_1_next, InputP);
			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[9], kernel_start, kernel_end);
        }

        /*-------------------For performance test-------------------*/
		CUDACHECK(cudaDeviceSynchronize());
		for (int ii = 0; ii < 10; ii++){
			kernel_time_tot[ii] += kernel_time[ii];
        }

        // write vtk files
        if (t % InputP.t_freq == 0) {
            std::cout << "Timestep " << t << std::endl;

            if (!InputP.mode){
                getPrimitive <<<dimGrid3D, dimBlock3D>>> (Var.rho, Var.p, Var.vx, Var.vy, \
                                                Var.Mass, Var.Energy, Var.Momx, Var.Momy, \
                                                InputP);
                CheckLastCudaError();
            } else {
                getPrimitive_wakeup <<<dimGrid3D, dimBlock3D>>> (Var.rho, Var.p, Var.vx, Var.vy, \
                                                Var.Mass, Var.Energy, Var.Momx, Var.Momy, \
                                                Var.metric1, \
                                                Var.wake_1, InputP);
                CheckLastCudaError();
            }
            
            CUDACHECK(cudaDeviceSynchronize()); // this is necessary
            write_output_vtk(Var.rho, Var.p, Var.vx, Var.vy, InputP, t);
            if (InputP.mode)
			{
				Output_wakeup("Active", Var.wake_1, InputP, t);
			}

            sim_time_end = clock();
			time_t sim_time = sim_time_end - sim_time_start;
			Calc_RealTime_Values(Var.wake_1, InputP, t, InputP.t_freq, sim_time);
			sim_time_start = clock();
            Integral(Var.Mass, Var.Energy, Var.Momx, Var.Momy, t, InputP);
            Calc_Vy_mode(Var.vy, t, InputP);
        }
        t++;
    }
    t_end_loop = clock();

    std::cout << std::endl << "Free allocated memory..." << std::endl;
    FreeMemory(Var);

    /*-------------------For performance test-------------------*/
	t_end_tot = clock();
	printf("\nThe overall running time is: %f sec.\n", ((float)(t_end_tot - t_start_tot)) / CLOCKS_PER_SEC);

	printf("The loop running time is: %f sec. %3f percent of overall running time.\n", ((float)(t_end_loop - t_start_loop)) / CLOCKS_PER_SEC, (float)(t_end_loop - t_start_loop) / (double)(t_end_tot - t_start_tot) * 100.);

    if (!InputP.mode){
        std::cout << "Regular running time: " << std::endl;
        std::cout << "getPrimitive: " << kernel_time_tot[0] / 1000.0 << " sec." << std::endl;
        std::cout << "extrap_in_time: " << kernel_time_tot[1] / 1000.0 << " sec." << std::endl;
        std::cout << "calc_flux: " << kernel_time_tot[2] / 1000.0 << " sec." << std::endl;
        std::cout << "apply_flux: " << kernel_time_tot[3] / 1000.0 << " sec." << std::endl;
    } else {
        std::cout << "DBA running time: " << std::endl;
        std::cout << "getPrimitive: " << kernel_time_tot[4] / 1000.0 << " sec." << std::endl;
        std::cout << "extrap_in_time: " << kernel_time_tot[5] / 1000.0 << " sec." << std::endl;
        std::cout << "calc_flux: " << kernel_time_tot[6] / 1000.0 << " sec." << std::endl;
        std::cout << "apply_flux: " << kernel_time_tot[7] / 1000.0 << " sec." << std::endl;
        std::cout << "Force conservation: " << kernel_time_tot[8] / 1000.0 << " sec." << std::endl;
        std::cout << "Wakeup_Next & Wakeup_Neighbor: " << kernel_time_tot[9] / 1000.0 << " sec." << std::endl;
    }

	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_end);

    std::cout << std::endl << "Program finished." << std::endl;
    return 0;
}
