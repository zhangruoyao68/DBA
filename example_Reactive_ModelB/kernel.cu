#include "cuda_runtime.h"
#include <iostream>
#include <fstream>
#include <string>
#include <curand.h>
#include <cmath>

dim3 dimGrid3D(8, 8, 8);
dim3 dimBlock3D(4, 4, 4);
//dim3 dimGrid3D(16, 16, 1); // adjust to 2d grid
//dim3 dimBlock3D(8, 8, 1);

// For CUDA check
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

// For cuRAND error check
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

class Variable
{
	public:
        double* x_old, *y_old, *z_old, *phi_old, *b;
        double* x_new, *y_new, *z_new, *phi_new;
        double* dfdx, *dfdy, *dfdz;
        double* rand;
		double* metric;		//metric for each block
		int* wake_up, * wake_up_next;		//parameter controlling mode of blocks
};

class InputPara
{
	public:
        int Nx, Ny, Nz;
		int t_total, t_freq;
		double dx, dy, dz, dt;
        double chi_xy, chi_xz, chi_yz, chi_xb, chi_yb, chi_zb;
        int vx, vy, vz, rx, ry, rz;
        int n, m;
        double epsilonx_sq, epsilony_sq, epsilonz_sq, epsilonphi_sq;
        double Mobility_x, Mobility_y, Mobility_z, Mobility_phi;
        double x0, y0, z0;
        double K, z_crit, p_gel;
        double k_0;
		int blockNum_x, blockNum_y, blockNum_z;
		double Metric_eps;
		int mode;

		void print_input()
		{
			printf("--------------Input parameters--------------\n");
			printf("Size of system: %d, %d, %d\n", Nx, Ny, Nz);
			printf("t_total = %d, printFreq = %d, dt = %f\n", t_total, t_freq, dt);
			printf("dx = %f, dy = %f, dz =%f\n", dx, dy, dz);
            printf("chi_xy = %f, chi_xz = %f, chi_yz = %f\n", chi_xy, chi_xz, chi_yz);
            printf("chi_xb = %lf, chi_yb = %lf, chi_zb = %lf\n", chi_xb, chi_yb, chi_zb);
            printf("vx = %d, vy = %d, vz = %d\n", vx, vy, vz);
            printf("rx = %d, ry = %d, rz = %d\n", rx, ry, rz);
            printf("n = %d, m = %d\n", n, m);
            printf("epsilonx_sq = %f, epsilony_sq = %f, epsilonz_sq = %f, epsilonphi_sq = %f\n", epsilonx_sq, epsilony_sq, epsilonz_sq, epsilonphi_sq);
            printf("Mobility_x = %f, Mobility_y = %f, Mobility_z = %f, Mobility_phi = %f\n", Mobility_x, Mobility_y, Mobility_z, Mobility_phi);
            printf("x0 = %f, y0 = %f, z0 = %f\n", x0, y0, z0);
            printf("K = %f, z_crit = %f, p_gel = %f\n", K, z_crit, p_gel);
            printf("k_0 = %f\n", k_0);
			printf("Metric_eps = %f\n", Metric_eps);
			printf("mode = %d\n", mode);
            printf("-------------------------------------------\n\n");
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
	infile >> InputP.chi_xy >> InputP.chi_xz >> InputP.chi_yz >> InputP.chi_xb >> InputP.chi_yb >> InputP.chi_zb;
    std::getline(infile, space);
    infile >> InputP.vx >> InputP.vy >> InputP.vz;
    std::getline(infile, space);
    infile >> InputP.rx >> InputP.ry >> InputP.rz;
    std::getline(infile, space);
    infile >> InputP.n >> InputP.m;
    std::getline(infile, space);
    infile >> InputP.epsilonx_sq >> InputP.epsilony_sq >> InputP.epsilonz_sq >> InputP.epsilonphi_sq;
    std::getline(infile, space);
    infile >> InputP.Mobility_x >> InputP.Mobility_y >> InputP.Mobility_z >> InputP.Mobility_phi;
    std::getline(infile, space);
    infile >> InputP.x0 >> InputP.y0 >> InputP.z0;
    std::getline(infile, space);
    infile >> InputP.K >> InputP.z_crit >> InputP.p_gel;
    std::getline(infile, space);
    infile >> InputP.k_0;
	std::getline(infile, space);
	infile >> InputP.Metric_eps;
	std::getline(infile, space);
	infile >> InputP.mode;

	InputP.blockNum_x = (InputP.Nx - 1) / dimBlock3D.x + 1;
	InputP.blockNum_y = (InputP.Ny - 1) / dimBlock3D.y + 1;
	InputP.blockNum_z = (InputP.Nz - 1) / dimBlock3D.z + 1;

	/* Close input file */
	infile.close();
	std::cout << "Done with input reading." << std::endl;

	InputP.print_input();
}

//Allocate memory
void Allocate(Variable & Var, InputPara InputP)
{
	size_t size = InputP.Nx * InputP.Ny * InputP.Nz * sizeof(double);
	size_t block_size = InputP.blockNum_x * InputP.blockNum_y * InputP.blockNum_z * sizeof(int);

	CUDACHECK(cudaMallocManaged(&Var.x_old, size));
	CUDACHECK(cudaMallocManaged(&Var.y_old, size));
	CUDACHECK(cudaMallocManaged(&Var.z_old, size));
	CUDACHECK(cudaMallocManaged(&Var.phi_old, size));
	CUDACHECK(cudaMallocManaged(&Var.b, size));
	CUDACHECK(cudaMallocManaged(&Var.x_new, size));
	CUDACHECK(cudaMallocManaged(&Var.y_new, size));
	CUDACHECK(cudaMallocManaged(&Var.z_new, size));
	CUDACHECK(cudaMallocManaged(&Var.phi_new, size));
	CUDACHECK(cudaMallocManaged(&Var.dfdx, size));
	CUDACHECK(cudaMallocManaged(&Var.dfdy, size));
	CUDACHECK(cudaMallocManaged(&Var.dfdz, size));
	CUDACHECK(cudaMallocManaged(&Var.rand, size));
	CUDACHECK(cudaMallocManaged(&Var.metric, size));

	CUDACHECK(cudaMallocManaged(&Var.wake_up, block_size));
	CUDACHECK(cudaMallocManaged(&Var.wake_up_next, block_size));

    std::cout << "Total managed memory allocated: " \
             << (14*size+2*block_size)/(1024.*1024.*1024.) << " Gb\n" << std::endl;
}

//Initialize system
__global__ void Initialize(double* x, double* y, double* z, double* b, double* phi, \
							int* wake_up, int* wake_up_next, InputPara InputP)
{
	int idx, idy, idz, index;
	double amp = 0.05;
    double z_amp = 0.001;

	int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;

	for (int k = 0; k < (InputP.Nz - 1) / t_tot_z + 1; k++)
		for (int j = 0; j < (InputP.Ny - 1) / t_tot_y + 1; j++)
			for (int i = 0; i < (InputP.Nx - 1) / t_tot_x + 1; i++)
			{
				if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)	// First thread in each block
				{
					int block_index_x = blockIdx.x + i * gridDim.x;
					int block_index_y = blockIdx.y + j * gridDim.y;
					int block_index_z = blockIdx.z + k * gridDim.z;
					if (block_index_x < InputP.blockNum_x && block_index_y < InputP.blockNum_y && block_index_z < InputP.blockNum_z)
					{
						int block_index = block_index_x + block_index_y * InputP.blockNum_x \
										 + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
						wake_up[block_index] = 1;
						wake_up_next[block_index] = 0;
					}
				}
				idz = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
				idy = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
				idx = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

				if (idx < InputP.Nx && idy < InputP.Ny && idz < InputP.Nz)
				{
					index = idx + idy * InputP.Nx + idz * InputP.Nx * InputP.Ny; // Coalesced memory access

					x[index] = InputP.x0 + amp * (x[index]-0.5);
					y[index] = InputP.y0 + amp * (y[index]-0.5);
					z[index] = InputP.z0 + z_amp * (z[index]-0.5);

					/*
					// initialize 2 Z droplets
					//double Radius = sqrt((idx - NX / 2) * (idx - NX / 2) + (idy - NY / 2) * (idy - NY / 2) + (idz - NZ / 2) * (idz - NZ / 2));
					double Radius = sqrt((idx - NX /4*3) * (idx - NX /4*3) + (idy - NY / 2) * (idy - NY / 2) + (idz - NZ / 2) * (idz - NZ / 2));
					//double Radius = sqrt((idx - NX /2) * (idx - NX /2) + (idy - NY /3* 2) * (idy - NY /3* 2) + (idz - NZ / 2) * (idz - NZ / 2));
					double Radius2 = sqrt((idx - NX / 4) * (idx - NX / 4) + (idy - NY / 2) * (idy - NY / 2) + (idz - NZ / 2) * (idz - NZ / 2));
					x[idx][idy][idz] = 0.01;
					y[idx][idy][idz] = 0.01;
					
					z[idx][idy][idz] = 0.04;
					z[idx][idy][idz] += (0.9 - z_0) / 2. * tanh((Radius - 10.) * -1.) + (0.9 - z_0) / 2.;
					z[idx][idy][idz] += (0.9 - z_0) / 2. * tanh((Radius2 - 10.) * -1.) + (0.9 - z_0) / 2.;
					*/
					
					b[index] = 1.0 - x[index] - y[index] - z[index];
					//c[idx][idy][idz] = distribution(generator);

					//fphi = (double)rand() / RAND_MAX;
					phi[index] = 0;
					//phi[idx][idy][idz] = fphi * phi_amp;
					//phi[idx][idy][idz] = phi_amp;
					//phi[idx][idy][idz] += (0.5 - 0) / 2. * tanh((Radius - 10.) * -1.) + (0.5 - 0) / 2.;
					//phi[idx][idy][idz] += (0.5 - 0) / 2. * tanh((Radius2 - 10.) * -1.) + (0.5 - 0) / 2.;
				}
			}
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

	//int index_xyz = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;	// Coalesced memory access
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

__device__ double sum_absGrad(double* c, int x, int y, int z, InputPara InputP)
{
	int xm, xp, ym, yp, zm, zp;
	xm = (x - 1 >= 0) ? x - 1 : InputP.Nx - 1;
	xp = (x + 1 < InputP.Nx) ? x + 1 : 0;
	ym = (y - 1 >= 0) ? y - 1 : InputP.Ny - 1;
	yp = (y + 1 < InputP.Ny) ? y + 1 : 0;
	zm = (z - 1 >= 0) ? z - 1 : InputP.Nz - 1;
	zp = (z + 1 < InputP.Nz) ? z + 1 : 0;

	//int index_xyz = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;	// Coalesced memory access
	int index_xm = xm + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_xp = xp + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_ym = x + ym * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_yp = x + yp * InputP.Nx + z * InputP.Nx * InputP.Ny;
	int index_zm = x + y * InputP.Nx + zm * InputP.Nx * InputP.Ny;
	int index_zp = x + y * InputP.Nx + zp * InputP.Nx * InputP.Ny;

	double dpx = (c[index_xp] - c[index_xm]) / 2.0 / InputP.dx;
	double dpy = (c[index_yp] - c[index_ym]) / 2.0 / InputP.dy;
	double dpz = (c[index_zp] - c[index_zm]) / 2.0 / InputP.dz;
	
	return fabs(dpx) + fabs(dpy) + fabs(dpz);
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

__global__ void Calc_mu(double* x, double* y, double* z, double* b, double* phi, \
                        double* dfdx, double* dfdy, double* dfdz, \
                        InputPara Param)
{
	int idx, idy, idz, index;

	int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;

	for (int k = 0; k < (Param.Nz - 1) / t_tot_z + 1; k++)
		for (int j = 0; j < (Param.Ny - 1) / t_tot_y + 1; j++)
			for (int i = 0; i < (Param.Nx - 1) / t_tot_x + 1; i++)
			{
				idz = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
				idy = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
				idx = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

				if (idx < Param.Nx && idy < Param.Ny && idz < Param.Nz)
				{
					index = idx + idy * Param.Nx + idz * Param.Nx * Param.Ny; // Coalesced memory access

					b[index] = 1.0 - x[index] - y[index] - z[index];
        
					dfdx[index] = -1.0 + 1.0/Param.rx + Param.chi_xy * y[index] + Param.chi_xz * z[index] \
								+ Param.chi_xb * b[index] - Param.chi_xb * x[index] - Param.chi_yb * y[index] \
								- Param.chi_zb * z[index] + log(x[index])/Param.rx - log(b[index]) \
								- Param.epsilonx_sq * Laplacian(x, idx, idy, idz, Param);

					dfdy[index] = -1.0 + 1.0/Param.ry + Param.chi_xy * x[index] + Param.chi_yz * z[index] \
								+ Param.chi_yb * b[index] - Param.chi_xb * x[index] - Param.chi_yb * y[index] \
								- Param.chi_zb * z[index] + log(y[index])/Param.ry - log(b[index]) \
								- Param.epsilony_sq * Laplacian(y, idx, idy, idz, Param);

					dfdz[index] = -1.0 + 1.0/Param.rz + Param.chi_xz * x[index] + Param.chi_yz * y[index] \
								+ Param.chi_zb * b[index] - Param.chi_xb * x[index] - Param.chi_yb * y[index] \
								- Param.chi_zb * z[index] + log(z[index])/Param.rz - log(b[index]) \
								- Param.p_gel * phi[index] * phi[index] / 2.0 / (1.0 - Param.z_crit) \
								- Param.epsilonz_sq * Laplacian(z, idx, idy, idz, Param) \
								- log(Param.K)/Param.rz;
				}
			}
}

__global__ void Update(double* xnew, double* ynew, double* znew, double* phinew, \
                        double* xold, double* yold, double* zold, double* phiold, \
                        double* b, \
                        double* dfdx, double* dfdy, double* dfdz, \
                        InputPara Param)
{
	int idx, idy, idz, index;

	double R;

	int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;

	for (int k = 0; k < (Param.Nz - 1) / t_tot_z + 1; k++)
		for (int j = 0; j < (Param.Ny - 1) / t_tot_y + 1; j++)
			for (int i = 0; i < (Param.Nx - 1) / t_tot_x + 1; i++)
			{
				idz = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
				idy = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
				idx = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

				if (idx < Param.Nx && idy < Param.Ny && idz < Param.Nz)
				{
					index = idx + idy * Param.Nx + idz * Param.Nx * Param.Ny;	// Coalsced memory access

					R = Param.k_0 * (exp(Param.n * Param.vx * dfdx[index] + Param.m * Param.vy * dfdy[index]) \
                        -exp(Param.vz * dfdz[index]));

        			xnew[index] = xold[index] + Param.dt * Param.vx \
								* (Param.Mobility_x * Laplacian(dfdx, idx, idy, idz, Param) \
                                    - Param.n * R );
        
        			ynew[index] = yold[index] + Param.dt * Param.vy \
								* (Param.Mobility_y * Laplacian(dfdy, idx, idy, idz, Param) \
                                    - Param.m * R );
        
        			znew[index] = zold[index] + Param.dt * Param.vz \
								* (Param.Mobility_z * Laplacian(dfdz, idx, idy, idz, Param) \
                                    + R );

					b[index] = 1.0 - xnew[index] - ynew[index] - znew[index];
				}
			}
}

__global__ void Swap_regular(double* x, double* x_new, \
							double* y, double* y_new, \
							double* z, double* z_new, \
							InputPara Param)
{
	int idx, idy, idz, index;

	int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;

	for (int k = 0; k < (Param.Nz - 1) / t_tot_z + 1; k++)
		for (int j = 0; j < (Param.Ny - 1) / t_tot_y + 1; j++)
			for (int i = 0; i < (Param.Nx - 1) / t_tot_x + 1; i++)
			{
				idz = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
				idy = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
				idx = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

				if (idx < Param.Nx && idy < Param.Ny && idz < Param.Nz)
				{
					index = idx + idy * Param.Nx + idz * Param.Nx * Param.Ny;	// Coalsced memory access

					x[index] = x_new[index];
					y[index] = y_new[index];
					z[index] = z_new[index];
				}
			}
}

__global__ void Swap_wakeup(double* x, double* x_new, \
							double* y, double* y_new, \
							double* z, double* z_new, \
							int* wake_up, InputPara Param)
{
	int idx, idy, idz, index;
	int block_index, block_index_x, block_index_y, block_index_z;

	__shared__ int wake_up_s;

	int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;

	for (int k = 0; k < (Param.Nz - 1) / t_tot_z + 1; k++)
		for (int j = 0; j < (Param.Ny - 1) / t_tot_y + 1; j++)
			for (int i = 0; i < (Param.Nx - 1) / t_tot_x + 1; i++)
			{
				if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)	// First thread in each block
				{
					block_index_x = blockIdx.x + i * gridDim.x;
					block_index_y = blockIdx.y + j * gridDim.y;
					block_index_z = blockIdx.z + k * gridDim.z;
					if (block_index_x < Param.blockNum_x && block_index_y < Param.blockNum_y && block_index_z < Param.blockNum_z)
					{
						block_index = block_index_x + block_index_y * Param.blockNum_x + block_index_z * Param.blockNum_x * Param.blockNum_y;
						wake_up_s = wake_up[block_index];
					}
				}

				// Synchronize threads in threadblock before using the shared memory
				__syncthreads();

				if (wake_up_s)		// Check if this block is awake
				{
					idz = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
					idy = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
					idx = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

					if (idx < Param.Nx && idy < Param.Ny && idz < Param.Nz)
					{
						index = idx + idy * Param.Nx + idz * Param.Nx * Param.Ny;	// Coalsced memory access
						x[index] = x_new[index];
						y[index] = y_new[index];
						z[index] = z_new[index];
					}
				}
			}
}

__global__ void Calc_mu_wakeup(double* x, double* y, double* z, double* b, double* phi, \
                        		double* dfdx, double* dfdy, double* dfdz, \
								double* metric, int* wake_up, InputPara Param)
{
	int idx, idy, idz, index;
	int block_index, block_index_x, block_index_y, block_index_z;

	__shared__ int wake_up_s;

	int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;

	for (int k = 0; k < (Param.Nz - 1) / t_tot_z + 1; k++)
		for (int j = 0; j < (Param.Ny - 1) / t_tot_y + 1; j++)
			for (int i = 0; i < (Param.Nx - 1) / t_tot_x + 1; i++)
			{
				if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)	// First thread in each block
				{
					block_index_x = blockIdx.x + i * gridDim.x;
					block_index_y = blockIdx.y + j * gridDim.y;
					block_index_z = blockIdx.z + k * gridDim.z;
					if (block_index_x < Param.blockNum_x && block_index_y < Param.blockNum_y && block_index_z < Param.blockNum_z)
					{
						block_index = block_index_x + block_index_y * Param.blockNum_x + block_index_z * Param.blockNum_x * Param.blockNum_y;
						wake_up_s = wake_up[block_index];
					}
				}

				// Synchronize threads in threadblock before using the shared memory
				__syncthreads();

				if (wake_up_s)		// Check if this block is awake
				{
					idz = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
					idy = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
					idx = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

					if (idx < Param.Nx && idy < Param.Ny && idz < Param.Nz)
					{
						index = idx + idy * Param.Nx + idz * Param.Nx * Param.Ny;	// Coalsced memory access
						
						b[index] = 1.0 - x[index] - y[index] - z[index];
        
						dfdx[index] = -1.0 + 1.0/Param.rx + Param.chi_xy * y[index] + Param.chi_xz * z[index] \
									+ Param.chi_xb * b[index] - Param.chi_xb * x[index] - Param.chi_yb * y[index] \
									- Param.chi_zb * z[index] + log(x[index])/Param.rx - log(b[index]) \
									- Param.epsilonx_sq * Laplacian(x, idx, idy, idz, Param);

						dfdy[index] = -1.0 + 1.0/Param.ry + Param.chi_xy * x[index] + Param.chi_yz * z[index] \
									+ Param.chi_yb * b[index] - Param.chi_xb * x[index] - Param.chi_yb * y[index] \
									- Param.chi_zb * z[index] + log(y[index])/Param.ry - log(b[index]) \
									- Param.epsilony_sq * Laplacian(y, idx, idy, idz, Param);

						double lap_z = Laplacian(z, idx, idy, idz, Param);
						dfdz[index] = -1.0 + 1.0/Param.rz + Param.chi_xz * x[index] + Param.chi_yz * y[index] \
									+ Param.chi_zb * b[index] - Param.chi_xb * x[index] - Param.chi_yb * y[index] \
									- Param.chi_zb * z[index] + log(z[index])/Param.rz - log(b[index]) \
									- Param.p_gel * phi[index] * phi[index] / 2.0 / (1.0 - Param.z_crit) \
									- Param.epsilonz_sq * lap_z \
									- log(Param.K)/Param.rz;

						// compute different metrics
						// 1. sum of abs(Grad(c)) --> used by Greenwood and Provatas
						// 2. abs(lap_c)
						// 3. abs(mu)
						// etc.
						//metric[index] = sum_absGrad(x, idx, idy, idz, Param);
						metric[index] = fabs(lap_z);
					}
				}
			}
}

//Evovle T (for wake-up mode)
__global__ void Update_wakeup(double* xnew, double* ynew, double* znew, double* phinew, \
                        		double* xold, double* yold, double* zold, double* phiold, \
                        		double* b, \
                        		double* dfdx, double* dfdy, double* dfdz, \
						 		int* wake_up, InputPara Param)
{
	int idx, idy, idz, index;
	double R;

	int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;

	__shared__ int wake_up_s;

	for (int k = 0; k < (Param.Nz - 1) / t_tot_z + 1; k++)
		for (int j = 0; j < (Param.Ny - 1) / t_tot_y + 1; j++)
			for (int i = 0; i < (Param.Nx - 1) / t_tot_x + 1; i++)
			{
				if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)	// First thread in each block
				{
					int block_index_x = blockIdx.x + i * gridDim.x;
					int block_index_y = blockIdx.y + j * gridDim.y;
					int block_index_z = blockIdx.z + k * gridDim.z;
					int block_index = block_index_x + block_index_y * Param.blockNum_x + block_index_z * Param.blockNum_x * Param.blockNum_y;
					wake_up_s = wake_up[block_index];
				}

				// Synchronize threads in threadblock before using the shared memory
				__syncthreads();

				if (wake_up_s)		// Check if this block is awake
				{
					idz = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
					idy = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
					idx = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

					if (idx < Param.Nx && idy < Param.Ny && idz < Param.Nz)
					{
						index = idx + idy * Param.Nx + idz * Param.Nx * Param.Ny;	// Coalsced memory access

						R = Param.k_0 * (exp(Param.n * Param.vx * dfdx[index] + Param.m * Param.vy * dfdy[index]) \
                        -exp(Param.vz * dfdz[index]));

						xnew[index] = xold[index] + Param.dt * Param.vx \
									* (Param.Mobility_x * Laplacian(dfdx, idx, idy, idz, Param) \
										- Param.n * R );
			
						ynew[index] = yold[index] + Param.dt * Param.vy \
									* (Param.Mobility_y * Laplacian(dfdy, idx, idy, idz, Param) \
										- Param.m * R );
			
						znew[index] = zold[index] + Param.dt * Param.vz \
									* (Param.Mobility_z * Laplacian(dfdz, idx, idy, idz, Param) \
										+ R );

						b[index] = 1.0 - xnew[index] - ynew[index] - znew[index];
					}
				}
			}
}

//Wakeup blocks for next step
__global__ void Wakeup_Next(double* metric, int* wake_up_p, int* wake_up_p_next, InputPara InputP, int timestep)
{
	int x, y, z, index;
	int block_index, block_index_x, block_index_y, block_index_z;
	__shared__ int wake_up_p_s, wake_up_p_s_next;

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
						if (block_index_x < InputP.blockNum_x && block_index_y < InputP.blockNum_y && block_index_z < InputP.blockNum_z)
						{
							block_index = block_index_x + block_index_y * InputP.blockNum_x + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
							wake_up_p_s = wake_up_p[block_index];
							wake_up_p[block_index] = 0;
							wake_up_p_s_next = 0;
						}
					}
					// Synchronize threads in threadblock before using the shared memory
					__syncthreads();

					if (wake_up_p_s)		// Check if this block is awake
					{
						//Check if this block awake in next step
						if ((metric[index] > InputP.Metric_eps)){ // additional criterion: || (timestep < 10000)
							wake_up_p_s_next = 1;
							//std::printf("Metric = %f\n", metric[index]);
						}
						// Synchronize threads in threadblock before using the shared memory
						__syncthreads();
						if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)	// First thread in each block
						{
							if (block_index_x < InputP.blockNum_x && block_index_y < InputP.blockNum_y && block_index_z < InputP.blockNum_z)
							{
								wake_up_p_next[block_index] = wake_up_p_s_next;
							}
						}
					}
				}
			}
}

//Wakeup neighbor blocks
__global__ void Wakeup_Neighbor(int* wake_up_p, int* wake_up_p_next, InputPara InputP)
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
					if (wake_up_p_next[block_index])
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

						if (wake_up_p_next[block_index])
						{
							wake_up_p[block_index] = 1;
							wake_up_p[block_index_xm] = 1;
							wake_up_p[block_index_xp] = 1;
							wake_up_p[block_index_ym] = 1;
							wake_up_p[block_index_yp] = 1;
							wake_up_p[block_index_zm] = 1;
							wake_up_p[block_index_zp] = 1;
						}
					}
					
				}
			}		
}

//Output data in .vtk form
void write_output_vtk(double* x, double* y, double* z, double* b, double* phi, int t, InputPara Param)
{
    std::string name = "output_" + std::to_string(t) + ".vtk";
    std::ofstream ofile(name);

    int index;

    // vtk preamble
    ofile << "# vtk DataFile Version 2.0" << std::endl;
    ofile << "OUTPUT by Roy Zhang\n";
    ofile << "ASCII" << std::endl;

    // write grid
    ofile << "DATASET RECTILINEAR_GRID" << std::endl;
    ofile << "DIMENSIONS " << Param.Nx << " " << Param.Ny << " " << Param.Nz << std::endl;
    ofile << "X_COORDINATES " << Param.Nx << " int" << std::endl;
    for (int i = 0; i < Param.Nx; i++)
        ofile << i << "\t";
    ofile << std::endl;
    ofile << "Y_COORDINATES " << Param.Ny << " int" << std::endl;
    for (int i = 0; i < Param.Ny; i++)
        ofile << i << "\t";
    ofile << std::endl;
    ofile << "Z_COORDINATES " << Param.Nz << " int" << std::endl;
    for (int i = 0; i < Param.Nz; i++)
        ofile << i << "\t";
    ofile << std::endl;

    // point data
    ofile << "POINT_DATA " << Param.Nx * Param.Ny * Param.Nz << std::endl;

    // write x
    ofile << "SCALARS X double" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int k = 0; k < Param.Nz; k++){
        for (int j = 0; j < Param.Ny; j++){
            for (int i = 0; i < Param.Nx; i++){
                index = i + j * Param.Nx + k * Param.Nx * Param.Ny;
                ofile << x[index] << " ";
            }
        }
    }

    // write y
    ofile << std::endl;
    ofile << "SCALARS Y double" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int k = 0; k < Param.Nz; k++){
        for (int j = 0; j < Param.Ny; j++){
            for (int i = 0; i < Param.Nx; i++){
                index = i + j * Param.Nx + k * Param.Nx * Param.Ny;
                ofile << y[index] << " ";
            }
        }
    }
    
    // write z
    ofile << std::endl;
    ofile << "SCALARS Z double" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int k = 0; k < Param.Nz; k++){
        for (int j = 0; j < Param.Ny; j++){
            for (int i = 0; i < Param.Nx; i++){
                index = i + j * Param.Nx + k * Param.Nx * Param.Ny;
                ofile << z[index] << " ";
            }
        }
    }
    
    // write b
    ofile << std::endl;
    ofile << "SCALARS B double" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int k = 0; k < Param.Nz; k++){
        for (int j = 0; j < Param.Ny; j++){
            for (int i = 0; i < Param.Nx; i++){
                index = i + j * Param.Nx + k * Param.Nx * Param.Ny;
                ofile << b[index] << " ";
            }
        }
    }
    
    // write phi
    ofile << std::endl;
    ofile << "SCALARS PHI double" << std::endl;
    ofile << "LOOKUP_TABLE default" << std::endl;
    for (int k = 0; k < Param.Nz; k++){
        for (int j = 0; j < Param.Ny; j++){
            for (int i = 0; i < Param.Nx; i++){
                index = i + j * Param.Nx + k * Param.Nx * Param.Ny;
                ofile << phi[index] << " ";
            }
        }
    }
    ofile << std::endl;
    /*
    // Vector field point data
    // write vec
    ofile << "VECTORS vec double" << endl;
    //ofile << "LOOKUP_TABLE default" << endl;
    
    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < ny; j++) {
            for (int i = 0; i < nx; i++) {
                for (int in = 0; in < 3; in++) {
                    ofile << vec[in][i][j][k] << " ";
                }
                ofile << endl;
            }
        }
    }*/

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
	std::cout << "Writting output into " << OutputFileName << std::endl;

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

//Calculate real-time values, e.g. the velocity of main tip, wake_up portion, steps per second
void Calc_RealTime_Values(int* wake_up_p, InputPara InputP, int time, int RT_freq, time_t sim_time)
{	
	//std::cout << "--------------Writting Output of Wakeup Portion Now--------------" << std::endl;
	std::string OutputFileName;
	OutputFileName = "RTvalues.out";
	
	std::ifstream inf(OutputFileName);
	std::ofstream outf;
	if(inf.good())
	{
		outf.open(OutputFileName, std::fstream::app);
		if (!outf.is_open())
		{
			std::cout << "!!!!!Can not open" << OutputFileName << "!! Exit!!!!!!!!" << std::endl;
			exit(1);
		}	
	}
	else
	{
		outf.open(OutputFileName, std::fstream::app);
		if (!outf.is_open())
		{
			std::cout << "!!!!!Can not open" << OutputFileName << "!! Exit!!!!!!!!" << std::endl;
			exit(1);
		}
		outf << "Time Act_portion_p Steps_per_second" << std::endl;	
	}
	
	//Calculate wakeup portion
	int awake_num_p = 0;
	for (int i = 0; i < InputP.blockNum_x * InputP.blockNum_y * InputP.blockNum_z; i++)
	{
		if (wake_up_p[i])
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

//Free memory
void FreeMemory(Variable & Var)
{
	CUDACHECK(cudaFree(Var.x_old));
	CUDACHECK(cudaFree(Var.y_old));
	CUDACHECK(cudaFree(Var.z_old));
	CUDACHECK(cudaFree(Var.x_new));
	CUDACHECK(cudaFree(Var.y_new));
	CUDACHECK(cudaFree(Var.z_new));
	CUDACHECK(cudaFree(Var.phi_old));
	CUDACHECK(cudaFree(Var.phi_new));
	CUDACHECK(cudaFree(Var.b));
	CUDACHECK(cudaFree(Var.dfdx));
	CUDACHECK(cudaFree(Var.dfdy));
	CUDACHECK(cudaFree(Var.dfdz));

	CUDACHECK(cudaFree(Var.metric));
	CUDACHECK(cudaFree(Var.wake_up));
	CUDACHECK(cudaFree(Var.wake_up_next));

	std::cout << "Memory freed" << std::endl;
}

void Integral(double* x, double* y, double* z, double* b, \
			int time, InputPara InputP)
{
    std::string OutputFileName;
	OutputFileName = "Integral.out";
	std::ofstream outf;
	outf.open(OutputFileName, std::fstream::app);

	double sum_x = 0.0;
	double sum_y = 0.0;
	double sum_z = 0.0;
	double sum_b = 0.0;
    for (int index = 0; index < InputP.Nx*InputP.Ny*InputP.Nz; index++){
		sum_x += x[index];
		sum_y += y[index];
		sum_z += z[index];
		sum_b += b[index];
	}
	outf << time << " " << sum_x << " " << sum_y << " " << sum_z << " " << sum_b << std::endl;
	std::cout << "Integral of buffer = " << sum_b << std::endl;
	outf.close();
}

int main()
{	
	cudaSetDevice(0);

	/*-------------------For performance test-------------------*/
	time_t t_start_tot, t_end_tot;
	float kernel_time[6] = { 0 };
	float kernel_time_tot[6] = { 0 };
	cudaEvent_t kernel_start, kernel_end;
	cudaEventCreate(&kernel_start);
	cudaEventCreate(&kernel_end);
	t_start_tot = clock();

	InputPara InputP;
	ReadInput("Input.txt", InputP);

	Variable Var;
	Allocate(Var, InputP);

	/* Create pseudo-random number generator */
	curandGenerator_t gen;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

	/* Set seed */
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

	/* Generate n doubles on device */
	//CURAND_CALL(curandGenerateNormalDouble(gen, Var.x_old, InputP.Nx*InputP.Ny*InputP.Nz, -0.55, 0.1)); // droplets -.55, stripes -.35
	CURAND_CALL(curandGenerateUniformDouble(gen, Var.x_old, InputP.Nx*InputP.Ny*InputP.Nz));
	CURAND_CALL(curandGenerateUniformDouble(gen, Var.y_old, InputP.Nx*InputP.Ny*InputP.Nz));
	CURAND_CALL(curandGenerateUniformDouble(gen, Var.z_old, InputP.Nx*InputP.Ny*InputP.Nz));

	Initialize << <dimGrid3D, dimBlock3D >> > (Var.x_old, Var.y_old, Var.z_old, Var.b, Var.phi_old, \
											Var.wake_up, Var.wake_up_next, InputP);
	CheckLastCudaError();

	CUDACHECK(cudaDeviceSynchronize());
	write_output_vtk(Var.x_old, Var.y_old, Var.z_old, Var.b, Var.phi_old, 0, InputP);
	Integral(Var.x_old, Var.y_old, Var.z_old, Var.b, 0, InputP);

	/*-------------------For performance test-------------------*/
	time_t t_start_loop, t_end_loop, sim_time_start, sim_time_end;
	t_start_loop = clock();
	sim_time_start = clock();

	int t = 1;
    while (t <= InputP.t_total) {
		//Check the mode of computing
		if (!InputP.mode)
		{
			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_start, 0);
			//For regular mode evolution of p and T
			Calc_mu << <dimGrid3D, dimBlock3D >> > (Var.x_old, Var.y_old, Var.z_old, Var.b, Var.phi_old, \
													Var.dfdx, Var.dfdy, Var.dfdz, InputP);
			cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[0], kernel_start, kernel_end);

			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_start, 0);
			//For regular mode evolution of p and T
			Update << <dimGrid3D, dimBlock3D >> > (Var.x_new, Var.y_new, Var.z_new, Var.phi_new, \
													Var.x_old, Var.y_old, Var.z_old, Var.phi_old, Var.b, \
													Var.dfdx, Var.dfdy, Var.dfdz, InputP);
			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[1], kernel_start, kernel_end);
			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_start, 0);
			Swap_regular << <dimGrid3D, dimBlock3D >> > (Var.x_old, Var.x_new, \
													Var.y_old, Var.y_new, \
													Var.z_old, Var.z_new, \
													InputP);
			cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[2], kernel_start, kernel_end);
		}
		else
		{
			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_start, 0);
			//For wake-up mode evolution of p and T, T_new stores grad p
			Calc_mu_wakeup << <dimGrid3D, dimBlock3D >> > (Var.x_old, Var.y_old, Var.z_old, Var.b, Var.phi_old, \
													Var.dfdx, Var.dfdy, Var.dfdz, \
													Var.metric, Var.wake_up, InputP);
			cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[0], kernel_start, kernel_end);

			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_start, 0);
			Update_wakeup << <dimGrid3D, dimBlock3D >> > (Var.x_new, Var.y_new, Var.z_new, Var.phi_new, \
													Var.x_old, Var.y_old, Var.z_old, Var.phi_old, Var.b, \
													Var.dfdx, Var.dfdy, Var.dfdz,\
													Var.wake_up, InputP);
			cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[1], kernel_start, kernel_end);

			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_start, 0);
			Swap_wakeup << <dimGrid3D, dimBlock3D >> > (Var.x_old, Var.x_new, \
													Var.y_old, Var.y_new, \
													Var.z_old, Var.z_new, \
													Var.wake_up, InputP);
			cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[2], kernel_start, kernel_end);
			
			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_start, 0);
			//Wakeup blocks for next step
			Wakeup_Next << <dimGrid3D, dimBlock3D >> > (Var.metric, Var.wake_up, Var.wake_up_next, InputP, t);
			//Wakeup neighboring blocks for next step
			Wakeup_Neighbor << <dimGrid3D, dimBlock3D >> > (Var.wake_up, Var.wake_up_next, InputP);
			cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[3], kernel_start, kernel_end);
		}

		/*-------------------For performance test-------------------*/
		CUDACHECK(cudaDeviceSynchronize());
		for (int ii = 0; ii < 6; ii++){
			kernel_time_tot[ii] += kernel_time[ii];
		}
		
		//Calculate the real-time values
		int RT_freq = InputP.t_freq;//1000;		//Frequency of calculation
		if (t %  RT_freq == 0)
		{	
			std::cout << "Timestep " << t << std::endl;
			CUDACHECK(cudaDeviceSynchronize());
			sim_time_end = clock();
			time_t sim_time = sim_time_end - sim_time_start;
			Calc_RealTime_Values(Var.wake_up, InputP, t, RT_freq, sim_time);
			sim_time_start = clock();

			Integral(Var.x_old, Var.y_old, Var.z_old, Var.b, t, InputP);

			// write .vtk files
			write_output_vtk(Var.x_old, Var.y_old, Var.z_old, Var.b, Var.phi_old, t, InputP);
			if (InputP.mode)
			{
				Output_wakeup("Wakeup", Var.wake_up, InputP, t);
			}
		}
		t++;
	}

	FreeMemory(Var);

	/*-------------------For performance test-------------------*/
	t_end_loop = clock();
	t_end_tot = clock();
	
	if (!InputP.mode){
		printf("\nPefromance test for Regular mode:\n");
		printf("The overall running time is: %f sec.\n", ((float)(t_end_tot - t_start_tot)) / CLOCKS_PER_SEC);
		printf("The loop running time is: %f sec. %3f percent of overall running time.\n", ((float)(t_end_loop - t_start_loop)) / CLOCKS_PER_SEC, (float)(t_end_loop - t_start_loop) / (double)(t_end_tot - t_start_tot) * 100.);
		printf("Calc_mu - %f sec\nupdate_c - %f sec\n",
			(float)(kernel_time_tot[0]) / 1000.0, (float)(kernel_time_tot[1]) / 1000.0);
		printf("Swap_regular (memory copy) - %f sec.\n", (float)(kernel_time_tot[2]) / 1000.0);
	}
	else{
		printf("\nPefromance test for DBA mode:\n");
		printf("The overall running time is: %f sec.\n", ((float)(t_end_tot - t_start_tot)) / CLOCKS_PER_SEC);
		printf("The loop running time is: %f sec. %3f percent of overall running time.\n", ((float)(t_end_loop - t_start_loop)) / CLOCKS_PER_SEC, (float)(t_end_loop - t_start_loop) / (double)(t_end_tot - t_start_tot) * 100.);
		printf("Calc_mu_wakeup - %f sec\nUpdate_wakeup - %f sec\n",
			(float)(kernel_time_tot[0]) / 1000.0, (float)(kernel_time_tot[1]) / 1000.0);
		printf("Swap_wakeup (memory copy) - %f sec.\n", (float)(kernel_time_tot[2]) / 1000.0);
		printf("Wakeup_Next & Neighbor - %f sec.\n", (float)(kernel_time_tot[3]) / 1000.0);

	}
	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_end);

	CURAND_CALL(curandDestroyGenerator(gen));

	return 0;
}
