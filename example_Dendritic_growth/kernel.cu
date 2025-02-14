
#include "cuda_runtime.h"
#include <iostream>
#include <fstream>
#include <string>

#define Pi 3.14159
dim3 dimGrid3D(8, 8, 8);
dim3 dimBlock3D(4, 4, 4);

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

class Variable
{
	public:
		double* p, *T, *p_new, *T_new;
		double* metric_p, *metric_T;
		double* lapP, *lapT;
		int* wake_up_p, * wake_up_p_next, * wake_up_T, * wake_up_T_next;		//parameter controlling mode of blocks
};

class InputPara
{
	public:
		int Nx, Ny, Nz;
		int t_total, t_freq;
		double dx, dy, dz, dt;
		double delta, tau, epsilon, alpha, gamma;
		double Delta;
		double factors[4];
		int blockNum_x, blockNum_y, blockNum_z;
		int mode;
		double metric_p, metric_T;

		void print_input()
		{
			printf("--------------Input parameters--------------\n");
			printf("Size of system: %d, %d, %d\n", Nx, Ny, Nz);
			printf("totalTime = %d,	printFreq = %d, dt = %lf\n", t_total, t_freq, dt);
			printf("dx = %lf, dy = %lf, dz = %lf\n", dx, dy, dz);
			printf("delta = %lf, tau = %lf,	epsilon = %lf, alpha = %lf, gamma = %lf\n", delta, tau, epsilon, alpha, gamma);
			printf("Delta = %lf\n", Delta);
			printf("Mode = %d (0 for regular while 1 for wake-up mode)\n", mode);
			printf("metric_p = %lf, metric_T = %lf\n", metric_p, metric_T);
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
	infile >> InputP.delta >> InputP.tau >> InputP.epsilon >> InputP.alpha >> InputP.gamma;
	std::getline(infile, space);
	infile >> InputP.Delta;
	std::getline(infile, space);
	infile >> InputP.mode;
	std::getline(infile, space);
	infile >> InputP.metric_p;
	std::getline(infile, space);
	infile >> InputP.metric_T;

	InputP.factors[0] = 1. / (2 * InputP.dx);
	InputP.factors[1] = 1. / InputP.dx / InputP.dx;
	InputP.factors[2] = -InputP.alpha / Pi;
	InputP.factors[3] = 1. / InputP.tau;
	InputP.blockNum_x = (InputP.Nx - 1) / dimBlock3D.x + 1;
	InputP.blockNum_y = (InputP.Ny - 1) / dimBlock3D.y + 1;
	InputP.blockNum_z = (InputP.Nz - 1) / dimBlock3D.z + 1;

	/* Close input file */
	infile.close();
	std::cout << "Done with input reading.\n\n";

	InputP.print_input();

}

//Allocate memory
void Allocate(Variable & Var, InputPara InputP)
{
	size_t size = InputP.Nx * InputP.Ny * InputP.Nz * sizeof(double);
	size_t block_size = InputP.blockNum_x * InputP.blockNum_y * InputP.blockNum_z * sizeof(int);

	CUDACHECK(cudaMallocManaged(&Var.p, size));
	CUDACHECK(cudaMallocManaged(&Var.T, size));
	CUDACHECK(cudaMallocManaged(&Var.p_new, size));
	CUDACHECK(cudaMallocManaged(&Var.T_new, size));
	CUDACHECK(cudaMallocManaged(&Var.lapP, size));
	CUDACHECK(cudaMallocManaged(&Var.lapT, size));
	CUDACHECK(cudaMallocManaged(&Var.metric_p, size));
	CUDACHECK(cudaMallocManaged(&Var.metric_T, size));

	CUDACHECK(cudaMallocManaged(&Var.wake_up_p, block_size));
	CUDACHECK(cudaMallocManaged(&Var.wake_up_T, block_size));
	CUDACHECK(cudaMallocManaged(&Var.wake_up_p_next, block_size));
	CUDACHECK(cudaMallocManaged(&Var.wake_up_T_next, block_size));

	std::cout << "Total memory allocated: " << (size * 8 + block_size * 4) / (1024.0*1024.0*1024.0) << " GB" << std::endl;
}

//Initialize system
__global__ void Initialize(double* p, double* T, int* wake_up_p, int* wake_up_T, int* wake_up_p_next, int* wake_up_T_next, InputPara InputP)
{
	int x, y, z, index;
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
						int block_index = block_index_x + block_index_y * InputP.blockNum_x + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
						wake_up_p[block_index] = 1;
						wake_up_T[block_index] = 1;
						wake_up_p_next[block_index] = 0;
						wake_up_T_next[block_index] = 0;
					}
				}
				z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
				y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
				x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

				if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz)
				{
					index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;	// Coalsced memory access
					double dist = sqrt((double)((x - InputP.Nx / 2) * (x - InputP.Nx / 2) + (y - InputP.Ny / 2) * (y - InputP.Ny / 2) + (z - InputP.Nz / 2) * (z - InputP.Nz / 2)));
					if (dist <= 5)
						p[index] = (1. - 0.2 * dist);
					else
						p[index] = 0;
					T[index] = -InputP.Delta;
				}
			}
}

//Evolve p (for regular mode)
__global__ void Evolve_p(double* p, double* T, double* p_new, InputPara InputP)
{
	int x, y, z, index;
	int xm, xp, ym, yp, zm, zp;
	int index_xm, index_xp, index_ym, index_yp, index_zm, index_zp;
	double dpx, dpy, dpz, lapP;
	double v2, sigma, m, p_change;

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

					//Finite Difference
					xm = (x - 1 >= 0) ? x - 1 : InputP.Nx - 1;
					xp = (x + 1 < InputP.Nx) ? x + 1 : 0;
					ym = (y - 1 >= 0) ? y - 1 : InputP.Ny - 1;
					yp = (y + 1 < InputP.Ny) ? y + 1 : 0;
					zm = (z - 1 >= 0) ? z - 1 : InputP.Nz - 1;
					zp = (z + 1 < InputP.Nz) ? z + 1 : 0;

					index_xm = xm + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
					index_xp = xp + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
					index_ym = x + ym * InputP.Nx + z * InputP.Nx * InputP.Ny;
					index_yp = x + yp * InputP.Nx + z * InputP.Nx * InputP.Ny;
					index_zm = x + y * InputP.Nx + zm * InputP.Nx * InputP.Ny;
					index_zp = x + y * InputP.Nx + zp * InputP.Nx * InputP.Ny;

					dpx = (p[index_xp] - p[index_xm]) * InputP.factors[0];
					dpy = (p[index_yp] - p[index_ym]) * InputP.factors[0];
					dpz = (p[index_zp] - p[index_zm]) * InputP.factors[0];

					lapP = (p[index_xp] + p[index_xm] + p[index_yp] + p[index_ym] + p[index_zp] + p[index_zm] - 6 * p[index]) * InputP.factors[1];

					//Update p and T
					//Pre-calculations
					v2 = pow(dpx, 2) + pow(dpy, 2) + pow(dpz, 2);
					if (v2 <= 1E-6)
						sigma = 1;
					else
						sigma = 1 - InputP.delta * (1 - (pow(dpx, 4) + pow(dpy, 4) + pow(dpz, 4))
							/ pow(v2, 2));
					m = InputP.factors[2] * atan(InputP.gamma * sigma * T[index]);

					//Calculate dp/dt and dT/dt
					p_change = InputP.factors[3] * (InputP.epsilon * InputP.epsilon * lapP
						+ p[index] * (1 - p[index]) * (p[index] - 0.5 + m));

					//Update p and T
					p_new[index] = p[index] + InputP.dt * p_change;
				}
			}
}
//Evolve T (for regular mode)
__global__ void Evolve_T(double* p, double* p_new, double* T, double* T_new, InputPara InputP)
{
	int x, y, z, index;
	int xm, xp, ym, yp, zm, zp;
	int index_xm, index_xp, index_ym, index_yp, index_zm, index_zp;
	double lapT;

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

					//Finite Difference

					xm = (x - 1 >= 0) ? x - 1 : InputP.Nx - 1;
					xp = (x + 1 < InputP.Nx) ? x + 1 : 0;
					ym = (y - 1 >= 0) ? y - 1 : InputP.Ny - 1;
					yp = (y + 1 < InputP.Ny) ? y + 1 : 0;
					zm = (z - 1 >= 0) ? z - 1 : InputP.Nz - 1;
					zp = (z + 1 < InputP.Nz) ? z + 1 : 0;

					index_xm = xm + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
					index_xp = xp + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
					index_ym = x + ym * InputP.Nx + z * InputP.Nx * InputP.Ny;
					index_yp = x + yp * InputP.Nx + z * InputP.Nx * InputP.Ny;
					index_zm = x + y * InputP.Nx + zm * InputP.Nx * InputP.Ny;
					index_zp = x + y * InputP.Nx + zp * InputP.Nx * InputP.Ny;

					lapT = (T[index_xp] + T[index_xm] + T[index_yp] + T[index_ym] + T[index_zp] + T[index_zm] - 6 * T[index]) * InputP.factors[1];

					//Update p and T
					T_new[index] = T[index] + InputP.dt * lapT + p_new[index] - p[index];

				}
			}
}
//Update p and T (for regular mode only, wakeup mode update has been incorporated in Wakeup_next)
__global__ void Update_p_and_T_regular(double* p, double* p_new, double* T, double* T_new, InputPara InputP)
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
					index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;	// Coalsced memory access

					T[index] = T_new[index];
					p[index] = p_new[index];
				}
			}
}

//Evolve p (for wake-up mode)
__global__ void Evolve_p_wakeup(double* p, double* T, double* p_new, double* metric_p, int* wake_up_p, InputPara InputP)
{
	int x, y, z, index;
	int xm, xp, ym, yp, zm, zp;
	int index_xm, index_xp, index_ym, index_yp, index_zm, index_zp;
	double dpx, dpy, dpz, lapP;
	double v2, sigma, m, p_change;
	int block_index, block_index_x, block_index_y, block_index_z;

	__shared__ int wake_up_p_s;

	int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;

	for (int k = 0; k < (InputP.Nz - 1) / t_tot_z + 1; k++)
		for (int j = 0; j < (InputP.Ny - 1) / t_tot_y + 1; j++)
			for (int i = 0; i < (InputP.Nx - 1) / t_tot_x + 1; i++)
			{
				if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)	// First thread in each block
				{
					block_index_x = blockIdx.x + i * gridDim.x;
					block_index_y = blockIdx.y + j * gridDim.y;
					block_index_z = blockIdx.z + k * gridDim.z;
					if (block_index_x < InputP.blockNum_x && block_index_y < InputP.blockNum_y && block_index_z < InputP.blockNum_z)
					{
						block_index = block_index_x + block_index_y * InputP.blockNum_x + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
						wake_up_p_s = wake_up_p[block_index];
					}
				}

				// Synchronize threads in threadblock before using the shared memory
				__syncthreads();

				if (wake_up_p_s)		// Check if this block is awake
				{
					z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
					y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
					x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

					if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz)
					{
						index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;	// Coalsced memory access

						//Finite Difference
						xm = (x - 1 >= 0) ? x - 1 : InputP.Nx - 1;
						xp = (x + 1 < InputP.Nx) ? x + 1 : 0;
						ym = (y - 1 >= 0) ? y - 1 : InputP.Ny - 1;
						yp = (y + 1 < InputP.Ny) ? y + 1 : 0;
						zm = (z - 1 >= 0) ? z - 1 : InputP.Nz - 1;
						zp = (z + 1 < InputP.Nz) ? z + 1 : 0;

						index_xm = xm + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
						index_xp = xp + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
						index_ym = x + ym * InputP.Nx + z * InputP.Nx * InputP.Ny;
						index_yp = x + yp * InputP.Nx + z * InputP.Nx * InputP.Ny;
						index_zm = x + y * InputP.Nx + zm * InputP.Nx * InputP.Ny;
						index_zp = x + y * InputP.Nx + zp * InputP.Nx * InputP.Ny;

						dpx = (p[index_xp] - p[index_xm]) * InputP.factors[0];
						dpy = (p[index_yp] - p[index_ym]) * InputP.factors[0];
						dpz = (p[index_zp] - p[index_zm]) * InputP.factors[0];

						lapP = (p[index_xp] + p[index_xm] + p[index_yp] + p[index_ym] + p[index_zp] + p[index_zm] - 6 * p[index]) * InputP.factors[1];

						//Update p and T
						//Pre-calculations
						v2 = pow(dpx, 2) + pow(dpy, 2) + pow(dpz, 2);
						if (v2 <= 1E-6)
							sigma = 1;
						else
							sigma = 1 - InputP.delta * (1 - (pow(dpx, 4) + pow(dpy, 4) + pow(dpz, 4))
								/ pow(v2, 2));
						m = InputP.factors[2] * atan(InputP.gamma * sigma * T[index]);

						//Calculate dp/dt and dT/dt
						p_change = InputP.factors[3] * (InputP.epsilon * InputP.epsilon * lapP
							+ p[index] * (1 - p[index]) * (p[index] - 0.5 + m));

						//Update p and T
						p_new[index] = p[index] + InputP.dt * p_change;

						metric_p[index] = abs(p_new[index] - p[index]);
					}
				}
			}
}

//Evovle T (for wake-up mode)
__global__ void Evolve_T_wakeup(double* p, double* p_new, double* T, double* T_new, double* metric_T, int* wake_up_p, int* wake_up_T, InputPara InputP)
{
	int x, y, z, index;
	int xm, xp, ym, yp, zm, zp;
	int index_xm, index_xp, index_ym, index_yp, index_zm, index_zp;
	int block_index, block_index_x, block_index_y, block_index_z;
	double lapT;

	int t_tot_x = blockDim.x * gridDim.x;
	int t_tot_y = blockDim.y * gridDim.y;
	int t_tot_z = blockDim.z * gridDim.z;

	__shared__ int wake_up_p_s, wake_up_T_s;

	for (int k = 0; k < (InputP.Nz - 1) / t_tot_z + 1; k++)
		for (int j = 0; j < (InputP.Ny - 1) / t_tot_y + 1; j++)
			for (int i = 0; i < (InputP.Nx - 1) / t_tot_x + 1; i++)
			{
				if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)	// First thread in each block
				{
					block_index_x = blockIdx.x + i * gridDim.x;
					block_index_y = blockIdx.y + j * gridDim.y;
					block_index_z = blockIdx.z + k * gridDim.z;
					block_index = block_index_x + block_index_y * InputP.blockNum_x + block_index_z * InputP.blockNum_x * InputP.blockNum_y;
					wake_up_p_s = wake_up_p[block_index];
					wake_up_T_s = wake_up_T[block_index];
				}

				// Synchronize threads in threadblock before using the shared memory
				__syncthreads();

				if (wake_up_T_s)			//Check if this block awake by T
				{
					z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
					y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
					x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

					if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz)
					{
						index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;	// Coalsced memory access

						//Finite Difference

						xm = (x - 1 >= 0) ? x - 1 : InputP.Nx - 1;
						xp = (x + 1 < InputP.Nx) ? x + 1 : 0;
						ym = (y - 1 >= 0) ? y - 1 : InputP.Ny - 1;
						yp = (y + 1 < InputP.Ny) ? y + 1 : 0;
						zm = (z - 1 >= 0) ? z - 1 : InputP.Nz - 1;
						zp = (z + 1 < InputP.Nz) ? z + 1 : 0;

						index_xm = xm + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
						index_xp = xp + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
						index_ym = x + ym * InputP.Nx + z * InputP.Nx * InputP.Ny;
						index_yp = x + yp * InputP.Nx + z * InputP.Nx * InputP.Ny;
						index_zm = x + y * InputP.Nx + zm * InputP.Nx * InputP.Ny;
						index_zp = x + y * InputP.Nx + zp * InputP.Nx * InputP.Ny;

						lapT = (T[index_xp] + T[index_xm] + T[index_yp] + T[index_ym] + T[index_zp] + T[index_zm] - 6 * T[index]) * InputP.factors[1];

						//Update T
						T_new[index] = T[index] + InputP.dt * lapT;

						metric_T[index] = abs(T_new[index] - T[index]);
						//metric_T[index] = fabs(lapT);
					}
				}

				if (wake_up_p_s)		// Check if this block is awake by p
				{
					z = blockIdx.z * blockDim.z + threadIdx.z + k * t_tot_z;
					y = blockIdx.y * blockDim.y + threadIdx.y + j * t_tot_y;
					x = blockIdx.x * blockDim.x + threadIdx.x + i * t_tot_x;

					if (x < InputP.Nx && y < InputP.Ny && z < InputP.Nz)
					{
						index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;	// Coalsced memory access
						T_new[index] += p_new[index] - p[index];

						metric_T[index] = abs(T_new[index] - T[index]);
					}
				}
			}
}

//Wakeup blocks for next step
__global__ void Wakeup_Next(double* p, double* T, double* p_new, double* T_new, double* metric_p, double* metric_T, \
							int* wake_up_p, int* wake_up_T, int* wake_up_p_next, int* wake_up_T_next, InputPara InputP)
{
	int x, y, z, index;
	int block_index, block_index_x, block_index_y, block_index_z;
	__shared__ int wake_up_p_s, wake_up_p_s_next, wake_up_T_s, wake_up_T_s_next;

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
							wake_up_T_s = wake_up_T[block_index];
							wake_up_p[block_index] = 0;
							wake_up_T[block_index] = 0;
							wake_up_p_s_next = 0;
							wake_up_T_s_next = 0;
						}
					}
					// Synchronize threads in threadblock before using the shared memory
					__syncthreads();

					if (wake_up_p_s)		// Check if this block is awake
					{
						//Update p for next step, so we don't need to do the memcpy
						p[index] = p_new[index];
						//Check if this block awake in next step
						if (p_new[index] >= InputP.metric_p && p_new[index] <= (1.0-InputP.metric_p) && wake_up_p_s_next == 0)
							wake_up_p_s_next = 1;
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

					if (wake_up_T_s)		// Check if this block is awake
					{
						//Update T for next step, so we don't need to do the memcpy
						T[index] = T_new[index];
						//Check if this block awake in next step
						if (metric_T[index] >= InputP.metric_T && wake_up_T_s_next == 0)
							wake_up_T_s_next = 1;
						// Synchronize threads in threadblock before using the shared memory
						__syncthreads();
						if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)	// First thread in each block
						{
							if (block_index_x < InputP.blockNum_x && block_index_y < InputP.blockNum_y && block_index_z < InputP.blockNum_z)
							{
								wake_up_T_next[block_index] = wake_up_T_s_next;
							}
						}
					}
				}
			}
}

//Wakeup neighbor blocks
__global__ void Wakeup_Neighbor(int* wake_up_p, int* wake_up_T, int* wake_up_p_next, int* wake_up_T_next, InputPara InputP)
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
					if (wake_up_p_next[block_index] || wake_up_T_next[block_index])
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
						if (wake_up_T_next[block_index])
						{
							wake_up_T[block_index] = 1;
							wake_up_T[block_index_xm] = 1;
							wake_up_T[block_index_xp] = 1;
							wake_up_T[block_index_ym] = 1;
							wake_up_T[block_index_yp] = 1;
							wake_up_T[block_index_zm] = 1;
							wake_up_T[block_index_zp] = 1;
						}
					}
					
				}
			}		
}

//Output data in .vtk form
void Output_vtk(std::string prefix, double* p, double* T, InputPara Param, int time)
{
	//std::cout << "--------------Writting Output Now--------------" << std::endl;
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

	int index;

    // vtk preamble
    outf << "# vtk DataFile Version 2.0" << std::endl;
    outf << "OUTPUT by Roy Zhang\n";
    outf << "ASCII" << std::endl;

    // write grid
    outf << "DATASET RECTILINEAR_GRID" << std::endl;
    outf << "DIMENSIONS " << Param.Nx << " " << Param.Ny << " " << Param.Nz << std::endl;
    outf << "X_COORDINATES " << Param.Nx << " int" << std::endl;
    for (int i = 0; i < Param.Nx; i++)
        outf << i << "\t";
    outf << std::endl;
    outf << "Y_COORDINATES " << Param.Ny << " int" << std::endl;
    for (int i = 0; i < Param.Ny; i++)
        outf << i << "\t";
    outf << std::endl;
    outf << "Z_COORDINATES " << Param.Nz << " int" << std::endl;
    for (int i = 0; i < Param.Nz; i++)
        outf << i << "\t";
    outf << std::endl;

    // point data
    outf << "POINT_DATA " << Param.Nx * Param.Ny * Param.Nz << std::endl;

    // write p
    outf << "SCALARS p double" << std::endl;
    outf << "LOOKUP_TABLE default" << std::endl;
    for (int k = 0; k < Param.Nz; k++){
        for (int j = 0; j < Param.Ny; j++){
            for (int i = 0; i < Param.Nx; i++){
                index = i + j * Param.Nx + k * Param.Nx * Param.Ny;
                outf << p[index] << " ";
            }
        }
    }

    // write T
    outf << std::endl;
    outf << "SCALARS T double" << std::endl;
    outf << "LOOKUP_TABLE default" << std::endl;
    for (int k = 0; k < Param.Nz; k++){
        for (int j = 0; j < Param.Ny; j++){
            for (int i = 0; i < Param.Nx; i++){
                index = i + j * Param.Nx + k * Param.Nx * Param.Ny;
                outf << T[index] << " ";
            }
        }
    }

    outf.close();
	//std::cout << "--------------Output Done--------------" << std::endl;
}


//Output wakeup parameter in .vtk form, as points rather than blocks
void Output_wakeup(std::string prefix, int* p, int* T, InputPara InputP, int time)
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
	outf << "SCALARS wakeup_p int" << std::endl << "LOOKUP_TABLE default" << std::endl;
	for (int z = 0; z < InputP.Nz; z++)
		for (int y = 0; y < InputP.Ny; y++)
			for (int x = 0; x < InputP.Nx; x++)
			{
				int block_x = x / dimBlock3D.x;
				int block_y = y / dimBlock3D.y;
				int block_z = z / dimBlock3D.z;
				int index_block = block_x + block_y * InputP.blockNum_x + block_z * InputP.blockNum_x * InputP.blockNum_y;
				outf << p[index_block] << " ";
			}
	outf << std::endl;
	outf << "SCALARS wakeup_T int" << std::endl << "LOOKUP_TABLE default" << std::endl;
	for (int z = 0; z < InputP.Nz; z++)
		for (int y = 0; y < InputP.Ny; y++)
			for (int x = 0; x < InputP.Nx; x++)
			{
				int block_x = x / dimBlock3D.x;
				int block_y = y / dimBlock3D.y;
				int block_z = z / dimBlock3D.z;
				int index_block = block_x + block_y * InputP.blockNum_x + block_z * InputP.blockNum_x * InputP.blockNum_y;
				outf << T[index_block] << " ";
			}
	outf.close();
	std::cout << "--------------Output Done--------------" << std::endl;
}

double growth_front(double* phi, int ix, InputPara Param)
{
    double tip_loc;            // y cooridinate for tip
	int z = Param.Nz / 2;
    for (int y = Param.Ny/2 ; y < Param.Ny; y++){
		int index = ix + y * Param.Nx + z * Param.Nx * Param.Ny;
        if (phi[index] < 0.5){
			int index_ym = ix + (y - 1) * Param.Nx + z * Param.Nx * Param.Ny;
            tip_loc = (phi[index_ym] - 0.5) / (phi[index_ym] - phi[index]) + y - 1 - Param.Ny/2;
            break;
        }
    }

    return tip_loc;
}

void tip_curvature(double* phi, int n_points, std::string file_name, InputPara Param){

    // append data to the file
    std::ofstream ofile(file_name, std::fstream::app);

    double tip;

    for (int ix = Param.Nx/2 - n_points ; ix <= Param.Nx/2 + n_points; ix++){
        tip = growth_front(phi, ix, Param);
        ofile << tip << " ";
    }
    ofile << std::endl;
    ofile.close();
}

//Calculate real-time values, e.g. the velocity of main tip, wake_up portion, steps per second
void Calc_RealTime_Values(std::string prefix, int* wake_up_p, int* wake_up_T, double* p, InputPara InputP, int time, int RT_freq, double & prev_dist, time_t sim_time)
{
	//std::cout << "--------------Writting Output of Velocity Now--------------" << std::endl;
	std::string OutputFileName;
	OutputFileName = prefix + ".out";
	
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
		outf << "Time Vel Act_portion_p Act_portion_T Steps_per_second" << std::endl;	
	}

	// Calculate tip velocity
	double dist = 0;

	/*
	for (int z = 0; z < InputP.Nz; z++)
		for (int y = 0; y < InputP.Ny; y++)
			for (int x = 0; x < InputP.Nx; x++)
			{
				int index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
				if (p[index] >= 0.5)
				{
					dist = std::max(dist, sqrt((x - InputP.Nx / 2) * (x - InputP.Nx / 2)
						+ (y - InputP.Ny / 2) * (y - InputP.Ny / 2)
						+ (z - InputP.Nz / 2) * (z - InputP.Nz / 2)));
				}
			}*/
	int x_half = InputP.Nx / 2;
	dist = growth_front(p, x_half, InputP);
	double vel = (dist - prev_dist) / RT_freq;
	prev_dist = dist;

	//Calculate wakeup portion
	int awake_num_p = 0;
	int awake_num_T = 0;
	for (int i = 0; i < InputP.blockNum_x * InputP.blockNum_y * InputP.blockNum_z; i++)
	{
		if (wake_up_p[i])
			awake_num_p++;
		if (wake_up_T[i])
			awake_num_T++;
	}
	double awake_portion_p = (double)awake_num_p / (InputP.blockNum_x * InputP.blockNum_y * InputP.blockNum_z);
	double awake_portion_T = (double)awake_num_T / (InputP.blockNum_x * InputP.blockNum_y * InputP.blockNum_z);
	printf("awake_p: %f, awake_T: %f\n", awake_portion_p, awake_portion_T);

	//Calculate steps per second
	double steps_per_second = RT_freq / ((double)sim_time / CLOCKS_PER_SEC);

	outf << time << " " << vel << " " << awake_portion_p << " " << awake_portion_T << " " << steps_per_second << std::endl;
	outf.close();
	//std::cout << "--------------Output Done--------------" << std::endl;

	// extract tip profile
	tip_curvature(p, 10, "tip_profile.out", InputP);
}

//Free memory
void FreeMemory(Variable & Var)
{
	CUDACHECK(cudaFree(Var.p));
	CUDACHECK(cudaFree(Var.T));
	CUDACHECK(cudaFree(Var.p_new));
	CUDACHECK(cudaFree(Var.T_new));
	CUDACHECK(cudaFree(Var.lapP));
	CUDACHECK(cudaFree(Var.lapT));

	CUDACHECK(cudaFree(Var.wake_up_p));
	CUDACHECK(cudaFree(Var.wake_up_T));
	CUDACHECK(cudaFree(Var.wake_up_p_next));
	CUDACHECK(cudaFree(Var.wake_up_T_next));
}

int main()
{
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

	Initialize << <dimGrid3D, dimBlock3D >> > (Var.p, Var.T, Var.wake_up_p, Var.wake_up_T, Var.wake_up_p_next, Var.wake_up_T_next, InputP);

	CheckLastCudaError();

	//CUDACHECK(cudaDeviceSynchronize());
	//Output_vtk("p", Var.p, InputP, 0);
	//Output_vtk("T", Var.T, InputP, 0);

	/*-------------------For performance test-------------------*/
	time_t t_start_loop, t_end_loop, sim_time_start, sim_time_end;
	t_start_loop = clock();
	sim_time_start = clock();

	double dist = 0;		//Initial distance of tip

	for (int t = 1; t < InputP.t_total + 1; t++)
	{
		//Check the mode of computing
		if (!InputP.mode)
		{
			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_start, 0);
			//For regular mode evolution of p and T
			Evolve_p << <dimGrid3D, dimBlock3D >> > (Var.p, Var.T, Var.p_new, InputP);
			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[0], kernel_start, kernel_end);

			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_start, 0);
			//For regular mode evolution of p and T
			Evolve_T << <dimGrid3D, dimBlock3D >> > (Var.p, Var.p_new, Var.T, Var.T_new, InputP);
			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[1], kernel_start, kernel_end);

			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_start, 0);
			Update_p_and_T_regular << <dimGrid3D, dimBlock3D >> > (Var.p, Var.p_new, Var.T, Var.T_new, InputP);
			//CUDACHECK(cudaMemcpy(Var.p, Var.p_new, InputP.Nx * InputP.Ny * InputP.Nz * sizeof(double), cudaMemcpyDeviceToDevice));
			//CUDACHECK(cudaMemcpy(Var.T, Var.T_new, InputP.Nx * InputP.Ny * InputP.Nz * sizeof(double), cudaMemcpyDeviceToDevice));
			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[2], kernel_start, kernel_end);
		}
		else
		{
			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_start, 0);
			//For wake-up mode evolution of p and T
			Evolve_p_wakeup << <dimGrid3D, dimBlock3D >> > (Var.p, Var.T, Var.p_new, Var.metric_p, \
															Var.wake_up_p, InputP);
			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[0], kernel_start, kernel_end);

			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_start, 0);
			Evolve_T_wakeup << <dimGrid3D, dimBlock3D >> > (Var.p, Var.p_new, Var.T, Var.T_new, Var.metric_T, \
															Var.wake_up_p, Var.wake_up_T, InputP);
			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[1], kernel_start, kernel_end);
			
			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_start, 0);
			//Wakeup blocks for next step
			Wakeup_Next <<<dimGrid3D, dimBlock3D >>> (Var.p, Var.T, Var.p_new, Var.T_new, Var.metric_p, Var.metric_T, \
			 							Var.wake_up_p, Var.wake_up_T, Var.wake_up_p_next, Var.wake_up_T_next, InputP);
			//Wakeup neighboring blocks for next step
			Wakeup_Neighbor << <dimGrid3D, dimBlock3D >> > (Var.wake_up_p, Var.wake_up_T, Var.wake_up_p_next, Var.wake_up_T_next, InputP);
			/*-------------------For performance test-------------------*/
			cudaEventRecord(kernel_end, 0);
			cudaEventSynchronize(kernel_end);
			cudaEventElapsedTime(&kernel_time[2], kernel_start, kernel_end);

			/*-------------------For performance test-------------------*/
			//cudaEventRecord(kernel_start, 0);
			//CUDACHECK(cudaMemcpy(Var.p, Var.p_new, InputP.Nx * InputP.Ny * InputP.Nz * sizeof(double), cudaMemcpyDeviceToDevice));
			//CUDACHECK(cudaMemcpy(Var.T, Var.T_new, InputP.Nx * InputP.Ny * InputP.Nz * sizeof(double), cudaMemcpyDeviceToDevice));
			/*-------------------For performance test-------------------*/
			//cudaEventRecord(kernel_end, 0);
			//cudaEventSynchronize(kernel_end);
			//cudaEventElapsedTime(&kernel_time[3], kernel_start, kernel_end);
		}

		/*-------------------For performance test-------------------*/
		CUDACHECK(cudaDeviceSynchronize());
		for (int ii = 0; ii < 6; ii++)
			kernel_time_tot[ii] += kernel_time[ii];

		//Output p and T
		if (t % InputP.t_freq == 0)
		{	
			std::cout << "Time step: " << t << std::endl;
			CUDACHECK(cudaDeviceSynchronize());
			sim_time_end = clock();
			time_t sim_time = sim_time_end - sim_time_start;
			Calc_RealTime_Values("RTValues", Var.wake_up_p, Var.wake_up_T, Var.p, InputP, t, InputP.t_freq, dist, sim_time);
			sim_time_start = clock();
			/*
			if (!InputP.mode){
				Output_vtk("output_regular", Var.p, Var.T, InputP, t);
			} else{
				Output_vtk("output_DBA", Var.p, Var.T, InputP, t);
				Output_wakeup("Active", Var.wake_up_p,  Var.wake_up_T, InputP, t);
			}*/
		}
	}

	FreeMemory(Var);

	/*-------------------For performance test-------------------*/
	t_end_loop = clock();
	t_end_tot = clock();
	printf("The overall running time is: %f sec.\n", ((float)(t_end_tot - t_start_tot)) / CLOCKS_PER_SEC);
	printf("The loop running time is: %f sec. %3f percent of overall running time.\n", ((float)(t_end_loop - t_start_loop)) / CLOCKS_PER_SEC, (float)(t_end_loop - t_start_loop) / (double)(t_end_tot - t_start_tot) * 100.);

	if (!InputP.mode){
		printf("In detail: Regular Evolve p - %f sec, Regular Evolve T - %f sec, Regular Update - %f sec.\n",
			(float)(kernel_time_tot[0]) / 1000.0, (float)(kernel_time_tot[1]) / 1000.0, (float)(kernel_time_tot[2]) / 1000.0);
		printf("In detail: DevtoDev memory copy - %f sec.\n", (float)(kernel_time_tot[2]) / 1000.0);
	}
	else{
		printf("In detail: Wakeup Evolve p - %f sec, Wakeup Evolve T - %f sec, Wakeup Neighbor Blocks - %f sec.\n",
		(float)(kernel_time_tot[0]) / 1000.0, (float)(kernel_time_tot[1])/ 1000.0, (float)(kernel_time_tot[2]) / 1000.0);
	}
	
	cudaEventDestroy(kernel_start);
	cudaEventDestroy(kernel_end);

	CheckLastCudaError();
	

	return 0;
}
