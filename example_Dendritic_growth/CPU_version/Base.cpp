#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstring>
#include <time.h>

#define Pi 3.14159

class Variable
{
public:
	float* p, * T, * p_new, * T_new;
	float* dp[3];
	float* lapP, * lapT;
};

class InputPara
{
public:
	int Nx, Ny, Nz;
	int t_total, t_freq;
	float dx, dy, dz, dt;
	float delta, tau, epsilon, alpha, gamma;
	float Delta;
	int mode;

	void print_input()
	{
		printf("--------------Input parameters--------------\n");
		printf("Size of system: %d, %d, %d\n", Nx, Ny, Nz);
		printf("totalTime = %d,	printFreq = %d, dt = %lf\n", t_total, t_freq, dt);
		printf("dx = %lf, dy = %lf, dz = %lf\n", dx, dy, dz);
		printf("delta = %lf, tau = %lf,	epsilon = %lf, alpha = %lf, gamma = %lf\n", delta, tau, epsilon, alpha, gamma);
		printf("Delta = %lf\n", Delta);
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

	/* Close input file */
	infile.close();
	std::cout << "Done with input reading." << std::endl;

	InputP.print_input();
}

//Allocate memory
void Allocate(Variable& Var, InputPara InputP)
{
	size_t size = (InputP.Nx + 2) * (InputP.Ny + 2) * (InputP.Nz + 2) * sizeof(float);

	Var.p = (float*)aligned_alloc(4096,size);
	Var.T = (float*)aligned_alloc(4096,size);
	Var.p_new = (float*)aligned_alloc(4096,size);
	Var.T_new = (float*)aligned_alloc(4096,size);;
	for (int i = 0; i < 3; i++)
		Var.dp[i] = (float*)aligned_alloc(4096,size);;
	Var.lapP = (float*)aligned_alloc(4096,size);;
	Var.lapT = (float*)aligned_alloc(4096,size);;
}

//Initialize system
void Initialize(float* p, float* T, InputPara InputP)
{
	for (int z = 0; z < InputP.Nz + 2; z++)
		for (int y = 0; y < InputP.Ny + 2; y++)
			for (int x = 0; x < InputP.Nx + 2; x++)
			{
				int index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
				float dist = std::sqrt((float)((x - (InputP.Nx + 2) / 2) * (x - (InputP.Nx + 2) / 2) + (y - (InputP.Ny + 2) / 2) * (y - (InputP.Ny + 2) / 2) + (z - (InputP.Nz + 2) / 2) * (z - (InputP.Nz + 2) / 2)));
				if (dist <= 5)
					p[index] = (1. - 0.2 * dist);
				else
					p[index] = 0;
				T[index] = -InputP.Delta;
			}
}

//Calculate derivatives and Laplacian of p via Finite Difference
void Finite_Diff_p(float* p, float* dpx, float* dpy, float* dpz, float* lapP, InputPara InputP)
{
	int xm, xp, ym, yp, zm, zp;
	int index_xm, index_xp, index_ym, index_yp, index_zm, index_zp;

	float factor1 = 1. / (2 * InputP.dx);
	float factor2 = 1. / InputP.dx / InputP.dx;

	for (int z = 1; z < InputP.Nz + 1; z++)
		for (int y = 1; y < InputP.Ny + 1; y++)
			for (int x = 1; x < InputP.Nx + 1; x++)
			{
				xm = x - 1;
				xp = x + 1;
				ym = y - 1;
				yp = y + 1;
				zm = z - 1;
				zp = z + 1;

				index_xm = xm + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
				index_xp = xp + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
				index_ym = x + ym * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
				index_yp = x + yp * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
				index_zm = x + y * (InputP.Nx + 2) + zm * (InputP.Nx + 2) * (InputP.Ny + 2);
				index_zp = x + y * (InputP.Nx + 2) + zp * (InputP.Nx + 2) * (InputP.Ny + 2);

				int index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);

				dpx[index] = (p[index_xp] - p[index_xm]) * factor1;
				dpy[index] = (p[index_yp] - p[index_ym]) * factor1;
				dpz[index] = (p[index_zp] - p[index_zm]) * factor1;

				lapP[index] = (p[index_xp] + p[index_xm] + p[index_yp] + p[index_ym] + p[index_zp] + p[index_zm] - 6 * p[index]) * factor2;
			}
}

//Calculate Laplacian of T via Finite Difference
void Finite_Diff_T(float* T, float* lapT, InputPara InputP)
{
	int xm, xp, ym, yp, zm, zp;
	int index_xm, index_xp, index_ym, index_yp, index_zm, index_zp;

	float factor2 = 1. / InputP.dx / InputP.dx;

	for (int z = 1; z < InputP.Nz + 1; z++)
		for (int y = 1; y < InputP.Ny + 1; y++)
			for (int x = 1; x < InputP.Nx + 1; x++)
			{
				xm = x - 1;
				xp = x + 1;
				ym = y - 1;
				yp = y + 1;
				zm = z - 1;
				zp = z + 1;

				index_xm = xm + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
				index_xp = xp + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
				index_ym = x + ym * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
				index_yp = x + yp * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
				index_zm = x + y * (InputP.Nx + 2) + zm * (InputP.Nx + 2) * (InputP.Ny + 2);
				index_zp = x + y * (InputP.Nx + 2) + zp * (InputP.Nx + 2) * (InputP.Ny + 2);

				int index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);

				lapT[index] = (T[index_xp] + T[index_xm] + T[index_yp] + T[index_ym] + T[index_zp] + T[index_zm] - 6 * T[index]) * factor2;
			}
}

//Calculate p_new and T_new
void Evolve_p_and_T(float* p, float* p_new, float* dpx, float* dpy, float* dpz, float* lapP, float* T, float* T_new, float* lapT, InputPara InputP)
{
	float v2, sigma, m, p_change, T_change;
	float factor1 = -InputP.alpha / Pi;
	float factor2 = 1. / InputP.tau;

	for (int z = 1; z < InputP.Nz + 1; z++)
		for (int y = 1; y < InputP.Ny + 1; y++)
			for (int x = 1; x < InputP.Nx + 1; x++)
			{
				int index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
				//Pre-calculations
				v2 = std::pow(dpx[index], 2) + std::pow(dpy[index], 2) + std::pow(dpz[index], 2);
				if (v2 <= 1E-6)
					sigma = 1;
				else
					sigma = 1 - InputP.delta * (1 - (std::pow(dpx[index], 4) + std::pow(dpy[index], 4) + std::pow(dpz[index], 4))
						/ std::pow(v2, 2));
				m = factor1 * std::atan(InputP.gamma * sigma * T[index]);

				//Calculate dp/dt and dT/dt
				p_change = factor2 * (InputP.epsilon * InputP.epsilon * lapP[index]
					+ p[index] * (1 - p[index]) * (p[index] - 0.5 + m));
				T_change = lapT[index] + p_change;

				//Update p and T
				p_new[index] = p[index] + InputP.dt * p_change;
				T_new[index] = T[index] + InputP.dt * T_change;	
			}
}

//Set the boundary conditions
void Set_boundary(float* p_new, float* T_new, InputPara InputP)
{
	for (int z = 0; z < InputP.Nz + 2; z++)
		for (int y = 0; y < InputP.Ny + 2; y++)
			for (int x = 0; x < InputP.Nx + 2; x++)
			{
				int index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
				if(x==0||y==0||z==0||x==InputP.Nx+1||y==InputP.Ny+1||z==InputP.Nz+1)
				{
					x == 0? x = InputP.Nx : x;
					x == InputP.Nx + 1? x = 1 : x;
					y == 0? y = InputP.Ny : y;
					y == InputP.Ny + 1? y = 1 : y;
					z == 0? z = InputP.Nz : z;
					z == InputP.Nz + 1? z = 1 : z;
					int index_new = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
					p_new[index] = p_new[index_new];
					T_new[index] = T_new[index_new];
				}
			}
}

//Output data in .vtk form
void Output_vtk(std::string prefix, float* data, InputPara InputP, int time)
{
	std::cout << "--------------Writting Output Now--------------" << std::endl;
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
	outf << prefix << std::endl << "ASCII" << std::endl << "DATASET STRUCTURED_POINTS" << std::endl;
	outf << "DIMENSIONS " << InputP.Nx << " " << InputP.Ny << " " << InputP.Nz << std::endl;
	outf << "ASPECT_RATIO 1 1 1" << std::endl << "ORIGIN 0 0 0" << std::endl << "POINT_DATA " << InputP.Nx * InputP.Ny * InputP.Nz << std::endl;
	outf << "SCALARS " << prefix << " double" << std::endl << "LOOKUP_TABLE default" << std::endl;
	for (int z = 1; z < InputP.Nz + 1; z++)
		for (int y = 1; y < InputP.Ny + 1; y++)
			for (int x = 1; x < InputP.Nx + 1; x++)
			{
				int index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
				outf << data[index] << " ";
			}
	outf << std::endl;
	outf.close();
	std::cout << "--------------Output Done--------------" << std::endl;
}

//Free memory
void FreeMemory(Variable& Var)
{
	free(Var.p);
	free(Var.T);
	free(Var.p_new);
	free(Var.T_new);
	for (int i = 0; i < 3; i++)
		free(Var.dp[i]);
	free(Var.lapP);
	free(Var.lapT);
}

int main()
{
	time_t start_total_time = time(NULL);

	InputPara InputP;
	ReadInput("Input.txt", InputP);

	Variable Var;
	Allocate(Var, InputP);

	Initialize(Var.p, Var.T, InputP);

	for (int t = 1; t < InputP.t_total + 1; t++)
	{
		Finite_Diff_p(Var.p, Var.dp[0], Var.dp[1], Var.dp[2], Var.lapP, InputP);

		Finite_Diff_T(Var.T, Var.lapT, InputP);

		Evolve_p_and_T(Var.p, Var.p_new, Var.dp[0], Var.dp[1], Var.dp[2], Var.lapP, Var.T, Var.T_new, Var.lapT, InputP);

		std::memcpy(Var.p, Var.p_new, (InputP.Nx + 2) * (InputP.Ny + 2) * (InputP.Nz + 2) * sizeof(float));
		std::memcpy(Var.T, Var.T_new, (InputP.Nx + 2) * (InputP.Ny + 2) * (InputP.Nz + 2) * sizeof(float));

		if (t % InputP.t_freq == 0)
		{
			Output_vtk("p", Var.p, InputP, t);
			Output_vtk("T", Var.T, InputP, t);
		}
	}

	FreeMemory(Var);
	
	time_t end_total_time = time(NULL);
	printf("\nTotal time in seconds: %f\n", difftime(end_total_time, start_total_time));

	return 0;
}
