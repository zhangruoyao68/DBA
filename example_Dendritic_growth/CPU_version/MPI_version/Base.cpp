#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <cstring>
#include <time.h>
#include <mpi.h>

#define Pi 3.14159

#define MPI_CALL(call)                                                                \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (0 != mpi_status) {                                                        \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
            int mpi_error_string_length = 0;                                          \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
            if (NULL != mpi_error_string)                                             \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %s "                                                    \
                        "(%d).\n",                                                    \
                        #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
            else                                                                      \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %d.\n",                                                 \
                        #call, __LINE__, __FILE__, mpi_status);                       \
        }                                                                             \
    }

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
	int mode, output_mode;

	void print_input()
	{
		printf("--------------Input parameters--------------\n");
		printf("Size of system: %d, %d, %d\n", Nx, Ny, Nz);
		printf("totalTime = %d,	printFreq = %d, dt = %lf\n", t_total, t_freq, dt);
		printf("dx = %lf, dy = %lf, dz = %lf\n", dx, dy, dz);
		printf("delta = %lf, tau = %lf,	epsilon = %lf, alpha = %lf, gamma = %lf\n", delta, tau, epsilon, alpha, gamma);
		printf("Delta = %lf\n", Delta);
		printf("output_mode = %d\n", output_mode);
	}
};

//Read input parameters
void ReadInput(std::string InputFileName, InputPara& InputP, int rank)
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
	infile >> InputP.output_mode;
	std::getline(infile, space);

	/* Close input file */
	infile.close();
	std::cout << "Done with input reading." << std::endl;

	if(rank == 0)
		InputP.print_input();
}

//Allocate memory
void Allocate(Variable& Var, InputPara InputP, int local_Nz)
{
	size_t size = (InputP.Nx + 2) * (InputP.Ny + 2) * (local_Nz + 2) * sizeof(float);

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
void Initialize(float* p, float* T, InputPara InputP, int local_Nz, int z_start)
{
	for (int z = 0; z < local_Nz + 2; z++)
		for (int y = 0; y < InputP.Ny + 2; y++)
			for (int x = 0; x < InputP.Nx + 2; x++)
			{
				int index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
				float dist = std::sqrt((float)((x - 1 - (InputP.Nx + 2) / 2) * (x - 1 - (InputP.Nx + 2) / 2) + (y - 1 - (InputP.Ny + 2) / 2) * (y - 1 - (InputP.Ny + 2) / 2) + (z - 1 + z_start - (InputP.Nz + 2) / 2) * (z - 1 + z_start - (InputP.Nz + 2) / 2)));
				if (dist <= 5)
					p[index] = (1. - 0.2 * dist);
				else
					p[index] = 0;
				T[index] = -InputP.Delta;
			}
}

//Communicate boundary between processes
void Communicate_boundary(float* p, float* T, InputPara InputP, int rank, int size, int local_Nz)
{
	int rankm = ((rank == 0) ? (size - 1) : (rank - 1));
	int rankp = ((rank == size - 1) ? (0) : (rank + 1));
	
	int layer_size = (InputP.Nx + 2) * (InputP.Ny + 2);
	
	MPI_CALL(MPI_Sendrecv(p + 1 * layer_size, layer_size, MPI_FLOAT, rankm, 0, p + (local_Nz + 1) * layer_size, layer_size, MPI_FLOAT, rankp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
	MPI_CALL(MPI_Sendrecv(p + (local_Nz) * layer_size, layer_size, MPI_FLOAT, rankp, 0, p, layer_size, MPI_FLOAT, rankm, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
	
	MPI_CALL(MPI_Sendrecv(T + 1 * layer_size, layer_size, MPI_FLOAT, rankm, 0, T + (local_Nz + 1) * layer_size, layer_size, MPI_FLOAT, rankp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
	MPI_CALL(MPI_Sendrecv(T + (local_Nz) * layer_size, layer_size, MPI_FLOAT, rankp, 0, T, layer_size, MPI_FLOAT, rankm, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));	
}

//Calculate derivatives and Laplacian of p via Finite Difference
void Finite_Diff_p(float* p, float* dpx, float* dpy, float* dpz, float* lapP, InputPara InputP, int local_Nz)
{
	int xm, xp, ym, yp, zm, zp;
	int index_xm, index_xp, index_ym, index_yp, index_zm, index_zp;

	float factor1 = 1. / (2 * InputP.dx);
	float factor2 = 1. / InputP.dx / InputP.dx;

	for (int z = 1; z < local_Nz + 1; z++)
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
void Finite_Diff_T(float* T, float* lapT, InputPara InputP, int local_Nz)
{
	int xm, xp, ym, yp, zm, zp;
	int index_xm, index_xp, index_ym, index_yp, index_zm, index_zp;

	float factor2 = 1. / InputP.dx / InputP.dx;

	for (int z = 1; z < local_Nz + 1; z++)
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
void Evolve_p_and_T(float* p, float* p_new, float* dpx, float* dpy, float* dpz, float* lapP, float* T, float* T_new, float* lapT, InputPara InputP, int local_Nz)
{
	float v2, sigma, m, p_change, T_change;
	float factor1 = -InputP.alpha / Pi;
	float factor2 = 1. / InputP.tau;

	for (int z = 1; z < local_Nz + 1; z++)
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

//Set boundary conditions
void Set_boundary(float* p, float* T, InputPara InputP, int local_Nz)
{
	// Set periodic BC for x and y directions, z no need due to MPI in z
	int index, index_t;
	for(int z = 0; z < local_Nz + 2; z++)
		for (int y = 0; y < InputP.Ny + 2; y++)
		{
			index = 0 + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
			index_t = InputP.Nx + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
			p[index] = p[index_t];
			T[index] = T[index_t];
			
			index = (InputP.Nx + 1) + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
			index_t = 1 + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
			p[index] = p[index_t];
			T[index] = T[index_t];
		}
		
	for(int z = 0; z < local_Nz + 2; z++)
		for (int x = 0; x < InputP.Nx + 2; x++)
		{
			index = x + 0 * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
			index_t = x + InputP.Ny * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
			p[index] = p[index_t];
			T[index] = T[index_t];
			
			index = x + (InputP.Ny + 1) * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
			index_t = x + 1 * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
			p[index] = p[index_t];
			T[index] = T[index_t];
		}
}

//Output data in .vtk form
void Output_vtk(std::string prefix, float* data, InputPara InputP, int time, int rank, int size, int local_Nz, int z_start)
{
	
	/* Gathering data */
	if(rank == 0)
	{
		float* buffer = (float*)aligned_alloc(4096, (InputP.Nx + 2)*(InputP.Ny + 2)*InputP.Nz*sizeof(float));
		std::memcpy(buffer, data + (InputP.Nx + 2)*(InputP.Ny + 2), local_Nz*(InputP.Nx + 2)*(InputP.Ny + 2)*sizeof(float));
		for (int i=1; i<size; i++)
        	{
            		int count, offset;
            		MPI_CALL(MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(&count, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(buffer + offset, count, MPI_FLOAT, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        	}
        	
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
		for (int z = 0; z < InputP.Nz; z++)
			for (int y = 1; y < InputP.Ny+1; y++)
				for (int x = 1; x < InputP.Nx+1; x++)
				{
					int index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
					outf << buffer[index] << " ";
				}
		outf << std::endl;
		outf.close();
		std::cout << "--------------Output Done--------------" << std::endl;	
		free(buffer);
		
	}
	else
	{
		int offset = z_start*(InputP.Nx + 2)*(InputP.Ny + 2);
        	int count = local_Nz*(InputP.Nx + 2)*(InputP.Ny + 2);
        	MPI_CALL(MPI_Send(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(data + (InputP.Nx + 2)*(InputP.Ny + 2), count, MPI_FLOAT, 0, 2, MPI_COMM_WORLD));
	}
}

//Calculate real-time values, e.g. steps per second
void Calc_RealTime_Values(std::string prefix, InputPara InputP, int time, int RT_freq, time_t sim_time, int rank)
{
	if(rank == 0)
	{
		//std::cout << "--------------Writting Output of Velocity Now--------------" << std::endl;
		std::string OutputFileName;
		OutputFileName = prefix + ".out";

		std::ifstream inf(OutputFileName);
		if(inf.good())
		{
			std::ofstream outf;
			outf.open(OutputFileName, std::fstream::app);
			if (!outf.is_open())
			{
				std::cout << "!!!!!Can not open" << OutputFileName << "!! Exit!!!!!!!!" << std::endl;
				exit(1);
			}
	
			double steps_per_second = RT_freq / ((double)sim_time / CLOCKS_PER_SEC);
	
			outf << time << " " << steps_per_second << std::endl;
			outf.close();
			//std::cout << "--------------Output Done--------------" << std::endl;	
		}
		else
		{
			std::ofstream outf;
			outf.open(OutputFileName, std::fstream::app);
			if (!outf.is_open())
			{
				std::cout << "!!!!!Can not open" << OutputFileName << "!! Exit!!!!!!!!" << std::endl;
				exit(1);
			}
	
			double steps_per_second = RT_freq / ((double)sim_time / CLOCKS_PER_SEC);
			outf << "Time Steps_per_second" << std::endl;
			outf << time << " " << steps_per_second << std::endl;
			outf.close();
			//std::cout << "--------------Output Done--------------" << std::endl;	
		}
	}
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

int main(int argc, char *argv[])
{
	int rank, size;
    	MPI_CALL(MPI_Init(&argc, &argv));
    	MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));  // Get the rank of the process
    	MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));  // Get the total number of processes
    	
    	InputPara InputP;
	ReadInput("Input.txt", InputP, rank);
    	
    	// Calculate the start and end indices for each process
    	int chunk_size = InputP.Nz / size;
    	int remainder = InputP.Nz % size;
    	int local_Nz, z_start;
    	if(rank < remainder)
    	{
    		local_Nz = chunk_size + 1;
    		z_start = local_Nz * rank;
    	}
    	else
    	{
    		local_Nz = chunk_size;
    		z_start = local_Nz * rank + remainder;
    	}
    	
    	std::cout << "I'm proc " << rank << "/" << size << ", my z_start = " << z_start << ", and my local_Nz = " << local_Nz << std::endl;

	time_t start_total_time = time(NULL);
	time_t start_output_time, end_output_time;
	float output_time_total = 0;

	Variable Var;
	Allocate(Var, InputP, local_Nz);

	Initialize(Var.p, Var.T, InputP, local_Nz, z_start);
	Communicate_boundary(Var.p, Var.T, InputP, rank, size, local_Nz);
	Set_boundary(Var.p, Var.T, InputP, local_Nz);
	
	MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
	
	start_output_time = clock();
	if(InputP.output_mode > 1)
	{
		Output_vtk("p", Var.p, InputP, 0, rank, size, local_Nz, z_start);
		MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
		Output_vtk("T", Var.T, InputP, 0, rank, size, local_Nz, z_start);
		MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
	}
	end_output_time = clock();
	output_time_total += (float)(end_output_time - start_output_time);
	
	time_t sim_time_start, sim_time_end;
	sim_time_start = clock();

	for (int t = 1; t < InputP.t_total + 1; t++)
	{
		Finite_Diff_p(Var.p, Var.dp[0], Var.dp[1], Var.dp[2], Var.lapP, InputP, local_Nz);

		Finite_Diff_T(Var.T, Var.lapT, InputP, local_Nz);

		Evolve_p_and_T(Var.p, Var.p_new, Var.dp[0], Var.dp[1], Var.dp[2], Var.lapP, Var.T, Var.T_new, Var.lapT, InputP, local_Nz);

		std::memcpy(Var.p + (InputP.Nx + 2) * (InputP.Ny + 2), Var.p_new + (InputP.Nx + 2) * (InputP.Ny + 2), (InputP.Nx + 2) * (InputP.Ny + 2) * local_Nz * sizeof(float));
		std::memcpy(Var.T + (InputP.Nx + 2) * (InputP.Ny + 2), Var.T_new + (InputP.Nx + 2) * (InputP.Ny + 2), (InputP.Nx + 2) * (InputP.Ny + 2) * local_Nz * sizeof(float));
		
		Communicate_boundary(Var.p, Var.T, InputP, rank, size, local_Nz);
		Set_boundary(Var.p, Var.T, InputP, local_Nz);
		
		MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));

		start_output_time = clock();
		if(InputP.output_mode)
		{
			if (t % InputP.t_freq == 0)
			{
				if(InputP.output_mode > 1)
				{
					Output_vtk("p", Var.p, InputP, t, rank, size, local_Nz, z_start);
					Output_vtk("T", Var.T, InputP, t, rank, size, local_Nz, z_start);
				}
				sim_time_end = clock();
				time_t sim_time = sim_time_end - sim_time_start;
				Calc_RealTime_Values("RTValues", InputP, t, InputP.t_freq, sim_time, rank);
				sim_time_start = clock();
			}
		}
		end_output_time = clock();
		output_time_total += (float)(end_output_time - start_output_time);
	}

	FreeMemory(Var);
	
	MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
	
	if(rank == 0)
	{
		time_t end_total_time = time(NULL);
		printf("\nTotal time in seconds: %f\n", difftime(end_total_time, start_total_time));
		printf("\nTotal time (subtract output time) in seconds: %f\n", difftime(end_total_time, start_total_time) - output_time_total/CLOCKS_PER_SEC);
	}
	
	MPI_CALL(MPI_Finalize());

	return 0;
}
