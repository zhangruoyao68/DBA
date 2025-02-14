/*
 * Compressible Euler equation: KH instability
 * 2D Version, Parallel with CUDA
 * 
 */

#include <iostream>
#include <random>
#include <fstream>

#include <math.h>
#include <ctime>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <cstring>

#include <mpi.h>

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
		double* rho, *p, *vx, *vy;
        double* Mass, *Momx, *Momy, *Energy;
        double* rho_dx, *rho_dy, *p_dx, *p_dy, *vx_dx, *vx_dy, *vy_dx, *vy_dy;
        double* rho_prime, *vx_prime, *vy_prime, *p_prime;
		double* flux_Mass_x, *flux_Momx_x, *flux_Momy_x, *flux_Energy_x;
        double* flux_Mass_y, *flux_Momx_y, *flux_Momy_y, *flux_Energy_y;
};

class InputPara
{
	public:
        int Nx, Ny, Nz;
		int t_total, t_freq;
		double dx, dy, dz, dt;
		int output_mode;

		void print_input()
		{
			printf("--------------Input parameters--------------\n");
			printf("Size of system: %d, %d, %d\n", Nx, Ny, Nz);
			printf("totalTime = %d,	printFreq = %d, dt = %f\n", t_total, t_freq, dt);
			printf("dx = %lf, dy = %lf, dz =%lf\n", dx, dy, dz);
			printf("output_mode = %d\n", output_mode);
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
	infile >> InputP.output_mode;
	std::getline(infile, space);

	/* Close input file */
	infile.close();
	std::cout << "Done with input reading." << std::endl;

	InputP.print_input();
}

//Allocate memory
void Allocate(Variable & Var, InputPara InputP, int local_Ny)
{
	size_t size = (InputP.Nx + 2) * (local_Ny + 2) * InputP.Nz * sizeof(double);

    	// Primitive variables
	Var.rho = (double*)aligned_alloc(4096, size);
	Var.p = (double*)aligned_alloc(4096, size);
	Var.vx = (double*)aligned_alloc(4096, size);
	Var.vy = (double*)aligned_alloc(4096, size);

    	// Conserved variables
    	Var.Mass = (double*)aligned_alloc(4096, size);
	Var.Momx = (double*)aligned_alloc(4096, size);
	Var.Momy = (double*)aligned_alloc(4096, size);
	Var.Energy = (double*)aligned_alloc(4096, size);

	// Gradients
	Var.rho_dx = (double*)aligned_alloc(4096, size);
	Var.rho_dy = (double*)aligned_alloc(4096, size);
	Var.p_dx = (double*)aligned_alloc(4096, size);
	Var.p_dy = (double*)aligned_alloc(4096, size);
	Var.vx_dx = (double*)aligned_alloc(4096, size);
	Var.vx_dy = (double*)aligned_alloc(4096, size);
	Var.vy_dx = (double*)aligned_alloc(4096, size);
	Var.vy_dy = (double*)aligned_alloc(4096, size);

    	// Extrapolated variables
    	Var.rho_prime = (double*)aligned_alloc(4096, size);
	Var.vx_prime = (double*)aligned_alloc(4096, size);
	Var.vy_prime = (double*)aligned_alloc(4096, size);
	Var.p_prime = (double*)aligned_alloc(4096, size);

    	// Fluxes
    	Var.flux_Mass_x = (double*)aligned_alloc(4096, size);
	Var.flux_Momx_x = (double*)aligned_alloc(4096, size);
	Var.flux_Momy_x = (double*)aligned_alloc(4096, size);
	Var.flux_Energy_x = (double*)aligned_alloc(4096, size);
	Var.flux_Mass_y = (double*)aligned_alloc(4096, size);
	Var.flux_Momx_y = (double*)aligned_alloc(4096, size);
	Var.flux_Momy_y = (double*)aligned_alloc(4096, size);
	Var.flux_Energy_y = (double*)aligned_alloc(4096, size);
}

//Free memory
void FreeMemory(Variable & Var)
{
	free(Var.rho);
    	free(Var.p);
    	free(Var.vx);
    	free(Var.vy);
    	free(Var.Mass);
    	free(Var.Energy);
    	free(Var.Momx);
    	free(Var.Momy);
    	free(Var.rho_dx);
    	free(Var.rho_dy);
    	free(Var.p_dx);
    	free(Var.p_dy);
    	free(Var.vx_dx);
    	free(Var.vx_dy);
    	free(Var.vy_dx);
    	free(Var.vy_dy);
    	free(Var.rho_prime);
    	free(Var.vx_prime);
    	free(Var.vy_prime);
    	free(Var.p_prime);
    	free(Var.flux_Mass_x);
    	free(Var.flux_Energy_x);
    	free(Var.flux_Momx_x);
    	free(Var.flux_Momy_x);
    	free(Var.flux_Mass_y);
    	free(Var.flux_Energy_y);
    	free(Var.flux_Momx_y);
    	free(Var.flux_Momy_y);
}

//Communicate boundary between processes
void Communicate_MPI_boundary(double* c, InputPara InputP, int rank, int size, int local_Ny)
{
	int rankm = ((rank == 0) ? (size - 1) : (rank - 1));
	int rankp = ((rank == size - 1) ? (0) : (rank + 1));
	
	int layer_size = (InputP.Nx + 2);
	
	MPI_CALL(MPI_Sendrecv(c + 1 * layer_size, layer_size, MPI_DOUBLE, rankm, 0, c + (local_Ny + 1) * layer_size, layer_size, MPI_DOUBLE, rankp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
	MPI_CALL(MPI_Sendrecv(c + (local_Ny) * layer_size, layer_size, MPI_DOUBLE, rankp, 1, c, layer_size, MPI_DOUBLE, rankm, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));	
}

//Set boundary conditions
void Set_boundary(double* c, InputPara InputP, int local_Ny)
{
	// Set periodic BC for x direction, y no need due to MPI in y
	int index, index_t;
	for(int y = 0; y < local_Ny + 2; y++)
		for (int x = 0; x < InputP.Nx + 2; x++)
		{
			index = 0 + y * (InputP.Nx + 2);
			index_t = InputP.Nx + y * (InputP.Nx + 2);
			c[index] = c[index_t];
			
			index = (InputP.Nx + 1) + y * (InputP.Nx + 2);
			index_t = 1 + y * (InputP.Nx + 2);
			c[index] = c[index_t];
		}
}

double Grad_sq(double* c, int x, int y, int z, InputPara InputP)
{
	int xm, xp, ym, yp, zm, zp;
	xm = x - 1;
	xp = x + 1;
	ym = y - 1;
	yp = y + 1;
	zm = 0;
	zp = 0;
    
    // Coalesced memory access
	//int index_xyz = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;	
	int index_xm = xm + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xp = xp + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_ym = x + ym * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_yp = x + yp * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_zm = x + y * (InputP.Nx + 2) + zm * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_zp = x + y * (InputP.Nx + 2) + zp * (InputP.Nx + 2) * (InputP.Ny + 2);

	double dpx = (c[index_xp] - c[index_xm]) / 2.0 / InputP.dx;
	double dpy = (c[index_yp] - c[index_ym]) / 2.0 / InputP.dy;
	double dpz = (c[index_zp] - c[index_zm]) / 2.0 / InputP.dz;
	
	return dpx*dpx + dpy*dpy + dpz*dpz;
}

double Grad_x(double*c, int x, int y, int z, InputPara InputP)
{
    	int xm, xp;
    	xm = x - 1;
    	xp = x + 1;

    	int index_xm = xm + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xp = xp + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);

    	double dpx = (c[index_xp] - c[index_xm]) / 2.0 / InputP.dx;

    	return dpx;
}

double Grad_y(double*c, int x, int y, int z, InputPara InputP)
{
    	int ym, yp;
    	ym = y - 1;
    	yp = y + 1;

    	int index_ym = x + ym * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_yp = x + yp * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);

    	double dpy = (c[index_yp] - c[index_ym]) / 2.0 / InputP.dy;

    	return dpy;
}

double Laplacian(double* c, int x, int y, int z, InputPara InputP)
{
	int xm, xp, ym, yp, zm, zp;

	//Finite Difference
	xm = x - 1;
	xp = x + 1;
	ym = y - 1;
	yp = y + 1;
	zm = 0;
	zp = 0;

	int index_xyz = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);	// Coalesced memory access
	int index_xmyz = xm + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xpyz = xp + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xymz = x + ym * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xypz = x + yp * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xyzm = x + y * (InputP.Nx + 2) + zm * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xyzp = x + y * (InputP.Nx + 2) + zp * (InputP.Nx + 2) * (InputP.Ny + 2);

	int index_xpypz = xp + yp * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xpynz = xp + ym * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xnypz = xm + yp * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xnynz = xm + ym * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xypzp = x + yp * (InputP.Nx + 2) + zp * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xynzp = x + ym * (InputP.Nx + 2) + zp * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xypzn = x + yp * (InputP.Nx + 2) + zm * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xynzn = x + ym * (InputP.Nx + 2) + zm * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xpyzp = xp + y * (InputP.Nx + 2) + zp * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xnyzp = xm + y * (InputP.Nx + 2) + zp * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xpyzn = xp + y * (InputP.Nx + 2) + zm * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xnyzn = xm + y * (InputP.Nx + 2) + zm * (InputP.Nx + 2) * (InputP.Ny + 2);

	int index_xpypzp = xp + yp * (InputP.Nx + 2) + zp * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xnypzp = xm + yp * (InputP.Nx + 2) + zp * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xnynzp = xm + ym * (InputP.Nx + 2) + zp * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xpynzp = xp + ym * (InputP.Nx + 2) + zp * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xpypzn = xp + yp * (InputP.Nx + 2) + zm * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xnypzn = xm + yp * (InputP.Nx + 2) + zm * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xnynzn = xm + ym * (InputP.Nx + 2) + zm * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_xpynzn = xp + ym * (InputP.Nx + 2) + zm * (InputP.Nx + 2) * (InputP.Ny + 2);

	// 27-point Laplacian
	double result = 1.0 / (InputP.dx * InputP.dx) * (\
		- 64.0 / 15.0 * c[index_xyz] \
		+ 7.0 / 15.0 * (c[index_xpyz] + c[index_xmyz] + c[index_xypz] \
			+ c[index_xymz] + c[index_xyzp] + c[index_xyzm]) \
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

void getConserved(double* Mass, double* Energy, double* Momx, double* Momy, \
                            double* rho, double* p, double* vx, double* vy, \
                            InputPara InputP, int local_Ny)
{   
    	int index;
    	double gamma = 5.0/3.0;
   	double vol = InputP.dx * InputP.dy;
    
    	for (int z = 0; z < InputP.Nz; z++){
		for (int y = 1; y < local_Ny + 1; y++){
			for (int x = 1; x < InputP.Nx + 1; x++){

				index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
                    
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

void getPrimitive(double* rho, double* p, double* vx, double* vy, \
                            double* Mass, double* Energy, double* Momx, double* Momy, \
                            InputPara InputP, int local_Ny)
{   
    	int index;
    	double gamma = 5.0/3.0;
    	double vol = InputP.dx * InputP.dy;
    
     	for (int z = 0; z < InputP.Nz; z++){
		for (int y = 1; y < local_Ny + 1; y++){
			for (int x = 1; x < InputP.Nx + 1; x++){

				index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
				
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

void extrap_in_time(double* rho, double* p, double* vx, double* vy, \
                            double* rho_dx, double* rho_dy, double* p_dx, double* p_dy, \
                            double* vx_dx, double* vx_dy, double* vy_dx, double* vy_dy, \
                            double* rho_prime, double* p_prime, double* vx_prime, double* vy_prime, \
                            InputPara InputP, int local_Ny)
{
    	int index;
    	double gamma = 5.0/3.0;
    
    	for (int z = 0; z < InputP.Nz; z++){
		for (int y = 1; y < local_Ny + 1; y++){
			for (int x = 1; x < InputP.Nx + 1; x++){

				index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
                    
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

void extrapolateInSpaceToFace(double face[4], double* f, double* f_dx, double* f_dy,
                                        int x, int y, int z, InputPara InputP)
{   
    // index for periodic boundary
	//int xm = (x - 1 >= 0) ? x - 1 : InputP.Nx - 1;
	int xp = x + 1;
	//int ym = (y - 1 >= 0) ? y - 1 : InputP.Ny - 1;
	int yp = y + 1;
	//int zm = (z - 1 >= 0) ? z - 1 : InputP.Nz - 1;
	//int zp = (z + 1 < InputP.Nz) ? z + 1 : 0;
    
    // xm and ym are not used
	int index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);	
	int index_xp = xp + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);	
	int index_yp = x + yp * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);	
	

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

void getFlux(double* flux_Mass, double* flux_Energy, double* flux_Momx, double* flux_Momy, \
                        double rho_L, double rho_R, double p_L, double p_R, \
                        double vx_L, double vx_R, double vy_L, double vy_R, \
                        int x, int y, int z, InputPara InputP)
{
    	int index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
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

void calc_flux(double* rho, double* p, double* vx, double* vy, \
                        double* rho_dx, double* rho_dy, double* p_dx, double* p_dy, \
                        double* vx_dx, double* vx_dy, double* vy_dx, double* vy_dy, \
                        double* rho_prime, double* p_prime, double* vx_prime, double* vy_prime, \
                        double* flux_Mass_x, double* flux_Energy_x, double* flux_Momx_x, double* flux_Momy_x, \
                        double* flux_Mass_y, double* flux_Energy_y, double* flux_Momx_y, double* flux_Momy_y, \
                        InputPara InputP, int local_Ny)
{
    	for (int z = 0; z < InputP.Nz; z++){
		for (int y = 1; y < local_Ny + 1; y++){
			for (int x = 1; x < InputP.Nx + 1; x++){

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

void applyFluxes(double* f, double* flux_x, double* flux_y, \
                                int x, int y, int z, InputPara InputP)
{
    	// index for periodic boundary
	int xm = x - 1;
	//int xp = (x + 1 < InputP.Nx) ? x + 1 : 0;
	int ym = y - 1;
	//int yp = (y + 1 < InputP.Ny) ? y + 1 : 0;
	//int zm = (z - 1 >= 0) ? z - 1 : InputP.Nz - 1;
	//int zp = (z + 1 < InputP.Nz) ? z + 1 : 0;
    
    	// xp and yp are not used
	int index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);	
	int index_xm = xm + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
	int index_ym = x + ym * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);

    	f[index] += InputP.dt * InputP.dx * (-1.0*flux_x[index] + flux_x[index_xm] \
                                        -1.0*flux_y[index] + flux_y[index_ym]);
}

void apply_flux(double* Mass, double* Energy, double* Momx, double* Momy, \
                        double* flux_Mass_x, double* flux_Energy_x, double* flux_Momx_x, double* flux_Momy_x, \
                        double* flux_Mass_y, double* flux_Energy_y, double* flux_Momx_y, double* flux_Momy_y, \
                        InputPara InputP, int local_Ny)
{
    	for (int z = 0; z < InputP.Nz; z++){
		for (int y = 1; y < local_Ny + 1; y++){
			for (int x = 1; x < InputP.Nx + 1; x++){
   
                    		applyFluxes(Mass, flux_Mass_x, flux_Mass_y, x, y, z, InputP);
                    		applyFluxes(Momx, flux_Momx_x, flux_Momx_y, x, y, z, InputP);
                    		applyFluxes(Momy, flux_Momy_x, flux_Momy_y, x, y, z, InputP);
                    		applyFluxes(Energy, flux_Energy_x, flux_Energy_y, x, y, z, InputP);
			}
		}
	}
}

//Initialize system
void Initialize(double* rho, double* p, double* vx, double* vy, InputPara InputP, int local_Ny, int y_start)
{
	int index;
    	for (int z = 0; z < InputP.Nz; z++){
		for (int y = 1; y < local_Ny + 1; y++){
			for (int x = 1; x < InputP.Nx + 1; x++){

				index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);

				// constant pressure everywhere
                    		p[index] = 2.5;

                    		// Smooth density interface
                    		// Reference Paper: A well-posed Kelvin-Helmholtz instability test and comparison
                    		double rho1 = 1.0;
                    		double rho2 = 2.0;
                    		double rho_m = 0.5*(rho1-rho2);
                    		double vx1 = 0.5;
                    		double vx2 = -0.5;
                    		double vx_m = 0.5*(vx1-vx2);
				
				int idy = y - 1 + y_start;
				
		            	if (idy >= 0 && idy < InputP.Ny/4){
		                	rho[index] = rho1 - rho_m*exp((1.0*idy/InputP.Ny - 0.25)/0.025);
		                	vx[index] = vx1 - vx_m*exp((1.0*idy/InputP.Ny - 0.25)/0.025);
		            	} else if (idy >= InputP.Ny/4 && idy < InputP.Ny/2){
		                	rho[index] = rho2 + rho_m*exp((-1.0*idy/InputP.Ny + 0.25)/0.025);
		                	vx[index] = vx2 + vx_m*exp((-1.0*idy/InputP.Ny + 0.25)/0.025);
		            	} else if (idy >= InputP.Ny/2 && idy < 3*InputP.Ny/4){
		                	rho[index] = rho2 + rho_m*exp((1.0*idy/InputP.Ny - 0.75)/0.025);
		                	vx[index] = vx2 + vx_m*exp((1.0*idy/InputP.Ny - 0.75)/0.025);
		            	} else {
		                	rho[index] = rho1 - rho_m*exp((-1.0*idy/InputP.Ny + 0.75)/0.025);
		                	vx[index] = vx1 - vx_m*exp((-1.0*idy/InputP.Ny + 0.75)/0.025);
		            	}

		            	vy[index] = 0.01*sin(4.0*M_PI*x/InputP.Nx);
			}
		}
	}
}

void write_output_vtk(double* rho, double* p, double* vx, double*vy, InputPara InputP, int t, int rank, int size, int local_Ny, int y_start)
{

	/* Gathering data */
	if(rank == 0)
	{
		int index;

	    	std::string name = "output_" + std::to_string(t) + ".vtk";
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
	    	
	    	double* buffer = (double*)aligned_alloc(4096, (InputP.Nx + 2)*InputP.Ny*InputP.Nz*sizeof(double));
	    	double* buffer2 = (double*)aligned_alloc(4096, (InputP.Nx + 2)*InputP.Ny*InputP.Nz*sizeof(double));
		int count, offset;
		
		// gather rho
		std::memcpy(buffer, rho + (InputP.Nx + 2), local_Ny * (InputP.Nx + 2)*sizeof(double));
		for (int i=1; i<size; i++)
        	{
            		MPI_CALL(MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(&count, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(buffer + offset, count, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        	}
        	
        	// write rho
		ofile << "SCALARS rho double" << std::endl;
		ofile << "LOOKUP_TABLE default" << std::endl;
		for (int k = 0; k < InputP.Nz; k++) {
			for (int j = 0; j < InputP.Ny; j++) {
				for (int i = 1; i < InputP.Nx + 1; i++) {
					index = i + j * (InputP.Nx + 2) + k * (InputP.Nx + 2) * (InputP.Ny + 2);
					ofile << buffer[index] << " ";
				}
			}
		}
		
		// gather p
		std::memcpy(buffer, p + (InputP.Nx + 2), local_Ny * (InputP.Nx + 2)*sizeof(double));
		for (int i=1; i<size; i++)
        	{
            		MPI_CALL(MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(&count, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(buffer + offset, count, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        	}
        	
        	// write p
		ofile << "SCALARS p double" << std::endl;
		ofile << "LOOKUP_TABLE default" << std::endl;
		for (int k = 0; k < InputP.Nz; k++) {
			for (int j = 0; j < InputP.Ny; j++) {
				for (int i = 1; i < InputP.Nx + 1; i++) {
					index = i + j * (InputP.Nx + 2) + k * (InputP.Nx + 2) * (InputP.Ny + 2);
					ofile << buffer[index] << " ";
				}
			}
		}
		
		// gather vx
		std::memcpy(buffer, vx + (InputP.Nx + 2), local_Ny * (InputP.Nx + 2)*sizeof(double));
		for (int i=1; i<size; i++)
        	{
            		MPI_CALL(MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(&count, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(buffer + offset, count, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        	}
        	
        	// gather vy
		std::memcpy(buffer2, vy + (InputP.Nx + 2), local_Ny * (InputP.Nx + 2)*sizeof(double));
		for (int i=1; i<size; i++)
        	{
            		MPI_CALL(MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(&count, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(buffer + offset, count, MPI_DOUBLE, i, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        	}
        	
        	// write vec
		ofile << "SCALARS vec double" << std::endl;
		ofile << "LOOKUP_TABLE default" << std::endl;
		for (int k = 0; k < InputP.Nz; k++) {
			for (int j = 0; j < InputP.Ny; j++) {
				for (int i = 1; i < InputP.Nx + 1; i++) {
					index = i + j * (InputP.Nx + 2) + k * (InputP.Nx + 2) * (InputP.Ny + 2);
					ofile << buffer[index] << " " << buffer2[index] << " 0.0 " << std::endl;
				}
			}
		}
		
        	free(buffer);
        	free(buffer2);
        	ofile << std::endl;	
		ofile.close();
	}
	else
	{
		int offset = y_start*(InputP.Nx + 2);
        	int count = local_Ny*(InputP.Nx + 2);
        	MPI_CALL(MPI_Send(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(rho + (InputP.Nx + 2), count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(p + (InputP.Nx + 2), count, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(vx + (InputP.Nx + 2), count, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(vy + (InputP.Nx + 2), count, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD));
	}
}

//Calculate real-time values, e.g. the velocity of main tip, wake_up portion, steps per second
void Calc_RealTime_Values(InputPara InputP, int time, int RT_freq, time_t sim_time, int rank)
{
	if(rank == 0)
	{
		//std::cout << "--------------Writting Output of Wakeup Portion Now--------------" << std::endl;
		std::string OutputFileName;
		OutputFileName = "RTValues.out";
		
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

			//Calculate steps per second
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

			//Calculate steps per second
			double steps_per_second = RT_freq / ((double)sim_time / CLOCKS_PER_SEC);

			outf << "Time Steps_per_second" << std::endl;
			outf << time << " " << steps_per_second << std::endl;
			outf.close();
			//std::cout << "--------------Output Done--------------" << std::endl;
		}
	}
}

void Integral(double* data, int time, InputPara InputP, int rank, int local_Ny)
{
	double sum = 0.0;
	for (int z = 0; z < InputP.Nz; z++){
		for (int y = 1; y < local_Ny + 1; y++){
			for (int x = 1; x < InputP.Nx + 1; x++){

				int index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
				sum += data[index];
			}
		}
	}
	
	double sum_tot = 0.0;
	
	MPI_Allreduce(&sum, &sum_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);	
				
 	if(rank == 0)
	{
		std::string OutputFileName;
		OutputFileName = "Integral_mass.out";
		std::ofstream outf;
		outf.open(OutputFileName, std::fstream::app);	
		
		outf << time << " " << sum_tot << std::endl;
		std::cout << "Integral of mass = " << sum_tot << std::endl;
		outf.close();
	}   	
}

// calculate the amplitude of the y-velocity mode of the instability
void Calc_Vy_mode(double* vy, int time, InputPara InputP, int rank, int size, int local_Ny, int y_start){

    	// normalized grid points
    	double sum_s = 0.0;
    	double sum_c = 0.0;
    	double sum_d = 0.0;
    	for (int z = 0; z < InputP.Nz; z++){
        	for (int y = 1; y < local_Ny + 1; y++){
            		for (int x = 1; x < InputP.Nx + 1; x++){
                		int index = x + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
                		double x_s = (double)(x - 1)/InputP.Nx; // scaled x
                		double y_s = (double)(y - 1 + y_start)/InputP.Ny;
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
    	
    	double sum_s_tot = 0.0;
    	double sum_c_tot = 0.0;
    	double sum_d_tot = 0.0;
    	
    	MPI_Allreduce(&sum_s, &sum_s_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    	MPI_Allreduce(&sum_c, &sum_c_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    	MPI_Allreduce(&sum_d, &sum_d_tot, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    	
    	if(rank == 0)
    	{
    		std::string OutputFileName;
		OutputFileName = "VY_mode.out";
		std::ofstream outf;
		outf.open(OutputFileName, std::fstream::app);
		
		double M = 2.0 * sqrt(sum_s_tot*sum_s_tot/(sum_d_tot*sum_d_tot) + sum_c_tot*sum_c_tot/(sum_d_tot*sum_d_tot));
		outf << time << " " << M << std::endl;
		std::cout << "Growth mode amplitude = " << M << std::endl;
    		outf.close();
    	}
}

int main(int argc, char *argv[])
{   
    	/*-------------------For performance test-------------------*/
	time_t t_start_tot, t_end_tot;
	t_start_tot = clock();
	time_t start_output_time, end_output_time;
	float output_time_total = 0;
	
	int rank, size;
    	MPI_CALL(MPI_Init(&argc, &argv));
    	MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));  // Get the rank of the process
    	MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));  // Get the total number of processes

    	InputPara InputP;
	ReadInput("Input.txt", InputP);
	
	// Calculate the start and end indices for each process
    	int chunk_size = InputP.Ny / size;
    	int remainder = InputP.Ny % size;
    	int local_Ny, y_start;
    	if(rank < remainder)
    	{
    		local_Ny = chunk_size + 1;
    		y_start = local_Ny * rank;
    	}
    	else
    	{
    		local_Ny = chunk_size;
    		y_start = local_Ny * rank + remainder;
    	}
    	
    	std::cout << "I'm proc " << rank << "/" << size << ", my y_start = " << y_start << ", and my local_Ny = " << local_Ny << std::endl;

    	Variable Var;
	Allocate(Var, InputP, local_Ny);

    	size_t total_size = (InputP.Nx + 2) * (local_Ny + 2) * InputP.Nz * sizeof(double);
    	std::cout << "Total memory allocated: " \
             << 32 * total_size/(1024.*1024.*1024.) << " Gb\n" << std::endl;
    
    	// Initialization and write file
    	Initialize(Var.rho, Var.p, Var.vx, Var.vy, InputP, local_Ny, y_start);
    	Communicate_MPI_boundary(Var.rho, InputP, rank, size, local_Ny);
    	Communicate_MPI_boundary(Var.p, InputP, rank, size, local_Ny);
    	Communicate_MPI_boundary(Var.vx, InputP, rank, size, local_Ny);
    	Communicate_MPI_boundary(Var.vy, InputP, rank, size, local_Ny);
    	Set_boundary(Var.rho, InputP, local_Ny);
    	Set_boundary(Var.p, InputP, local_Ny);
    	Set_boundary(Var.vx, InputP, local_Ny);
    	Set_boundary(Var.vy, InputP, local_Ny);
    	
    	MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    	
    	std::cout << "Initialization finished." << std::endl;
    
    	start_output_time = clock();
    	if(InputP.output_mode > 1)
    	{
    		write_output_vtk(Var.rho, Var.p, Var.vx, Var.vy, InputP, 0, rank, size, local_Ny, y_start);
    		std::cout << "Write initial output file finished." << std::endl;
	}
	end_output_time = clock();
	output_time_total += (float)(end_output_time - start_output_time);
    
    	/*-------------------For performance test-------------------*/
    	time_t t_start_loop, t_end_loop, sim_time_start, sim_time_end;
	t_start_loop = clock();
	sim_time_start = clock();

    	// calculate conserved variables
    	getConserved(Var.Mass, Var.Energy, Var.Momx, Var.Momy, \
                                            Var.rho, Var.p, Var.vx, Var.vy, InputP, local_Ny);

    	// test conservation laws
    	Integral(Var.Mass, 0, InputP, rank, local_Ny);

    	// Main loop
    	int t = 1;
    	printf("Start main loop...\n\n");
    	while (t <= InputP.t_total) {
            	// calculate primitive variables
            	getPrimitive(Var.rho, Var.p, Var.vx, Var.vy, \
                                                Var.Mass, Var.Energy, Var.Momx, Var.Momy, \
                                                InputP, local_Ny);
                
                // Communication of new results
            	Communicate_MPI_boundary(Var.rho, InputP, rank, size, local_Ny);
    		Communicate_MPI_boundary(Var.p, InputP, rank, size, local_Ny);
    		Communicate_MPI_boundary(Var.vx, InputP, rank, size, local_Ny);
    		Communicate_MPI_boundary(Var.vy, InputP, rank, size, local_Ny);
    		Set_boundary(Var.rho, InputP, local_Ny);
    		Set_boundary(Var.p, InputP, local_Ny);
    		Set_boundary(Var.vx, InputP, local_Ny);
    		Set_boundary(Var.vy, InputP, local_Ny);
    		MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
            
            	// extrapolate primitive variables in time
            	extrap_in_time(Var.rho, Var.p, Var.vx, Var.vy, \
                                                Var.rho_dx, Var.rho_dy, Var.p_dx, Var.p_dy, \
                                                Var.vx_dx, Var.vx_dy, Var.vy_dx, Var.vy_dy, \
                                                Var.rho_prime, Var.p_prime, Var.vx_prime, Var.vy_prime, \
                                                InputP, local_Ny);
                                                
                // Communication of new results
                Communicate_MPI_boundary(Var.rho_prime, InputP, rank, size, local_Ny);
    		Communicate_MPI_boundary(Var.p_prime, InputP, rank, size, local_Ny);
    		Communicate_MPI_boundary(Var.vx_prime, InputP, rank, size, local_Ny);
    		Communicate_MPI_boundary(Var.vy_prime, InputP, rank, size, local_Ny);
    		Communicate_MPI_boundary(Var.rho_dx, InputP, rank, size, local_Ny);
    		Communicate_MPI_boundary(Var.p_dx, InputP, rank, size, local_Ny);
    		Communicate_MPI_boundary(Var.vx_dx, InputP, rank, size, local_Ny);
    		Communicate_MPI_boundary(Var.vy_dx, InputP, rank, size, local_Ny);
    		Communicate_MPI_boundary(Var.rho_dy, InputP, rank, size, local_Ny);
    		Communicate_MPI_boundary(Var.p_dy, InputP, rank, size, local_Ny);
    		Communicate_MPI_boundary(Var.vx_dy, InputP, rank, size, local_Ny);
    		Communicate_MPI_boundary(Var.vy_dy, InputP, rank, size, local_Ny);
    		Set_boundary(Var.rho_prime, InputP, local_Ny);
    		Set_boundary(Var.p_prime, InputP, local_Ny);
    		Set_boundary(Var.vx_prime, InputP, local_Ny);
    		Set_boundary(Var.vy_prime, InputP, local_Ny);
    		Set_boundary(Var.rho_dx, InputP, local_Ny);
    		Set_boundary(Var.p_dx, InputP, local_Ny);
    		Set_boundary(Var.vx_dx, InputP, local_Ny);
    		Set_boundary(Var.vy_dx, InputP, local_Ny);
    		Set_boundary(Var.rho_dy, InputP, local_Ny);
    		Set_boundary(Var.p_dy, InputP, local_Ny);
    		Set_boundary(Var.vx_dy, InputP, local_Ny);
    		Set_boundary(Var.vy_dy, InputP, local_Ny);
    		MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));

            
            // calculate fluxes to conserved variables
            calc_flux(Var.rho, Var.p, Var.vx, Var.vy, \
                                                Var.rho_dx, Var.rho_dy, Var.p_dx, Var.p_dy, \
                                                Var.vx_dx, Var.vx_dy, Var.vy_dx, Var.vy_dy, \
                                                Var.rho_prime, Var.p_prime, Var.vx_prime, Var.vy_prime, \
                                                Var.flux_Mass_x, Var.flux_Energy_x, Var.flux_Momx_x, Var.flux_Momy_x, \
                                                Var.flux_Mass_y, Var.flux_Energy_y, Var.flux_Momx_y, Var.flux_Momy_y, \
                                                InputP, local_Ny);
                                                
            // Communication of new results
            Communicate_MPI_boundary(Var.flux_Mass_x, InputP, rank, size, local_Ny);
            Communicate_MPI_boundary(Var.flux_Energy_x, InputP, rank, size, local_Ny);
            Communicate_MPI_boundary(Var.flux_Momx_x, InputP, rank, size, local_Ny);
            Communicate_MPI_boundary(Var.flux_Momy_x, InputP, rank, size, local_Ny);
            Communicate_MPI_boundary(Var.flux_Mass_y, InputP, rank, size, local_Ny);
            Communicate_MPI_boundary(Var.flux_Energy_y, InputP, rank, size, local_Ny);
            Communicate_MPI_boundary(Var.flux_Momx_y, InputP, rank, size, local_Ny);
            Communicate_MPI_boundary(Var.flux_Momy_y, InputP, rank, size, local_Ny);
            Set_boundary(Var.flux_Mass_x, InputP, local_Ny);
            Set_boundary(Var.flux_Energy_x, InputP, local_Ny);
            Set_boundary(Var.flux_Momx_x, InputP, local_Ny);
            Set_boundary(Var.flux_Momy_x, InputP, local_Ny);
            Set_boundary(Var.flux_Mass_y, InputP, local_Ny);
            Set_boundary(Var.flux_Energy_y, InputP, local_Ny);
            Set_boundary(Var.flux_Momx_y, InputP, local_Ny);
            Set_boundary(Var.flux_Momy_y, InputP, local_Ny);
            MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
            
            // apply fluxes to conserved variables
            apply_flux(Var.Mass, Var.Energy, Var.Momx, Var.Momy, \
                                                Var.flux_Mass_x, Var.flux_Energy_x, Var.flux_Momx_x, Var.flux_Momy_x, \
                                                Var.flux_Mass_y, Var.flux_Energy_y, Var.flux_Momx_y, Var.flux_Momy_y, \
                                                InputP, local_Ny);
        
        // write vtk files
        start_output_time = clock();
        if(InputP.output_mode)
        {
        	if (t % InputP.t_freq == 0) {
            		std::cout << "Timestep " << t << std::endl;

            		sim_time_end = clock();
			time_t sim_time = sim_time_end - sim_time_start;
			Calc_RealTime_Values(InputP, t, InputP.t_freq, sim_time, rank);
			sim_time_start = clock();
		
			if(InputP.output_mode > 1)
			{
				getPrimitive(Var.rho, Var.p, Var.vx, Var.vy, \
                                                Var.Mass, Var.Energy, Var.Momx, Var.Momy, \
                                                InputP, local_Ny);
        
            			write_output_vtk(Var.rho, Var.p, Var.vx, Var.vy, InputP, t, rank, size, local_Ny, y_start);
		
            			Integral(Var.Mass, t, InputP, rank, local_Ny);
            			Calc_Vy_mode(Var.vy, t, InputP, rank, size, local_Ny, y_start);
            		}
        	}
        }
        end_output_time = clock();
	output_time_total += (float)(end_output_time - start_output_time);
        t++;
    }
    	t_end_loop = clock();

	std::cout << std::endl << "Free allocated memory..." << std::endl;
	FreeMemory(Var);

    	/*-------------------For performance test-------------------*/
    	MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
	t_end_tot = clock();
	if(rank == 0)
	{
		printf("\nThe overall running time is: %f sec.\n", ((float)(t_end_tot - t_start_tot)) / CLOCKS_PER_SEC);

		printf("The loop running time is: %f sec. %3f percent of overall running time.\n", ((float)(t_end_loop - t_start_loop)) / CLOCKS_PER_SEC, (float)(t_end_loop - t_start_loop) / (double)(t_end_tot - t_start_tot) * 100.);
		
		printf("\nThe overall running time (subtract output) is: %f sec.\n", (((float)(t_end_tot - t_start_tot)) - output_time_total) / CLOCKS_PER_SEC);

		std::cout << std::endl << "Program finished." << std::endl;
	}
	
	MPI_CALL(MPI_Finalize());
	return 0;
}
