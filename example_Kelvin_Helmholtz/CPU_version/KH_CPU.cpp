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

		void print_input()
		{
			printf("--------------Input parameters--------------\n");
			printf("Size of system: %d, %d, %d\n", Nx, Ny, Nz);
			printf("totalTime = %d,	printFreq = %d, dt = %f\n", t_total, t_freq, dt);
			printf("dx = %lf, dy = %lf, dz =%lf\n", dx, dy, dz);
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

	/* Close input file */
	infile.close();
	std::cout << "Done with input reading." << std::endl;

	InputP.print_input();
}

//Allocate memory
void Allocate(Variable & Var, InputPara InputP)
{
	size_t size = InputP.Nx * InputP.Ny * InputP.Nz * sizeof(double);

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

double Grad_sq(double* c, int x, int y, int z, InputPara InputP)
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

double Grad_x(double*c, int x, int y, int z, InputPara InputP)
{
    int xm, xp;
    xm = (x - 1 >= 0) ? x - 1 : InputP.Nx - 1;
    xp = (x + 1 < InputP.Nx) ? x + 1 : 0;

    int index_xm = xm + y * InputP.Nx + z * InputP.Nx * InputP.Ny;
    int index_xp = xp + y * InputP.Nx + z * InputP.Nx * InputP.Ny;

    double dpx = (c[index_xp] - c[index_xm]) / 2.0 / InputP.dx;

    return dpx;
}

double Grad_y(double*c, int x, int y, int z, InputPara InputP)
{
    int ym, yp;
    ym = (y - 1 >= 0) ? y - 1 : InputP.Ny - 1;
    yp = (y + 1 < InputP.Ny) ? y + 1 : 0;

    int index_ym = x + ym * InputP.Nx + z * InputP.Nx * InputP.Ny;
    int index_yp = x + yp * InputP.Nx + z * InputP.Nx * InputP.Ny;

    double dpy = (c[index_yp] - c[index_ym]) / 2.0 / InputP.dy;

    return dpy;
}

double Laplacian(double* c, int x, int y, int z, InputPara InputP)
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

void getConserved(double* Mass, double* Energy, double* Momx, double* Momy, \
                            double* rho, double* p, double* vx, double* vy, \
                            InputPara InputP)
{   
    	int x, y, z, index;
    	double gamma = 5.0/3.0;
   	double vol = InputP.dx * InputP.dy;
    
    	for (int z = 0; z < InputP.Nz; z++){
		for (int y = 0; y < InputP.Ny; y++){
			for (int x = 0; x < InputP.Nx; x++){

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

void getPrimitive(double* rho, double* p, double* vx, double* vy, \
                            double* Mass, double* Energy, double* Momx, double* Momy, \
                            InputPara InputP)
{   
    	int x, y, z, index;
    	double gamma = 5.0/3.0;
    	double vol = InputP.dx * InputP.dy;
    
    	for (int z = 0; z < InputP.Nz; z++){
		for (int y = 0; y < InputP.Ny; y++){
			for (int x = 0; x < InputP.Nx; x++){

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

void extrap_in_time(double* rho, double* p, double* vx, double* vy, \
                            double* rho_dx, double* rho_dy, double* p_dx, double* p_dy, \
                            double* vx_dx, double* vx_dy, double* vy_dx, double* vy_dy, \
                            double* rho_prime, double* p_prime, double* vx_prime, double* vy_prime, \
                            InputPara InputP)
{
    	int x, y, z, index;
    	double gamma = 5.0/3.0;
    
	    for (int z = 0; z < InputP.Nz; z++){
		for (int y = 0; y < InputP.Ny; y++){
			for (int x = 0; x < InputP.Nx; x++){

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

void extrapolateInSpaceToFace(double face[4], double* f, double* f_dx, double* f_dy,
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

void getFlux(double* flux_Mass, double* flux_Energy, double* flux_Momx, double* flux_Momy, \
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

void calc_flux(double* rho, double* p, double* vx, double* vy, \
                        double* rho_dx, double* rho_dy, double* p_dx, double* p_dy, \
                        double* vx_dx, double* vx_dy, double* vy_dx, double* vy_dy, \
                        double* rho_prime, double* p_prime, double* vx_prime, double* vy_prime, \
                        double* flux_Mass_x, double* flux_Energy_x, double* flux_Momx_x, double* flux_Momy_x, \
                        double* flux_Mass_y, double* flux_Energy_y, double* flux_Momx_y, double* flux_Momy_y, \
                        InputPara InputP)
{
	    for (int z = 0; z < InputP.Nz; z++){
		for (int y = 0; y < InputP.Ny; y++){
			for (int x = 0; x < InputP.Nx; x++){

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

    f[index] += InputP.dt * InputP.dx * (-1.0*flux_x[index] + flux_x[index_xm] \
                                        -1.0*flux_y[index] + flux_y[index_ym]);
}

void apply_flux(double* Mass, double* Energy, double* Momx, double* Momy, \
                        double* flux_Mass_x, double* flux_Energy_x, double* flux_Momx_x, double* flux_Momy_x, \
                        double* flux_Mass_y, double* flux_Energy_y, double* flux_Momx_y, double* flux_Momy_y, \
                        InputPara InputP)
{
	    for (int z = 0; z < InputP.Nz; z++){
		for (int y = 0; y < InputP.Ny; y++){
			for (int x = 0; x < InputP.Nx; x++){
   
                    		applyFluxes(Mass, flux_Mass_x, flux_Mass_y, x, y, z, InputP);
                    		applyFluxes(Momx, flux_Momx_x, flux_Momx_y, x, y, z, InputP);
                    		applyFluxes(Momy, flux_Momy_x, flux_Momy_y, x, y, z, InputP);
                    		applyFluxes(Energy, flux_Energy_x, flux_Energy_y, x, y, z, InputP);
			}
		}
	}
}

//Initialize system
void Initialize(double* rho, double* p, double* vx, double* vy, InputPara InputP)
{
	int x, y, z, index;
	for (int z = 0; z < InputP.Nz; z++){
		for (int y = 0; y < InputP.Ny; y++){
			for (int x = 0; x < InputP.Nx; x++){

				index = x + y * InputP.Nx + z * InputP.Nx * InputP.Ny;

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

void write_output_vtk(double* rho, double* p, double* vx, double*vy, InputPara InputP, int t)
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

//Calculate real-time values, e.g. the velocity of main tip, wake_up portion, steps per second
void Calc_RealTime_Values(InputPara InputP, int time, int RT_freq, time_t sim_time)
{
	//std::cout << "--------------Writting Output of Wakeup Portion Now--------------" << std::endl;
	std::string OutputFileName;
	OutputFileName = "RTValues.out";

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

void Integral(double* data, int time, InputPara InputP)
{
    std::string OutputFileName;
	OutputFileName = "Integral_mass.out";
	std::ofstream outf;
	outf.open(OutputFileName, std::fstream::app);

	double sum = 0.0;
    for (int index = 0; index < InputP.Nx*InputP.Ny*InputP.Nz; index++){
		sum += data[index];
	}
	outf << time << " " << sum << std::endl;
	std::cout << "Integral of mass = " << sum << std::endl;

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

int main()
{   
    	/*-------------------For performance test-------------------*/
	time_t t_start_tot, t_end_tot;
	t_start_tot = clock();

    	InputPara InputP;
	ReadInput("Input.txt", InputP);

    	Variable Var;
	Allocate(Var, InputP);

    	size_t total_size = InputP.Nx * InputP.Ny * InputP.Nz * sizeof(double);
    	std::cout << "Total memory allocated: " \
             << 32 * total_size/(1024.*1024.*1024.) << " Gb\n" << std::endl;
    
    	// Initialization and write file
    	Initialize(Var.rho, Var.p, Var.vx, Var.vy, InputP);
    	std::cout << "Initialization finished." << std::endl;
    
    	write_output_vtk(Var.rho, Var.p, Var.vx, Var.vy, InputP, 0);
    	std::cout << "Write initial output file finished." << std::endl;

    
    	/*-------------------For performance test-------------------*/
    	time_t t_start_loop, t_end_loop, sim_time_start, sim_time_end;
	t_start_loop = clock();
	sim_time_start = clock();

    	// calculate conserved variables
    	getConserved(Var.Mass, Var.Energy, Var.Momx, Var.Momy, \
                                            Var.rho, Var.p, Var.vx, Var.vy, InputP);

    	// test conservation laws
    	Integral(Var.Mass, 0, InputP);

    	// Main loop
    	int t = 1;
    	printf("Start main loop...\n\n");
    	while (t <= InputP.t_total) {
            // calculate primitive variables
            getPrimitive(Var.rho, Var.p, Var.vx, Var.vy, \
                                                Var.Mass, Var.Energy, Var.Momx, Var.Momy, \
                                                InputP);
            
            // extrapolate primitive variables in time
            extrap_in_time(Var.rho, Var.p, Var.vx, Var.vy, \
                                                Var.rho_dx, Var.rho_dy, Var.p_dx, Var.p_dy, \
                                                Var.vx_dx, Var.vx_dy, Var.vy_dx, Var.vy_dy, \
                                                Var.rho_prime, Var.p_prime, Var.vx_prime, Var.vy_prime, \
                                                InputP);

            
            // calculate fluxes to conserved variables
            calc_flux(Var.rho, Var.p, Var.vx, Var.vy, \
                                                Var.rho_dx, Var.rho_dy, Var.p_dx, Var.p_dy, \
                                                Var.vx_dx, Var.vx_dy, Var.vy_dx, Var.vy_dy, \
                                                Var.rho_prime, Var.p_prime, Var.vx_prime, Var.vy_prime, \
                                                Var.flux_Mass_x, Var.flux_Energy_x, Var.flux_Momx_x, Var.flux_Momy_x, \
                                                Var.flux_Mass_y, Var.flux_Energy_y, Var.flux_Momx_y, Var.flux_Momy_y, \
                                                InputP);
            
            // apply fluxes to conserved variables
            apply_flux(Var.Mass, Var.Energy, Var.Momx, Var.Momy, \
                                                Var.flux_Mass_x, Var.flux_Energy_x, Var.flux_Momx_x, Var.flux_Momy_x, \
                                                Var.flux_Mass_y, Var.flux_Energy_y, Var.flux_Momx_y, Var.flux_Momy_y, \
                                                InputP);
        
        // write vtk files
        if (t % InputP.t_freq == 0) {
            std::cout << "Timestep " << t << std::endl;

            getPrimitive(Var.rho, Var.p, Var.vx, Var.vy, \
                                                Var.Mass, Var.Energy, Var.Momx, Var.Momy, \
                                                InputP);
        
            write_output_vtk(Var.rho, Var.p, Var.vx, Var.vy, InputP, t);

            sim_time_end = clock();
			time_t sim_time = sim_time_end - sim_time_start;
			Calc_RealTime_Values(InputP, t, InputP.t_freq, sim_time);
			sim_time_start = clock();
            Integral(Var.Mass, t, InputP);
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

	std::cout << std::endl << "Program finished." << std::endl;
	return 0;
}
