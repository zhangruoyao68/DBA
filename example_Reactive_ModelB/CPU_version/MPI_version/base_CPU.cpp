#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <random>
#include <cstring>
#include <cpuid.h>  // GCC-provided

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
	double* x_old, * y_old, * z_old, * phi_old, * b;
	double* x_new, * y_new, * z_new, * phi_new;
	double* dfdx, * dfdy, * dfdz;
	double* rand;
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
	double Metric_eps;
	int mode, output_mode;

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
		printf("output_mode = %d\n", output_mode);
		printf("-------------------------------------------\n\n");
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
	size_t size = (InputP.Nx + 2) * (InputP.Ny + 2) * (local_Nz + 2) * sizeof(double);

	Var.x_old = (double*)aligned_alloc(4096, size);
	Var.y_old = (double*)aligned_alloc(4096, size);
	Var.z_old = (double*)aligned_alloc(4096, size);
	Var.phi_old = (double*)aligned_alloc(4096, size);
	Var.b = (double*)aligned_alloc(4096, size);
	Var.x_new = (double*)aligned_alloc(4096, size);
	Var.y_new = (double*)aligned_alloc(4096, size);
	Var.z_new = (double*)aligned_alloc(4096, size);
	Var.phi_new = (double*)aligned_alloc(4096, size);
	Var.dfdx = (double*)aligned_alloc(4096, size);
	Var.dfdy = (double*)aligned_alloc(4096, size);
	Var.dfdz = (double*)aligned_alloc(4096, size);
	Var.rand = (double*)aligned_alloc(4096, size);

	std::cout << "Total managed memory allocated: " \
		<< (13 * size ) / (1024. * 1024. * 1024.) << " Gb\n" << std::endl;
}

//Initialize system
void Initialize(double* x, double* y, double* z, double* b, double* phi, InputPara InputP, int local_Nz)
{
	double amp = 0.05;
	double z_amp = 0.001;

	for (int k = 1; k < local_Nz + 1; k++)
		for (int j = 1; j < InputP.Ny + 1; j++)
			for (int i = 1; i < InputP.Nx + 1; i++)
			{
				int index = i + j * (InputP.Nx + 2) + k * (InputP.Nx + 2) * (InputP.Ny + 2);

				x[index] = InputP.x0 + amp * (x[index] - 0.5);
				y[index] = InputP.y0 + amp * (y[index] - 0.5);
				z[index] = InputP.z0 + z_amp * (z[index] - 0.5);

				b[index] = 1.0 - x[index] - y[index] - z[index];
				phi[index] = 0;
			}
}

//Communicate boundary between processes
void Communicate_MPI_boundary(double* c, InputPara InputP, int rank, int size, int local_Nz)
{
	int rankm = ((rank == 0) ? (size - 1) : (rank - 1));
	int rankp = ((rank == size - 1) ? (0) : (rank + 1));
	
	int layer_size = (InputP.Nx + 2) * (InputP.Ny + 2);
	
	MPI_CALL(MPI_Sendrecv(c + 1 * layer_size, layer_size, MPI_DOUBLE, rankm, 0, c + (local_Nz + 1) * layer_size, layer_size, MPI_DOUBLE, rankp, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
	MPI_CALL(MPI_Sendrecv(c + (local_Nz) * layer_size, layer_size, MPI_DOUBLE, rankp, 1, c, layer_size, MPI_DOUBLE, rankm, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));	
}

//Set boundary conditions
void Set_boundary(double* c, InputPara InputP, int local_Nz)
{
	// Set periodic BC for x and y directions, z no need due to MPI in z
	int index, index_t;
	for(int z = 0; z < local_Nz + 2; z++)
		for (int y = 0; y < InputP.Ny + 2; y++)
		{
			index = 0 + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
			index_t = InputP.Nx + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
			c[index] = c[index_t];
			
			index = (InputP.Nx + 1) + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
			index_t = 1 + y * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
			c[index] = c[index_t];
		}
		
	for(int z = 0; z < local_Nz + 2; z++)
		for (int x = 0; x < InputP.Nx + 2; x++)
		{
			index = x + 0 * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
			index_t = x + InputP.Ny * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
			c[index] = c[index_t];
			
			index = x + (InputP.Ny + 1) * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
			index_t = x + 1 * (InputP.Nx + 2) + z * (InputP.Nx + 2) * (InputP.Ny + 2);
			c[index] = c[index_t];
		}
}

double Laplacian(double* c, int x, int y, int z, InputPara InputP)
{
	int xm, xp, ym, yp, zm, zp;

	//Finite Difference
	xm = x - 1;
	xp = x + 1;
	ym = y - 1;
	yp = y + 1;
	zm = z - 1;
	zp = z + 1;

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

void Calc_mu(double* x, double* y, double* z, double* b, double* phi, \
	double* dfdx, double* dfdy, double* dfdz, \
InputPara Param, int local_Nz)
{
	int idx, idy, idz;
	for (int k = 1; k < local_Nz + 1; k++)
		for (int j = 1; j < Param.Ny + 1; j++)
			for (int i = 1; i < Param.Nx + 1; i++)
			{
				idx = i;
				idy = j;
				idz = k;
				int index = i + j * (Param.Nx + 2) + k * (Param.Nx + 2) * (Param.Ny + 2);
				b[index] = 1.0 - x[index] - y[index] - z[index];

				dfdx[index] = -1.0 + 1.0 / Param.rx + Param.chi_xy * y[index] + Param.chi_xz * z[index] \
					+ Param.chi_xb * b[index] - Param.chi_xb * x[index] - Param.chi_yb * y[index] \
					- Param.chi_zb * z[index] + log(x[index]) / Param.rx - log(b[index]) \
					- Param.epsilonx_sq * Laplacian(x, idx, idy, idz, Param);

				dfdy[index] = -1.0 + 1.0 / Param.ry + Param.chi_xy * x[index] + Param.chi_yz * z[index] \
					+ Param.chi_yb * b[index] - Param.chi_xb * x[index] - Param.chi_yb * y[index] \
					- Param.chi_zb * z[index] + log(y[index]) / Param.ry - log(b[index]) \
					- Param.epsilony_sq * Laplacian(y, idx, idy, idz, Param);

				dfdz[index] = -1.0 + 1.0 / Param.rz + Param.chi_xz * x[index] + Param.chi_yz * y[index] \
					+ Param.chi_zb * b[index] - Param.chi_xb * x[index] - Param.chi_yb * y[index] \
					- Param.chi_zb * z[index] + log(z[index]) / Param.rz - log(b[index]) \
					- Param.p_gel * phi[index] * phi[index] / 2.0 / (1.0 - Param.z_crit) \
					- Param.epsilonz_sq * Laplacian(z, idx, idy, idz, Param) \
					- log(Param.K) / Param.rz;
			}
}

void Update(double* xnew, double* ynew, double* znew, double* phinew, \
	double* xold, double* yold, double* zold, double* phiold, \
	double* b, \
	double* dfdx, double* dfdy, double* dfdz, \
	InputPara Param, int local_Nz)
{
	int idx, idy, idz;
	double R;
	for (int k = 1; k < local_Nz + 1; k++)
		for (int j = 1; j < Param.Ny + 1; j++)
			for (int i = 1; i < Param.Nx + 1; i++)
			{
				idx = i;
				idy = j;
				idz = k;
				int index = i + j * (Param.Nx + 2) + k * (Param.Nx + 2) * (Param.Ny + 2);

				R = Param.k_0 * (exp(Param.n * Param.vx * dfdx[index] + Param.m * Param.vy * dfdy[index]) \
					- exp(Param.vz * dfdz[index]));

				xnew[index] = xold[index] + Param.dt * Param.vx \
					* (Param.Mobility_x * Laplacian(dfdx, idx, idy, idz, Param) \
						- Param.n * R);

				ynew[index] = yold[index] + Param.dt * Param.vy \
					* (Param.Mobility_y * Laplacian(dfdy, idx, idy, idz, Param) \
						- Param.m * R);

				znew[index] = zold[index] + Param.dt * Param.vz \
					* (Param.Mobility_z * Laplacian(dfdz, idx, idy, idz, Param) \
						+ R);

				b[index] = 1.0 - xnew[index] - ynew[index] - znew[index];
			}
}

void Swap_regular(double* x, double* x_new, \
	double* y, double* y_new, \
	double* z, double* z_new, \
	InputPara Param, int local_Nz)
{
	for (int k = 1; k < local_Nz + 1; k++)
		for (int j = 1; j < Param.Ny + 1; j++)
			for (int i = 1; i < Param.Nx + 1; i++)
			{
				int index = i + j * (Param.Nx + 2) + k * (Param.Nx + 2) * (Param.Ny + 2);
				x[index] = x_new[index];
				y[index] = y_new[index];
				z[index] = z_new[index];
			}
}

//Output data in .vtk form
void write_output_vtk(double* x, double* y, double* z, double* b, double* phi, int t, InputPara Param, int rank, int size, int local_Nz, int z_start)
{
	/* Gathering data */
	if(rank == 0)
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

		double* buffer = (double*)aligned_alloc(4096, (Param.Nx + 2)*(Param.Ny + 2)*Param.Nz*sizeof(double));
		int count, offset;
		
		// gather x
		std::memcpy(buffer, x + (Param.Nx + 2)*(Param.Ny + 2), local_Nz*(Param.Nx + 2)*(Param.Ny + 2)*sizeof(double));
		for (int i=1; i<size; i++)
        	{
            		MPI_CALL(MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(&count, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(buffer + offset, count, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        	}
        	// write x
		ofile << "SCALARS X double" << std::endl;
		ofile << "LOOKUP_TABLE default" << std::endl;
		for (int k = 0; k < Param.Nz; k++) {
			for (int j = 1; j < Param.Ny + 1; j++) {
				for (int i = 1; i < Param.Nx + 1; i++) {
					index = i + j * (Param.Nx + 2) + k * (Param.Nx + 2) * (Param.Ny + 2);
					ofile << buffer[index] << " ";
				}
			}
		}
		
		// gather y
		ofile << std::endl;
		std::memcpy(buffer, y + (Param.Nx + 2)*(Param.Ny + 2), local_Nz*(Param.Nx + 2)*(Param.Ny + 2)*sizeof(double));
		for (int i=1; i<size; i++)
        	{
        		MPI_CALL(MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(&count, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(buffer + offset, count, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        	}
        	// write y
		ofile << "SCALARS Y double" << std::endl;
		ofile << "LOOKUP_TABLE default" << std::endl;
		for (int k = 0; k < Param.Nz; k++) {
			for (int j = 1; j < Param.Ny + 1; j++) {
				for (int i = 1; i < Param.Nx + 1; i++) {
					index = i + j * (Param.Nx + 2) + k * (Param.Nx + 2) * (Param.Ny + 2);
					ofile << buffer[index] << " ";
				}
			}
		}	
		
		// gather z
		ofile << std::endl;
		std::memcpy(buffer, z + (Param.Nx + 2)*(Param.Ny + 2), local_Nz*(Param.Nx + 2)*(Param.Ny + 2)*sizeof(double));
		for (int i=1; i<size; i++)
        	{
        		MPI_CALL(MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(&count, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(buffer + offset, count, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        	}
        	// write z
		ofile << "SCALARS Z double" << std::endl;
		ofile << "LOOKUP_TABLE default" << std::endl;
		for (int k = 0; k < Param.Nz; k++) {
			for (int j = 1; j < Param.Ny + 1; j++) {
				for (int i = 1; i < Param.Nx + 1; i++) {
					index = i + j * (Param.Nx + 2) + k * (Param.Nx + 2) * (Param.Ny + 2);
					ofile << buffer[index] << " ";
				}
			}
		}
		
		// gather b
		ofile << std::endl;
		std::memcpy(buffer, b + (Param.Nx + 2)*(Param.Ny + 2), local_Nz*(Param.Nx + 2)*(Param.Ny + 2)*sizeof(double));
		for (int i=1; i<size; i++)
        	{
        		MPI_CALL(MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(&count, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(buffer + offset, count, MPI_DOUBLE, i, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        	}
        	// write b
		ofile << "SCALARS B double" << std::endl;
		ofile << "LOOKUP_TABLE default" << std::endl;
		for (int k = 0; k < Param.Nz; k++) {
			for (int j = 1; j < Param.Ny + 1; j++) {
				for (int i = 1; i < Param.Nx + 1; i++) {
					index = i + j * (Param.Nx + 2) + k * (Param.Nx + 2) * (Param.Ny + 2);
					ofile << buffer[index] << " ";
				}
			}
		}
		
		// gather phi
		ofile << std::endl;
		std::memcpy(buffer, phi + (Param.Nx + 2)*(Param.Ny + 2), local_Nz*(Param.Nx + 2)*(Param.Ny + 2)*sizeof(double));
		for (int i=1; i<size; i++)
        	{
        		MPI_CALL(MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(&count, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(buffer + offset, count, MPI_DOUBLE, i, 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        	}
        	// write phi
		ofile << "SCALARS PHI double" << std::endl;
		ofile << "LOOKUP_TABLE default" << std::endl;
		for (int k = 0; k < Param.Nz; k++) {
			for (int j = 1; j < Param.Ny + 1; j++) {
				for (int i = 1; i < Param.Nx + 1; i++) {
					index = i + j * (Param.Nx + 2) + k * (Param.Nx + 2) * (Param.Ny + 2);
					ofile << buffer[index] << " ";
				}
			}
		}
        	
        	free(buffer);
        	ofile << std::endl;	
		ofile.close();
	}
	else
	{
		int offset = z_start*(Param.Nx + 2)*(Param.Ny + 2);
        	int count = local_Nz*(Param.Nx + 2)*(Param.Ny + 2);
        	MPI_CALL(MPI_Send(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(x + (Param.Nx + 2)*(Param.Ny + 2), count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(y + (Param.Nx + 2)*(Param.Ny + 2), count, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(z + (Param.Nx + 2)*(Param.Ny + 2), count, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(b + (Param.Nx + 2)*(Param.Ny + 2), count, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(phi + (Param.Nx + 2)*(Param.Ny + 2), count, MPI_DOUBLE, 0, 6, MPI_COMM_WORLD));
	}
}

//Calculate real-time values, e.g. the velocity of main tip, wake_up portion, steps per second
void Calc_RealTime_Values(InputPara InputP, int time, int RT_freq, time_t sim_time)
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

//Free memory
void FreeMemory(Variable& Var)
{
	free(Var.x_old);
	free(Var.y_old);
	free(Var.z_old);
	free(Var.x_new);
	free(Var.y_new);
	free(Var.z_new);
	free(Var.phi_old);
	free(Var.phi_new);
	free(Var.b);
	free(Var.dfdx);
	free(Var.dfdy);
	free(Var.dfdz);

	std::cout << "Memory freed" << std::endl;
}

void Integral(double* x, double* y, double* z, double* b, \
	int time, InputPara InputP, int rank, int size, int local_Nz, int z_start)
{
	/* Gathering data */
	if(rank == 0)
	{
		std::string OutputFileName;
		OutputFileName = "Integral.out";
		std::ofstream outf;
		outf.open(OutputFileName, std::fstream::app);

		double sum_x = 0.0;
		double sum_y = 0.0;
		double sum_z = 0.0;
		double sum_b = 0.0;
		
		double* buffer = (double*)aligned_alloc(4096, (InputP.Nx + 2)*(InputP.Ny + 2)*InputP.Nz*sizeof(double));
		int count, offset;
		
		int index;
		
		// gather x
		std::memcpy(buffer, x + (InputP.Nx + 2)*(InputP.Ny + 2), local_Nz*(InputP.Nx + 2)*(InputP.Ny + 2)*sizeof(double));
		for (int i=1; i<size; i++)
        	{
            		MPI_CALL(MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(&count, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(buffer + offset, count, MPI_DOUBLE, i, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        	}
		for (int k = 0; k < InputP.Nz; k++) {
			for (int j = 1; j < InputP.Ny + 1; j++) {
				for (int i = 1; i < InputP.Nx + 1; i++) {
					index = i + j * (InputP.Nx + 2) + k * (InputP.Nx + 2) * (InputP.Ny + 2);
					sum_x += buffer[index];
				}
			}
		}
		// gather y
		std::memcpy(buffer, y + (InputP.Nx + 2)*(InputP.Ny + 2), local_Nz*(InputP.Nx + 2)*(InputP.Ny + 2)*sizeof(double));
		for (int i=1; i<size; i++)
        	{
            		MPI_CALL(MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(&count, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(buffer + offset, count, MPI_DOUBLE, i, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        	}
		for (int k = 0; k < InputP.Nz; k++) {
			for (int j = 1; j < InputP.Ny + 1; j++) {
				for (int i = 1; i < InputP.Nx + 1; i++) {
					index = i + j * (InputP.Nx + 2) + k * (InputP.Nx + 2) * (InputP.Ny + 2);
					sum_y += buffer[index];
				}
			}
		}
		// gather z
		std::memcpy(buffer, z + (InputP.Nx + 2)*(InputP.Ny + 2), local_Nz*(InputP.Nx + 2)*(InputP.Ny + 2)*sizeof(double));
		for (int i=1; i<size; i++)
        	{
            		MPI_CALL(MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(&count, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(buffer + offset, count, MPI_DOUBLE, i, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        	}
		for (int k = 0; k < InputP.Nz; k++) {
			for (int j = 1; j < InputP.Ny + 1; j++) {
				for (int i = 1; i < InputP.Nx + 1; i++) {
					index = i + j * (InputP.Nx + 2) + k * (InputP.Nx + 2) * (InputP.Ny + 2);
					sum_z += buffer[index];
				}
			}
		}
		// gather b
		std::memcpy(buffer, b + (InputP.Nx + 2)*(InputP.Ny + 2), local_Nz*(InputP.Nx + 2)*(InputP.Ny + 2)*sizeof(double));
		for (int i=1; i<size; i++)
        	{
            		MPI_CALL(MPI_Recv(&offset, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(&count, 1, MPI_INT, i, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
            		MPI_CALL(MPI_Recv(buffer + offset, count, MPI_DOUBLE, i, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        	}
		for (int k = 0; k < InputP.Nz; k++) {
			for (int j = 1; j < InputP.Ny + 1; j++) {
				for (int i = 1; i < InputP.Nx + 1; i++) {
					index = i + j * (InputP.Nx + 2) + k * (InputP.Nx + 2) * (InputP.Ny + 2);
					sum_b += buffer[index];
				}
			}
		}
		outf << time << " " << sum_x << " " << sum_y << " " << sum_z << " " << sum_b << std::endl;
		std::cout << "Integral of buffer = " << sum_b << std::endl;
		outf.close();	
		free(buffer);	
	}
	else
	{
		int offset = z_start*(InputP.Nx + 2)*(InputP.Ny + 2);
        	int count = local_Nz*(InputP.Nx + 2)*(InputP.Ny + 2);
        	MPI_CALL(MPI_Send(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(x + (InputP.Nx + 2)*(InputP.Ny + 2), count, MPI_DOUBLE, 0, 2, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(y + (InputP.Nx + 2)*(InputP.Ny + 2), count, MPI_DOUBLE, 0, 3, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(z + (InputP.Nx + 2)*(InputP.Ny + 2), count, MPI_DOUBLE, 0, 4, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&offset, 1, MPI_INT, 0, 0, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(&count, 1, MPI_INT, 0, 1, MPI_COMM_WORLD));
        	MPI_CALL(MPI_Send(b + (InputP.Nx + 2)*(InputP.Ny + 2), count, MPI_DOUBLE, 0, 5, MPI_COMM_WORLD));
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

	if (rank == 0)
	{
		// Print CPU and GPU information
		uint32_t brand[12];
		if (!__get_cpuid_max(0x80000004, NULL)) {
			fprintf(stderr, "Feature not implemented.");
			return 2;
		}
		__get_cpuid(0x80000002, brand+0x0, brand+0x1, brand+0x2, brand+0x3);
		__get_cpuid(0x80000003, brand+0x4, brand+0x5, brand+0x6, brand+0x7);
		__get_cpuid(0x80000004, brand+0x8, brand+0x9, brand+0xa, brand+0xb);
		printf("CPU: %s\n", brand);
	}

	Variable Var;
	Allocate(Var, InputP, local_Nz);

	// Create a random number generator and set the seed
	std::mt19937_64 rng(1234ULL);  // Mersenne Twister 64-bit

	// Create distributions
	std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

	// Generate n doubles on CPU
	for (int k = 1; k < local_Nz + 1; k++) {
		for (int j = 1; j < InputP.Ny + 1; j++) {
			for (int i = 1; i < InputP.Nx + 1; i++) {
				int index = i + j * (InputP.Nx + 2) + k * (InputP.Nx + 2) * (InputP.Ny + 2);
				Var.x_old[index] = uniform_dist(rng);
				Var.y_old[index] = uniform_dist(rng);
				Var.z_old[index] = uniform_dist(rng);
			}
		}
	}

	Initialize(Var.x_old, Var.y_old, Var.z_old, Var.b, Var.phi_old, InputP, local_Nz);
	 
	Communicate_MPI_boundary(Var.x_old, InputP, rank, size, local_Nz);
	Communicate_MPI_boundary(Var.y_old, InputP, rank, size, local_Nz);
	Communicate_MPI_boundary(Var.z_old, InputP, rank, size, local_Nz);
	Set_boundary(Var.x_old, InputP, local_Nz);
	Set_boundary(Var.y_old, InputP, local_Nz);
	Set_boundary(Var.z_old, InputP, local_Nz);

	MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
	
	start_output_time = clock();
	if(InputP.output_mode > 1)
	{
		write_output_vtk(Var.x_old, Var.y_old, Var.z_old, Var.b, Var.phi_old, 0, InputP, rank, size, local_Nz, z_start);
		Integral(Var.x_old, Var.y_old, Var.z_old, Var.b, 0, InputP, rank, size, local_Nz, z_start);
	}
	end_output_time = clock();
	output_time_total += (float)(end_output_time - start_output_time);

	/*-------------------For performance test-------------------*/
	time_t t_start_loop, t_end_loop, sim_time_start, sim_time_end;
	t_start_loop = clock();
	sim_time_start = clock();

	int t = 1;
	while (t <= InputP.t_total) {
			
		Calc_mu(Var.x_old, Var.y_old, Var.z_old, Var.b, Var.phi_old, \
				Var.dfdx, Var.dfdy, Var.dfdz, InputP, local_Nz);
			
		//Communicate dfdx, dfdy, dfdz before lap of them	
		Communicate_MPI_boundary(Var.dfdx, InputP, rank, size, local_Nz);
		Communicate_MPI_boundary(Var.dfdy, InputP, rank, size, local_Nz);
		Communicate_MPI_boundary(Var.dfdz, InputP, rank, size, local_Nz);
		Set_boundary(Var.dfdx, InputP, local_Nz);
		Set_boundary(Var.dfdy, InputP, local_Nz);
		Set_boundary(Var.dfdz, InputP, local_Nz);
			
		MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));

		Update(Var.x_new, Var.y_new, Var.z_new, Var.phi_new, \
				Var.x_old, Var.y_old, Var.z_old, Var.phi_old, Var.b, \
				Var.dfdx, Var.dfdy, Var.dfdz, InputP, local_Nz);
			
		Swap_regular(Var.x_old, Var.x_new, \
				Var.y_old, Var.y_new, \
				Var.z_old, Var.z_new, \
				InputP, local_Nz);
			
		//Communicate x, y, z after updated
		Communicate_MPI_boundary(Var.x_old, InputP, rank, size, local_Nz);
		Communicate_MPI_boundary(Var.y_old, InputP, rank, size, local_Nz);
		Communicate_MPI_boundary(Var.z_old, InputP, rank, size, local_Nz);
		Set_boundary(Var.x_old, InputP, local_Nz);
		Set_boundary(Var.y_old, InputP, local_Nz);
		Set_boundary(Var.z_old, InputP, local_Nz);
			
		MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
			
		//Calculate the real-time values
		int RT_freq = InputP.t_freq;//1000;		//Frequency of calculation
		start_output_time = clock();
		if(InputP.output_mode)
		{
			if (t % RT_freq == 0)
			{
				if(rank == 0)
				{
					std::cout << "Timestep " << t << std::endl;
					sim_time_end = clock();
					time_t sim_time = sim_time_end - sim_time_start;
					Calc_RealTime_Values(InputP, t, RT_freq, sim_time);
					sim_time_start = clock();
				}
	
				if(InputP.output_mode > 1)
				{
					Integral(Var.x_old, Var.y_old, Var.z_old, Var.b, t, InputP, rank, size, local_Nz, z_start);
					// write .vtk files
					write_output_vtk(Var.x_old, Var.y_old, Var.z_old, Var.b, Var.phi_old, t, InputP, rank, size, local_Nz, z_start);
				}
			}
		}
		end_output_time = clock();
		output_time_total += (float)(end_output_time - start_output_time);
		t++;
	}

	FreeMemory(Var);

	/*-------------------For performance test-------------------*/
	t_end_loop = clock();
	t_end_tot = clock();

	MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
	if(rank == 0)
	{
		printf("\nPefromance test for Regular mode:\n");
		printf("The overall running time is: %f sec.\n", ((float)(t_end_tot - t_start_tot)) / CLOCKS_PER_SEC);
		printf("The loop running time is: %f sec. %3f percent of overall running time.\n", ((float)(t_end_loop - t_start_loop)) / CLOCKS_PER_SEC, (float)(t_end_loop - t_start_loop) / (double)(t_end_tot - t_start_tot) * 100.);
		printf("The overall running time (subtract output time) is: %f sec.\n", (((float)(t_end_tot - t_start_tot)) - output_time_total) / CLOCKS_PER_SEC);
	}
	MPI_CALL(MPI_Finalize());

	return 0;
}
