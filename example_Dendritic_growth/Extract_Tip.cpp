#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char** argv)
{
	if (argc < 2)
	{
		std::cout << "Error: please enter the filename of vtk files you want to use to extract tip shape." << std::endl;
		exit(1);
	}
	
	std::string* filenamein = new std::string[argc - 1];
	std::ifstream* inf = new std::ifstream[argc - 1];
	int** tip_x = new int*[argc - 1];
	int* tip_x_max = new int[argc - 1];
	const int dz = 40;
	
	for(int p = 0; p < argc - 1; p++)
	{
		filenamein[p] = argv[p + 1];
		tip_x[p] = new int[dz+1];	
	}
	std::ofstream outf;
	std::string filenameout("Tip_Profile.dat");
	
	for(int p = 0; p < argc - 1; p++)
	{
		inf[p].open(filenamein[p]);
		if (!inf[p].is_open())
		{
			std::cout << "Error: can not open" << filenamein[p] << "!! Exit!!!!!!!!" << std::endl;
			exit(2);
		}
	}
	outf.open(filenameout);
	if (!outf.is_open())
	{
		std::cout << "Error: can not open" << filenameout << "!! Exit!!!!!!!!" << std::endl;
		exit(2);
	}
	
	outf << "z ";
	for(int p = 0; p < argc - 1; p++)
		outf << filenamein[p] << " ";
	outf << std::endl;
	
	for(int p = 0; p < argc - 1; p++)
	{
		int Nx, Ny, Nz;
		std::string block;
		std::cout << "-------" << filenamein[p] << "-------" << std::endl;
		for (int i = 0; i < 4; i++)
		{
			std::getline(inf[p], block);
			std::cout << block << std::endl;
		}
		inf[p] >> block >> Nx >> Ny >> Nz;
		std::cout << block << " " << Nx << " " << Ny << " " << Nz << std::endl;
		for (int i = 0; i < 6; i++)
		{
			std::getline(inf[p], block);
			std::cout << block << std::endl;
		}
	
		int tip_loc;
		double phi;
		for(int z = 0; z < Nz; z++)
		{
			tip_loc = 0;
			for(int y = 0; y < Ny; y++)
			for(int x = 0; x < Nx; x++)
			{
				inf[p] >> phi;
				if(z >= Nz/2 && z <= Nz/2 + dz)
				{
					if(phi > 0.5)
					{
						tip_loc = std::max(tip_loc,x);
					}
				}
			}
			if(z >= Nz/2 && z <= Nz/2 + dz)
			{
				tip_x[p][z-Nz/2] = tip_loc;
			}
		}
	
		tip_x_max[p] = 0;
		for(int i = 0; i < dz + 1; i++)
			tip_x_max[p] = std::max(tip_x_max[p], tip_x[p][i]); 
		std::cout << " Tip at x = " << tip_x_max[p] << std::endl;
		
		inf[p].close();
	}
	
	for(int i = 0; i < dz + 1; i++)
	{
		outf << i << " ";
		for(int p = 0; p < argc - 1; p++)
			outf << tip_x_max[p] - tip_x[p][i] << " ";
		outf << std::endl;	
	}
	
	outf.close();
	
	for(int p = 0; p < argc - 1; p++)
		delete[] tip_x[p];
	delete[] tip_x_max;
	delete[] inf;
	delete[] filenamein;
	
	return 0;
}
