#include <iostream>
#include <fstream>
#include <string>

int main(int argc, char** argv)
{
	if (argc < 3 || argc > 4)
	{
		std::cout << "Error: please enter the filename of vtk files you want to compare\
		and the name of the output difference file (if wanted)." << std::endl;
		exit(1);
	}
	std::string filename1(argv[1]);
	std::string filename2(argv[2]);

	std::ifstream inf1, inf2;
	inf1.open(filename1);
	if (!inf1.is_open())
	{
		std::cout << "Error: can not open" << filename1 << "!! Exit!!!!!!!!" << std::endl;
		exit(2);
	}
	inf2.open(filename2);
	if (!inf2.is_open())
	{
		std::cout << "Error: can not open" << filename2 << "!! Exit!!!!!!!!" << std::endl;
		exit(2);
	}

	std::string block;
	for (int i = 0; i < 4; i++)
	{
		std::getline(inf1, block);
		std::getline(inf2, block);
		std::cout << block << std::endl;
	}
	int Nx1, Ny1, Nz1, Nx2, Ny2, Nz2;
	inf1 >> block >> Nx1 >> Ny1 >> Nz1;
	inf2 >> block >> Nx2 >> Ny2 >> Nz2;
	int N_total = Nx1 * Ny1 * Nz1;
	if (N_total != Nx2 * Ny2 * Nz2)
	{
		std::cout << "Error: number of elements in vtk files do not match, exit!!" << std::endl;
		exit(3);
	}
	std::cout << block << " " << Nx1 << " " << Ny1 << " " << Nz1 << std::endl;
	for (int i = 0; i < 6; i++)
	{
		std::getline(inf1, block);
		std::getline(inf2, block);
		std::cout << block << std::endl;
	}

	if (argc == 3)
	{
		std::cout << "Comparing " << filename1 << " and " << filename2 << std::endl;
		double diff = 0;
		int num = 0;
		for (double data1; inf1 >> data1; num++)
		{
			double data2;
			inf2 >> data2;
			diff += std::abs(data1 - data2);
		}
		if (num != N_total)
		{
			std::cout << "Error: number of elements compared does not match to the number indicated by the vtk files , exit!!\n"
				<< "num from vtk = " << N_total << " while num compared = " << num << std::endl;
			std::cout << diff << std::endl;
			exit(3);
		}
		std::cout << num << " elements have been compared, the total difference is: " << diff
			<< ", while the average difference is: " << diff / num << std::endl;
	}

	if (argc == 4)
	{
		std::cout << "Comparing " << filename1 << " and " << filename2 << std::endl;
		std::string filenameout(argv[3]);
		std::ofstream outf;
		std::cout << "Difference will be saved in " << filenameout << std::endl;
		outf.open(filenameout);
		if (!outf.is_open())
		{
			std::cout << "Error: can not open" << filenameout << "!! Exit!!!!!!!!" << std::endl;
			exit(2);
		}

		outf << "# vtk DataFile Version 2.0" << std::endl;
		outf << "diff" << std::endl << "ASCII" << std::endl << "DATASET STRUCTURED_POINTS" << std::endl;
		outf << "DIMENSIONS " << Nx1 << " " << Ny1 << " " << Nz1 << std::endl;
		outf << "ASPECT_RATIO 1 1 1" << std::endl << "ORIGIN 0 0 0" << std::endl << "POINT_DATA " << N_total << std::endl;
		outf << "SCALARS diff double" << std::endl << "LOOKUP_TABLE default" << std::endl;

		double diff = 0;
		int num = 0;
		for (double data1; inf1 >> data1; num++)
		{
			double data2;
			inf2 >> data2;
			outf << data1 - data2 << " ";
			diff += std::abs(data1 - data2);
		}
		std::cout << num << " elements have been compared, the total difference is: " << diff
			<< ", while the average difference is: " << diff / num << std::endl;
	}

	return 0;
}