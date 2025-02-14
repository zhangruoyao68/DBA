/* code to read 3d field and output size distributions and average radius*/
#include <iostream>
#include <fstream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <vector>
#include <limits>

#define L 256
#define NX L
#define NY L
#define NZ L

using namespace std;

void readArrayFromFile(const string& filename, double z[][NY][NZ])
{
    ifstream file(filename);
    if (!file)
    {
        cout << "Error opening the file." << endl;
        return;
    }

    // Skip the first 20 lines
    for (int i = 0; i < 20; i++)
    {
        file.ignore(numeric_limits<streamsize>::max(), '\n');
    }

    // Read the values from the 21st line
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                file >> z[i][j][k];
                //cout << y[i][j][k];
            }
        }
    }
    file.close();
}

void eraseIslands_BMC_new(double z[][NY][NZ], int i, int j, int k, double avg, unsigned int* total_volume) {
    if (i < 0 || i == NX || j < 0 || j == NY || k < 0 || k == NZ || z[i][j][k] < avg) {
        return;
    }

    // assign 0 to visited grid
    z[i][j][k] = 0;

    total_volume[0] += 1;
    
    eraseIslands_BMC_new(z, i - 1, j, k, avg, total_volume);
    eraseIslands_BMC_new(z, i + 1, j, k, avg, total_volume);
    eraseIslands_BMC_new(z, i, j - 1, k, avg, total_volume);
    eraseIslands_BMC_new(z, i, j + 1, k, avg, total_volume);
    eraseIslands_BMC_new(z, i, j, k - 1, avg, total_volume);
    eraseIslands_BMC_new(z, i, j, k + 1, avg, total_volume);

    // periodic boundary
    if (i == 0){
        eraseIslands_BMC_new(z, NX - 1, j, k, avg, total_volume);
    }
    if (i == NX-1){
        eraseIslands_BMC_new(z, 0, j, k, avg, total_volume);
    }
    if (j == 0){
        eraseIslands_BMC_new(z, i, NY - 1, k, avg, total_volume);
    }
    if (j == NY-1){
        eraseIslands_BMC_new(z, i, 0, k, avg, total_volume);
    }
    if (k == 0){
        eraseIslands_BMC_new(z, i, j, NZ - 1, avg, total_volume);
    }
    if (k == NZ-1){
        eraseIslands_BMC_new(z, i, j, 0, avg, total_volume);
    }
}

int numIslands_BMC_new(double z[][NY][NZ], double avg) {

    int islands = 0; // count number of islands

    unsigned int * total_volume = new unsigned int[1]();
    unsigned int volume_before; // calculate size of each island
    unsigned int* island_size = new unsigned int[NX*NY](); 

    // Append size data to file
    std::ofstream out;
    out.open("size_dist.txt", std::ios::app);

    //cout << "DFS algorithm starts" << endl;
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                if (z[i][j][k] > avg) {

                    volume_before = total_volume[0];
                    eraseIslands_BMC_new(z, i, j, k, avg, total_volume);
                    island_size[islands] = total_volume[0] - volume_before;
                    islands++;
                }
            }
        }
    }

    // iterate over number of islands
    for(short i = 0; i < islands; i++){ 
        // volume of the island
        unsigned int volume = island_size[i];
        out << volume << " ";
    }

    out << endl;
    out.close();

    delete [] total_volume;
    delete [] island_size;

    return islands;
}

double average_value(double z[][NY][NZ]){
    double Max = 0.5;
    double Min = 0.5;

    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                if (z[i][j][k] > Max){
                    Max = z[i][j][k];
                }
                if (z[i][j][k] < Min){
                    Min = z[i][j][k];
                }
            }
        }
    }
    //std::cout << "Max = " << Max << std::endl;
    //std::cout << "Min = " << Min << std::endl;

    double avg = (Max+Min)/2.;

    return avg;
}

int droplet_volume(double z[][NY][NZ], double avg){
    int volume = 0;
    for (int i = 0; i < NX; i++) {
        for (int j = 0; j < NY; j++) {
            for (int k = 0; k < NZ; k++) {
                if (z[i][j][k] > avg){
                    volume++;
                }   
            }
        }
    }
    return volume;
}

int main(int argc, char* argv[])
{
    typedef double nRarray[NY][NZ];
    nRarray* z_host;
    
    if ((z_host = (nRarray*)malloc((NX * NY * NZ) * sizeof(double))) == 0) { fprintf(stderr, "malloc1 Fail \n"); return 1;}

    ofstream ofile_radius("avg_radius.txt");
    //ofstream ofile_volume("volume_frac.txt");
    ofstream ofile_volume("size_dist.txt");

    for (int i = 0; i<101;i++){
        readArrayFromFile("output_"+to_string(i*100000)+".vtk", z_host);
        cout << "Read output_" +to_string(i*100000)+ ".vtk success" << endl;
        double avg = average_value(z_host);
        //cout << "Z_avg = " << avg << endl;

        int volumeDroplets = droplet_volume(z_host, avg);
        //cout << "Total volume = " << volumeDroplets << endl;

        int numDroplets = numIslands_BMC_new(z_host, avg);
        //cout << "Number of islands = " << numDroplets << endl;

        double R_avg = cbrt(volumeDroplets*0.75/M_PI/numDroplets);
        //cout << "R_avg = " << R_avg << endl;

        ofile_radius << i << "," << R_avg << endl;
        //ofile_volume << i << "," << 1.0*volumeDroplets/NX/NY/NZ << endl;
    }
    
    cout << "Processing finished." << endl;

    free(z_host);
    ofile_radius.close();
    ofile_volume.close();

    return 0;
}