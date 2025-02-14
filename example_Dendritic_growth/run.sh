rm *.out *.vtk

nvcc -o 3D_dendritic_growth kernel.cu -std=c++11
./3D_dendritic_growth