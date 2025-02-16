## DBA

A dynamic block activation (DBA) framework for continuum models on GPUs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14868458.svg)](https://doi.org/10.5281/zenodo.14868458)
![License](https://img.shields.io/github/license/zhangruoyao68/DBA)

## System requirements
This code requires CUDA and a Linux operating system to be executed. The multi-CPU and multi-GPU version requires MPI library installed.

The versions of CUDA and operating system we have tested on are:
CUDA Toolkit 12.4 and Springdale Linux 8 operating system. 

The version of MPI library we have tested the multi-CPU version on is:
openmpi-5.0.6

The version of libraries we have tested the multi-GPU version on are:
cudatoolkit/12.4, nvhpc/24.5, and openmpi/cuda-12.4/nvhpc-24.5/4.1.6

NVIDIA GPUs are required for running the program.


## Structure of this folder
All codes for 3 example models used in the manuscript are included. 

- example_XXX: single GPU version with DBA framework of model XXX
- example_XXX/CUDA-MPI: multi-GPU version with DBA framework of model XXX
- example_XXX/CPU_version: serial CPU version with regular mode only of model XXX
- example_XXX/CPU_version/MPI_version: multi-CPU version with regular mode only of model XXX
- example_XXX/Conserved: single GPU version with conserved DBA framework of model XXX

## Installation guide
A bash script is provided under each sample folder. They can be compiled and executed by running:
```
./run.sh 
```
If the run.sh file is not an executable, please run the following command first:
```
chmod +x run.sh
```

The installation or compilation should be completed immediately, and the execution time will depend on the user's input file.

## Demo
A demo is provided in the Example_1_Dendritic_growth/Demo folder. This demo provides a standard input file to generate a small system of 3D dendritic growth with output files in .vtk format. The expected run time depends on the hardware system (especially GPU performance). Typically, it should be completed within 5 minutes on a laptop computer with an NVIDIA 4070 GPU.

## Instruction for use
All the data we provided in the manuscript can be reproduced by this code. Users can modify the input file accordingly and run the code following the installation guide to obtain the data in .vtk format. To visualize the .vtk output data, ParaView is recommended.
