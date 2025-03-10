## DBA

A dynamic block activation (DBA) framework for continuum models on GPUs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14868458.svg)](https://doi.org/10.5281/zenodo.14868458)
![License](https://img.shields.io/github/license/zhangruoyao68/DBA)

## Structure of the current repository
All code for 3 example models used in the manuscript are included and organized as follows:

- example_MODEL: single GPU version with DBA
- example_MODEL/CUDA-MPI: multi-GPU version with DBA
- example_MODEL/CPU_version: serial CPU version with regular mode only
- example_MODEL/CPU_version/MPI_version: multi-CPU version with regular mode only
- example_MODEL/Conserved: single GPU version with conserved DBA

## System and library requirements
This code requires CUDA and a Linux operating system to be executed.

The operating system we have tested was the Springdale Linux 8 operating system. 

The MPI library we have tested the multi-CPU version on is:
`openmpi-5.0.6`

The version of libraries we have tested the multi-GPU version on are:
`cudatoolkit/12.4`, `nvhpc/24.5`, and `openmpi/cuda-12.4/nvhpc-24.5/4.1.6`

NVIDIA GPUs are required for running the program.

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


## Citations

If you use the code in this repository, please cite the following article:

```
@article{Zhang_DBA_2025,
  author = {Ruoyao Zhang and Yang Xia},
  title = {A dynamic block activation framework for continuum models},
  journal = {Nature Computational Science},
  year = {2025},
  doi = {10.1038/s43588-025-00780-2}
}
```

All the data we provided in the manuscript can be reproduced by this code. Users can modify the input file accordingly and run the code following the installation guide to obtain the data in .vtk format. To visualize the .vtk output data, ParaView is recommended.

## License

This project is licensed under the GNU General Public License v3 - see the [LICENSE](https://www.gnu.org/licenses/) for details.
