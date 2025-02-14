NP ?= 1
NVCC=nvcc
MPICXX=mpicxx
MPIRUN ?= mpirun
CUDA_HOME ?= /usr/local/cuda
GENCODE_SM30	:= -gencode arch=compute_30,code=sm_30
GENCODE_SM35	:= -gencode arch=compute_35,code=sm_35
GENCODE_SM37	:= -gencode arch=compute_37,code=sm_37
GENCODE_SM50	:= -gencode arch=compute_50,code=sm_50
GENCODE_SM52	:= -gencode arch=compute_52,code=sm_52
GENCODE_SM60    := -gencode arch=compute_60,code=sm_60
GENCODE_SM70    := -gencode arch=compute_70,code=sm_70# -gencode arch=compute_70,code=compute_70
#GENCODE_FLAGS	:= $(GENCODE_SM60) $(GENCODE_SM70)
GENCODE_SM80    := -gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80
GENCODE_FLAGS	:= $(GENCODE_SM70) $(GENCODE_SM80)

ifdef DISABLE_CUB
        NVCC_FLAGS = -Xptxas --optimize-float-atomics
else
        NVCC_FLAGS = -DHAVE_CUB
endif
ifdef SKIP_CUDA_AWARENESS_CHECK
        MPICXX_FLAGS = -DSKIP_CUDA_AWARENESS_CHECK
endif

NVCC_FLAGS += -lineinfo $(GENCODE_FLAGS) -std=c++14
MPICXX_FLAGS += -DUSE_NVTX -I$(CUDA_HOME)/include -std=c++14
LD_FLAGS = -L$(CUDA_HOME)/lib64 -lcudart -lnvToolsExt -lcurand

dendrite: Makefile dendrite.cpp dendrite_kernels.o
	$(MPICXX) $(MPICXX_FLAGS) dendrite.cpp dendrite_kernels.o $(LD_FLAGS) -o dendrite

dendrite_kernels.o: Makefile dendrite_kernels.cu
	$(NVCC) $(NVCC_FLAGS) dendrite_kernels.cu -c

.PHONY.: clean
clean:
	rm -f dendrite dendrite.o dendrite_kernels.o *.qdrep dendrite.*.compute-sanitizer.log

sanitize: dendrite
	$(MPIRUN) -np $(NP) compute-sanitizer --log-file dendrite.%q{OMPI_COMM_WORLD_RANK}.compute-sanitizer.log ./dendrite -niter 10

run: dendrite
	$(MPIRUN) -np $(NP) ./dendrite

profile: dendrite
	$(MPIRUN) -np $(NP) nsys profile --trace=mpi,cuda,nvtx -o dendrite.%q{OMPI_COMM_WORLD_RANK} ./dendrite -niter 10