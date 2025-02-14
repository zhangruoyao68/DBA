# Define the NVCC compiler
NVCC = nvcc

# Specify the compute capability of your GPU
# ARCH = -arch=sm_86 # RTX 3060
#ARCH = -arch=sm_89 # RTX 4070
ARCH = -arch=sm_80 # A100

# Define compiler flags
NVCCFLAGS = -Xcompiler -Wall -Xcompiler -Wextra -use_fast_math -m64 $(ARCH)

# Define the target
TARGET = KH_DBA

# Rule to build the target
all: $(TARGET)

$(TARGET): KH_DBA.cu
	$(NVCC) $(NVCCFLAGS) KH_DBA.cu -o $(TARGET)

# Rule to clean the build
clean:
	rm -f $(TARGET)
