# Define the NVCC compiler
NVCC = nvcc

# Specify the compute capability of your GPU
#ARCH = -arch=sm_89 # RTX 4070
ARCH = -arch=sm_80 # A100

# Define compiler flags
NVCCFLAGS = -Xcompiler -Wall -Xcompiler -Wextra -use_fast_math -lcurand -m64 $(ARCH)

# Define the target
TARGET = CH

# Rule to build the target
all: $(TARGET)

$(TARGET): kernel.cu
	$(NVCC) $(NVCCFLAGS) kernel.cu -o $(TARGET)

# Rule to clean the build
clean:
	rm -f $(TARGET)