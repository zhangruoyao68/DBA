# Define the NVCC compiler
CC = g++
STD = -std=c++11


# Define compiler flags
CCFLAGS = -O2 $(STD)

# Define the target
TARGET = base_CPU
SRC = base_CPU.cpp

# Rule to build the target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CCFLAGS) $(INCFLAGS) $< -o $(TARGET)

# Rule to clean the build
clean:
	rm -f $(TARGET)
