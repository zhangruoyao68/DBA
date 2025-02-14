# Define the NVCC compiler
CC = mpicxx
STD = -std=c++11

# Define compiler flags
CCFLAGS = -O2 $(STD)
INCFLAGS = -I/usr/lib/x86_64-linux-gnu/openmpi/include/

# Define the target
TARGET = Base
SRC = Base.cpp

# Rule to build the target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CCFLAGS) $(INCFLAGS) $< -o $(TARGET)

# Rule to clean the build
clean:
	rm -f $(TARGET)
