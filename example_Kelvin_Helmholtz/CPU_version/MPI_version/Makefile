# Define the mpi compiler
CC = mpicxx
STD = -std=c++11

# Define compiler flags
CCFLAGS = -O2 $(STD)

# Define the target
TARGET = KH_CPU
SRC = KH_CPU.cpp

# Rule to build the target
all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CCFLAGS) $(INCFLAGS) $< -o $(TARGET)

# Rule to clean the build
clean:
	rm -f $(TARGET)
