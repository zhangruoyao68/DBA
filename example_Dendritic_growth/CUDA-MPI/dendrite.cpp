/* 
 * Dendritic growth -- CUDA-Aware MPI version
 * 
 * 10/28/2024
 * Ruoyao Zhang
 * Princeton Univeristy
 */

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <iomanip>
#include <cstdlib>

#include <mpi.h>

using namespace std;

#define BLKXSIZE 4
#define BLKYSIZE 4
#define BLKZSIZE 4
#define MPI_CALL(call)                                                                \
    {                                                                                 \
        int mpi_status = call;                                                        \
        if (0 != mpi_status) {                                                        \
            char mpi_error_string[MPI_MAX_ERROR_STRING];                              \
            int mpi_error_string_length = 0;                                          \
            MPI_Error_string(mpi_status, mpi_error_string, &mpi_error_string_length); \
            if (NULL != mpi_error_string)                                             \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %s "                                                    \
                        "(%d).\n",                                                    \
                        #call, __LINE__, __FILE__, mpi_error_string, mpi_status);     \
            else                                                                      \
                fprintf(stderr,                                                       \
                        "ERROR: MPI call \"%s\" in line %d of file %s failed "        \
                        "with %d.\n",                                                 \
                        #call, __LINE__, __FILE__, mpi_status);                       \
        }                                                                             \
    }

#include <cuda_runtime.h>
#include <curand.h>

#ifdef USE_NVTX
#include <nvToolsExt.h>

const uint32_t colors[] = {0x0000ff00, 0x000000ff, 0x00ffff00, 0x00ff00ff,
                           0x0000ffff, 0x00ff0000, 0x00ffffff};
const int num_colors = sizeof(colors) / sizeof(uint32_t);

#define PUSH_RANGE(name, cid)                              \
    {                                                      \
        int color_id = cid;                                \
        color_id = color_id % num_colors;                  \
        nvtxEventAttributes_t eventAttrib = {0};           \
        eventAttrib.version = NVTX_VERSION;                \
        eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;  \
        eventAttrib.colorType = NVTX_COLOR_ARGB;           \
        eventAttrib.color = colors[color_id];              \
        eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII; \
        eventAttrib.message.ascii = name;                  \
        nvtxRangePushEx(&eventAttrib);                     \
    }
#define POP_RANGE nvtxRangePop();
#else
#define PUSH_RANGE(name, cid)
#define POP_RANGE
#endif

#define CUDA_RT_CALL(call)                                                                  \
    {                                                                                       \
        cudaError_t cudaStatus = call;                                                      \
        if (cudaSuccess != cudaStatus)                                                      \
            fprintf(stderr,                                                                 \
                    "ERROR: CUDA RT call \"%s\" in line %d of file %s failed "              \
                    "with "                                                                 \
                    "%s (%d).\n",                                                           \
                    #call, __LINE__, __FILE__, cudaGetErrorString(cudaStatus), cudaStatus); \
    }

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

//#ifdef USE_DOUBLE
typedef double real;
#define MPI_REAL_TYPE MPI_DOUBLE
//#else
//typedef float real;
//#define MPI_REAL_TYPE MPI_FLOAT
//#endif

// Device functions
void launch_initialize(real* __restrict__ const phi, real* __restrict__ const T, \
                    real* __restrict__ const init_rand,\
                    const int offset, const int nx, const int my_ny, const int ny, const int nz);

void launch_initialize_DBA(int *__restrict__ const wakeup_phi, int *__restrict__ const wakeup_phi_next, \
                        int* __restrict__ const wakeup_T, int* __restrict__ const wakeup_T_next, \
                        real* __restrict__ const phi, real* __restrict__ const T, \
                        real *__restrict__ const init_rand, \
                        const int offset, const int nx, const int my_ny, const int ny, const int nz);

void launch_phi_kernel(real *__restrict__ const phi, real *__restrict__ const T, \
                    real *__restrict__ const phi_new, \
                    const int iy_start, const int iy_end, const int nx, const int nz, const real dx, \
                    cudaStream_t stream);

void launch_phi_DBA_kernel(int *__restrict__ const wakeup, \
                        real *__restrict__ const phi, real *__restrict__ const T, \
                        real *__restrict__ const phi_new, real *__restrict__ const metric, \
                        const int iy_start, const int iy_end, const int nx, const int nz, const real dx, \
                        cudaStream_t stream);

void launch_T_kernel(real *__restrict__ const T, real *__restrict__ const T_new, \
                    real* __restrict__ const phi, real* __restrict__ const phi_new, \
                    real *__restrict__ const sum_c, \
                    const int iy_start, const int iy_end, const int nx, const int nz, const real dx, \
                    const bool calculate_sum, cudaStream_t stream);

void launch_T_DBA_kernel(int *__restrict__ const wakeup_T, int *__restrict__ const wakeup_phi, \
                        real *__restrict__ const T, real *__restrict__ const T_new, \
                        real* __restrict__ const phi, real* __restrict__ const phi_new, \
                        real* __restrict__ const metric_T, \
                        real *__restrict__ const sum_c, int *__restrict__ const sum_wakeup, \
                        const int iy_start, const int iy_end, const int nx, const int nz, const real dx, \
                        const bool calculate_sum, cudaStream_t stream);

void launch_wakeup_DBA_kernel(real* __restrict__ const phi_new, real* __restrict__ const T_new, \
                            int *__restrict__ const wakeup_phi, int *__restrict__ const wakeup_phi_next, \
                            int *__restrict__ const wakeup_T, int *__restrict__ const wakeup_T_next, \
                            real *__restrict__ const metric_phi, real *__restrict__ const metric_T, \
                            const int my_ny, const int nx, const int nz, cudaStream_t stream);

void launch_neighbor_DBA_kernel(int *__restrict__ const wakeup_phi, int *__restrict__ const wakeup_phi_next, \
                            int *__restrict__ const wakeup_T, int *__restrict__ const wakeup_T_next, \
                            const int my_ny, const int nx, const int nz, cudaStream_t stream);

// Host functions
void integral_c(real* __restrict__ const c, int nx, int iy_start, int iy_end, int nz);

void integral_wakeup(int* __restrict__ const wakeup, int nx, int iy_start, int iy_end, int nz);

void write_output_vtr(real *__restrict__ const phi, real *__restrict__ const T, \
                    int *__restrict__ const wakeup_phi, int *__restrict__ const wakeup_T, \
                    int t, int local_rank, int nx, int iy_start, int iy_end, int nz);
void write_output_pvtr(const int total_rank, const int t, const int nx, const int ny, const int nz);

template <typename T>
T get_argval(char** begin, char** end, const string& arg, const T default_val) {
    T argval = default_val;
    char** itr = find(begin, end, arg);
    if (itr != end && ++itr != end) {
        istringstream inbuf(*itr);
        inbuf >> argval;
    }
    return argval;
}

bool get_arg(char** begin, char** end, const string& arg) {
    char** itr = find(begin, end, arg);
    if (itr != end) {
        return true;
    }
    return false;
}

int main(int argc, char* argv[]) {

    // Initialize MPI library
    MPI_CALL(MPI_Init(&argc, &argv));

    // Determine the calling process rank and total number of ranks
    int rank;
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
    int size;
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &size));

    // Program inputs and default values
    const int iter_max = get_argval<int>(argv, argv + argc, "-niter", 1000);
    const int nccheck = get_argval<int>(argv, argv + argc, "-nccheck", 1000);
    const int nx = get_argval<int>(argv, argv + argc, "-nx", 16384);
    const int ny = get_argval<int>(argv, argv + argc, "-ny", 16384);
    const int nz = get_argval<int>(argv, argv + argc, "-nz", 16384);
    const real c_init = get_argval<real>(argv, argv + argc, "-c_init", -0.55);
    const real dt = get_argval<real>(argv, argv + argc, "-dt", 0.00005);
    const real dx = get_argval<real>(argv, argv + argc, "-dx", 0.02);
    const real Delta = get_argval<real>(argv, argv + argc, "-Delta", 0.25);
    const real epsilon_phi = get_argval<real>(argv, argv + argc, "-epsilon_phi", 0.01);
    const real epsilon_T = get_argval<real>(argv, argv + argc, "-epsilon_T", 1e-5);
    const bool vtk = get_arg(argv, argv + argc, "-vtk");

    // Handling multiple multi-GPU nodes
    // calculating local rank for each device
    int local_rank = -1;
    {
        MPI_Comm local_comm;
        MPI_CALL(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL,
                                     &local_comm));

        MPI_CALL(MPI_Comm_rank(local_comm, &local_rank));

        MPI_CALL(MPI_Comm_free(&local_comm));
    }

    // setting it to global rank can sometimes cause problem 
    //CUDA_RT_CALL(cudaSetDevice(local_rank));
    CUDA_RT_CALL(cudaSetDevice(0));
    CUDA_RT_CALL(cudaFree(0));

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Set up multiple multi-gpu calculation
    /////////////////////////////////////////////////////////////////////////////////////////////////

    // Calculate chunk_size

    // ny - 2 rows are distributed amongst `size` ranks in such a way
    // that each rank gets either (ny - 2) / size or (ny - 2) / size + 1 rows.
    // This optimizes load balancing when (ny - 2) % size != 0
    int chunk_size;
    int chunk_size_low = (ny - 2) / size;
    int chunk_size_high = chunk_size_low + 1;

    // To calculate the number of ranks that need to compute an extra row,
    // the following formula is derived from this equation:
    // num_ranks_low * chunk_size_low + (size - num_ranks_low) * (chunk_size_low + 1) = ny - 2
    int num_ranks_low = size * chunk_size_low + size - \
                        (ny - 2);  // Number of ranks with chunk_size = chunk_size_low
    if (rank < num_ranks_low)
        chunk_size = chunk_size_low;
    else
        chunk_size = chunk_size_high;

    // Calculate local domain boundaries
    int iy_start_global;  // My start index in the global array
    if (rank < num_ranks_low) {
        iy_start_global = rank * chunk_size_low + 1;
    } else {
        iy_start_global =
            num_ranks_low * chunk_size_low + (rank - num_ranks_low) * chunk_size_high + 1;
    }
    int iy_end_global = iy_start_global + chunk_size - 1;  // My last index in the global array
    
    int iy_start = 1;
    int iy_end = iy_start + chunk_size;

    // ********Override chunk_size for testing ********
    //chunk_size = ny / size;
    

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Allocate host memory
    /////////////////////////////////////////////////////////////////////////////////////////////////
    size_t total_size_count = nx * (chunk_size + 2) * nz;
    size_t total_size = total_size_count * sizeof(real);
    //cout << "size_t total_size = " << total_size << endl;
    //cout << "chuck_size = " << chunk_size << endl;
    real* phi_h;
    real* T_h;
    CUDA_RT_CALL(cudaMallocHost(&phi_h, total_size));
    CUDA_RT_CALL(cudaMallocHost(&T_h, total_size));

    int num_block_x = (nx + BLKXSIZE - 1) / BLKXSIZE;
    int num_block_y = (chunk_size + 2 + BLKYSIZE - 1) / BLKYSIZE;
    //int num_block_y = (chunk_size + BLKYSIZE - 1) / BLKYSIZE; // testing
    int num_block_z = (nz + BLKZSIZE - 1) / BLKZSIZE;
    int total_num_blocks = num_block_x * num_block_y * num_block_z;
    // Note: we could also use a fixed number of blocks like in the single DBA version
    cout << "total_num_blocks = " << total_num_blocks << endl;
    int* wakeup_phi_h;
    int* wakeup_T_h;
    CUDA_RT_CALL(cudaMallocHost(&wakeup_phi_h, total_num_blocks * sizeof(int)));
    CUDA_RT_CALL(cudaMallocHost(&wakeup_T_h, total_num_blocks * sizeof(int)));

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Allocate device memory
    /////////////////////////////////////////////////////////////////////////////////////////////////

    real* phi;
    real* phi_new;
    real* T;
    real* T_new;
    CUDA_RT_CALL(cudaMalloc(&phi, total_size));
    CUDA_RT_CALL(cudaMalloc(&phi_new, total_size));
    CUDA_RT_CALL(cudaMalloc(&T, total_size));
    CUDA_RT_CALL(cudaMalloc(&T_new, total_size));

    real* init_rand; // store random values for initial conditions
    CUDA_RT_CALL(cudaMalloc(&init_rand, total_size));

    real* metric_phi; // store metric values for activation
    real* metric_T;
    CUDA_RT_CALL(cudaMalloc(&metric_phi, total_size));
    CUDA_RT_CALL(cudaMalloc(&metric_T, total_size));

    int* wakeup_phi;
    int* wakeup_phi_next;
    int* wakeup_T;
    int* wakeup_T_next;
    CUDA_RT_CALL(cudaMalloc(&wakeup_phi, total_num_blocks * sizeof(int)));
    CUDA_RT_CALL(cudaMalloc(&wakeup_phi_next, total_num_blocks * sizeof(int)));
    CUDA_RT_CALL(cudaMalloc(&wakeup_T, total_num_blocks * sizeof(int)));
    CUDA_RT_CALL(cudaMalloc(&wakeup_T_next, total_num_blocks * sizeof(int)));


    cout << "Host memory allocated on CPU rank " << rank << ": " \
             << 2 * total_size/(1024.*1024.*1024.) << " Gigabytes" << endl;
    
    cout << "Device memory allocated on GPU rank " << rank << ": " \
             << 7 * total_size/(1024.*1024.*1024.) << " Gigabytes\n" << endl;


    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Set initial conditions
    /////////////////////////////////////////////////////////////////////////////////////////////////
    
    curandGenerator_t gen;
    // Create pseudo-random number generator
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));

    // Set seed
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(0)+rank)); // default seed 1234ULL

    // Generate nx * (chunk_size + 2) * nz doubles on device for initialization
    //CURAND_CALL(curandGenerateUniformDouble(gen, init_rand, nx * (chunk_size + 2) * nz));
    
    //launch_initialize(phi, T, init_rand, iy_start_global - 1, nx, (chunk_size + 2), ny, nz);
    launch_initialize_DBA(wakeup_phi, wakeup_phi_next, wakeup_T, wakeup_T_next, phi, T, \
                        init_rand, iy_start_global - 1, nx, (chunk_size + 2), ny, nz);

    CUDA_RT_CALL(cudaDeviceSynchronize());

    cout << "Initialization on device " << rank << " is done\n" << endl;

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Set up Streams and Events
    /////////////////////////////////////////////////////////////////////////////////////////////////
    int leastPriority = 0;
    int greatestPriority = leastPriority;
    CUDA_RT_CALL(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    cudaStream_t compute_stream;
    CUDA_RT_CALL(cudaStreamCreateWithPriority(&compute_stream, cudaStreamDefault, leastPriority));
    cudaStream_t push_top_stream;
    CUDA_RT_CALL(
        cudaStreamCreateWithPriority(&push_top_stream, cudaStreamDefault, greatestPriority));
    cudaStream_t push_bottom_stream;
    CUDA_RT_CALL(
        cudaStreamCreateWithPriority(&push_bottom_stream, cudaStreamDefault, greatestPriority));

    cudaEvent_t compute_done;
    CUDA_RT_CALL(cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming));
    cudaEvent_t push_top_done;
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_top_done, cudaEventDisableTiming));
    cudaEvent_t push_bottom_done;
    CUDA_RT_CALL(cudaEventCreateWithFlags(&push_bottom_done, cudaEventDisableTiming));
    cudaEvent_t reset_sum_done;
    CUDA_RT_CALL(cudaEventCreateWithFlags(&reset_sum_done, cudaEventDisableTiming));

    // calculate sum of field
    real* sum_c_d;
    CUDA_RT_CALL(cudaMalloc(&sum_c_d, sizeof(real)));
    real* sum_c_h;
    CUDA_RT_CALL(cudaMallocHost(&sum_c_h, sizeof(real)));
    int* sum_wakeup_d;
    CUDA_RT_CALL(cudaMalloc(&sum_wakeup_d, sizeof(int)));
    int* sum_wakeup_h;
    CUDA_RT_CALL(cudaMallocHost(&sum_wakeup_h, sizeof(int)));


    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Fill device boundary and MPI warm-up
    /////////////////////////////////////////////////////////////////////////////////////////////////
    const int top = rank > 0 ? rank - 1 : (size - 1);
    const int bottom = (rank + 1) % size;

    PUSH_RANGE("Fill boundary", 5)

    CUDA_RT_CALL(cudaStreamSynchronize(push_top_stream));

    // Boundary for all fields
    MPI_CALL(MPI_Sendrecv(phi + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0, \
                        phi + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD, \
                        MPI_STATUS_IGNORE));
    MPI_CALL(MPI_Sendrecv(T + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0, \
                        T + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD, \
                        MPI_STATUS_IGNORE));
    CUDA_RT_CALL(cudaStreamSynchronize(push_bottom_stream));

    MPI_CALL(MPI_Sendrecv(phi + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, phi, nx * nz, \
                        MPI_REAL_TYPE, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    MPI_CALL(MPI_Sendrecv(T + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, T, nx * nz, \
                        MPI_REAL_TYPE, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    
    POP_RANGE

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (0 == rank) {
        printf(
            "Simulation: %d iterations on %d x %d x %d mesh with sum check " \
            "every %d iterations\n", \
            iter_max, nx, ny, nz, nccheck);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Create files for writing data
    /////////////////////////////////////////////////////////////////////////////////////////////////

    int iter = 1;
    bool calculate_sum;
    real sum_c = 0.0;
    int sum_wakeup = 0;

    
    // prevent memory overflow when calculating sendback_size
    size_t nx_s = (size_t)nx;
    size_t nz_s = (size_t)nz;
    size_t ny_part = (size_t)(ny - iy_start_global);
    size_t sendback_size = std::min(ny_part * nx_s * nz_s, chunk_size * nx_s * nz_s) * sizeof(real);
    //size_t sendback_size = min((ny - iy_start_global) * (nx * nz), chunk_size * (nx * nz)) * sizeof(real);
    
    // Save initial configuration
    //Copy calculated data to CPU host on respective rank
    CUDA_RT_CALL(cudaMemcpy(phi_h + (nx * nz), phi + (nx * nz), \
                        sendback_size, \
                        cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaMemcpy(T_h + (nx * nz), T + (nx * nz), \
                        sendback_size, \
                        cudaMemcpyDeviceToHost));
    
    CUDA_RT_CALL(cudaMemcpy(wakeup_phi_h, wakeup_phi, \
                        total_num_blocks * sizeof(int), \
                        cudaMemcpyDeviceToHost));
    
    CUDA_RT_CALL(cudaMemcpy(wakeup_T_h, wakeup_T, \
                        total_num_blocks * sizeof(int), \
                        cudaMemcpyDeviceToHost));

    // Write initial condition files
    if (vtk) {
        write_output_vtr(phi_h, T_h, wakeup_phi_h, wakeup_T_h, 0, rank, nx, iy_start_global-1, iy_end_global+1, nz);
        if (rank == 0) {
            write_output_pvtr(size, 0, nx, ny, nz);
        }
    }

    /**/
    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Main loop
    /////////////////////////////////////////////////////////////////////////////////////////////////

    MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
    real start = MPI_Wtime();
    PUSH_RANGE("Evolve", 0)
    while (iter <= iter_max) {
        
        CUDA_RT_CALL(cudaMemsetAsync(sum_c_d, 0, sizeof(real), compute_stream));
        CUDA_RT_CALL(cudaMemsetAsync(sum_wakeup_d, 0, sizeof(int), compute_stream));
        CUDA_RT_CALL(cudaEventRecord(reset_sum_done, compute_stream));
        
        calculate_sum = (iter % nccheck) == 0;
        
        /////////////////////////////////////////////////////////////////////////////////////////////////
        // Calculate phi
        /////////////////////////////////////////////////////////////////////////////////////////////////

        // Calculate boundary regions
        CUDA_RT_CALL(cudaStreamWaitEvent(push_top_stream, reset_sum_done, 0));
        //launch_phi_kernel(phi, T, phi_new, iy_start, (iy_start + 1), nx, nz, dx, push_top_stream);
        launch_phi_DBA_kernel(wakeup_phi, phi, T, phi_new, metric_phi, \
                            iy_start, (iy_start + 1), nx, nz, dx, push_top_stream);
        CUDA_RT_CALL(cudaEventRecord(push_top_done, push_top_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(push_bottom_stream, reset_sum_done, 0));
        //launch_phi_kernel(phi, T, phi_new, (iy_end - 1), iy_end, nx, nz, dx, push_bottom_stream);
        launch_phi_DBA_kernel(wakeup_phi, phi, T, phi_new, metric_phi, \
                            (iy_end - 1), iy_end, nx, nz, dx, push_bottom_stream);
        CUDA_RT_CALL(cudaEventRecord(push_bottom_done, push_bottom_stream));

        // Calculate center region
        //launch_phi_kernel(phi, T, phi_new, (iy_start + 1), (iy_end - 1), nx, nz, dx, compute_stream);
        launch_phi_DBA_kernel(wakeup_phi, phi, T, phi_new, metric_phi, \
                            (iy_start + 1), (iy_end - 1), nx, nz, dx, compute_stream);
        CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));

        // Apply periodic boundary conditions
        CUDA_RT_CALL(cudaStreamSynchronize(push_top_stream));
        PUSH_RANGE("MPI", 5)
        MPI_CALL(MPI_Sendrecv(phi_new + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0, \
                            phi_new + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, \
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        
        CUDA_RT_CALL(cudaStreamSynchronize(push_bottom_stream));
        MPI_CALL(MPI_Sendrecv(phi_new + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, \
                            phi_new, nx * nz, MPI_REAL_TYPE, top, 0, \
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        POP_RANGE
        
        /////////////////////////////////////////////////////////////////////////////////////////////////
        // Calculate T
        /////////////////////////////////////////////////////////////////////////////////////////////////
        //CURAND_CALL(curandSetStream(gen, push_top_stream));
        // Generate nx * (chunk_size + 2) * nz doubles on device for initialization
        //CURAND_CALL(curandGenerateUniformDouble(gen, phi_rand, nx * (chunk_size + 2) * nz));

        // Calculate boundary regions
        CUDA_RT_CALL(cudaStreamWaitEvent(push_top_stream, compute_done, 0));
        //launch_T_kernel(T, T_new, phi, phi_new, sum_c_d, iy_start, (iy_start + 1), nx, nz, dx, \
                        calculate_sum, push_top_stream);
        launch_T_DBA_kernel(wakeup_T, wakeup_phi, T, T_new, phi, phi_new, metric_T, sum_c_d, sum_wakeup_d, \
                            iy_start, (iy_start + 1), nx, nz, dx, calculate_sum, push_top_stream);
        CUDA_RT_CALL(cudaEventRecord(push_top_done, push_top_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(push_bottom_stream, compute_done, 0));
        //launch_T_kernel(T, T_new, phi, phi_new, sum_c_d, (iy_end - 1), iy_end, nx, nz, dx, \
                        calculate_sum, push_bottom_stream);
        launch_T_DBA_kernel(wakeup_T, wakeup_phi, T, T_new, phi, phi_new, metric_T, sum_c_d, sum_wakeup_d, \
                            (iy_end - 1), iy_end, nx, nz, dx, calculate_sum, push_bottom_stream);
        CUDA_RT_CALL(cudaEventRecord(push_bottom_done, push_bottom_stream));

        // Calculate center region
        //launch_T_kernel(T, T_new, phi, phi_new, sum_c_d, (iy_start + 1), (iy_end - 1), nx, nz, dx, \
                        calculate_sum, compute_stream);
        launch_T_DBA_kernel(wakeup_T, wakeup_phi, T, T_new, phi, phi_new, metric_T, sum_c_d, sum_wakeup_d, \
                            (iy_start + 1), (iy_end - 1), nx, nz, dx, calculate_sum, compute_stream);

        if (calculate_sum) {
            CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_top_done, 0));
            CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_bottom_done, 0));
            CUDA_RT_CALL(cudaMemcpyAsync(sum_c_h, sum_c_d, sizeof(real), cudaMemcpyDeviceToHost, \
                                         compute_stream));
            CUDA_RT_CALL(cudaMemcpyAsync(sum_wakeup_h, sum_wakeup_d, sizeof(int), cudaMemcpyDeviceToHost, \
                                         compute_stream));
        }

        // Apply periodic boundary conditions
        CUDA_RT_CALL(cudaStreamSynchronize(push_top_stream));
        PUSH_RANGE("MPI", 5)

        MPI_CALL(MPI_Sendrecv(T_new + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0, \
                            T_new + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD, \
                            MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(metric_T + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0, \
                            metric_T + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD, \
                            MPI_STATUS_IGNORE));

        CUDA_RT_CALL(cudaStreamSynchronize(push_bottom_stream));

        MPI_CALL(MPI_Sendrecv(T_new + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, \
                            T_new, nx * nz, MPI_REAL_TYPE, top, 0, \
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(metric_T + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, \
                            metric_T, nx * nz, MPI_REAL_TYPE, top, 0, \
                            MPI_COMM_WORLD, MPI_STATUS_IGNORE));

        POP_RANGE

        /////////////////////////////////////////////////////////////////////////////////////////////////
        // Save configuration and output sum
        /////////////////////////////////////////////////////////////////////////////////////////////////
        if (calculate_sum) {
            CUDA_RT_CALL(cudaStreamSynchronize(compute_stream));
            MPI_CALL(MPI_Allreduce(sum_c_h, &sum_c, 1, MPI_REAL_TYPE, MPI_SUM, MPI_COMM_WORLD));
            //MPI_CALL(MPI_Allreduce(sum_wakeup_h, &sum_wakeup, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

            if (0 == rank) {
                printf("Iteration: %d, Sum: %8.4f, Wakeup: %d\n", iter, sum_c, sum_wakeup);
                fflush(stdout);
                //cout << "Iteration: " << iter << ", Sum: " << sum_c << ", Wakeup: " << sum_wakeup << endl;
            }
            
            // Copy calculated data to CPU host on respective rank
            CUDA_RT_CALL(cudaMemcpy(phi_h + (nx * nz), phi_new + (nx * nz), \
                                    sendback_size, cudaMemcpyDeviceToHost));
            CUDA_RT_CALL(cudaMemcpy(T_h + (nx * nz), T_new + (nx * nz), \
                                    sendback_size, cudaMemcpyDeviceToHost));
            CUDA_RT_CALL(cudaMemcpy(wakeup_phi_h, wakeup_phi, \
                        total_num_blocks * sizeof(int), \
                        cudaMemcpyDeviceToHost));
            CUDA_RT_CALL(cudaMemcpy(wakeup_T_h, wakeup_T, \
                        total_num_blocks * sizeof(int), \
                        cudaMemcpyDeviceToHost));

            // Write files on each CPU rank
            if (vtk){
                write_output_vtr(phi_h, T_h, wakeup_phi_h, wakeup_T_h, iter, rank, \
                                nx, iy_start_global-1, iy_end_global + 1, nz);
                if (rank == 0) {
                write_output_pvtr(size, iter, nx, ny, nz);
                }
            }
            
            MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
        }

        /////////////////////////////////////////////////////////////////////////////////////////////////
        // Update active blocks for next iteration
        /////////////////////////////////////////////////////////////////////////////////////////////////

        // evaluate metric for activation
        launch_wakeup_DBA_kernel(phi_new, T_new, wakeup_phi, wakeup_phi_next, \
                                wakeup_T, wakeup_T_next, metric_phi, metric_T, \
                                (chunk_size + 2), nx, nz, compute_stream);

        /*
        // Sync boundary blocks
        CUDA_RT_CALL(cudaStreamSynchronize(push_top_stream));
        PUSH_RANGE("MPI", 5)

        MPI_CALL(MPI_Sendrecv(wakeup_phi, layer_size, MPI_INT, top, 0, \
                        wakeup_phi_buffer_bottom, layer_size, MPI_INT, bottom, 0, MPI_COMM_WORLD, \
                        MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(wakeup_T, layer_size, MPI_INT, top, 0, \
                        wakeup_T_buffer_bottom, layer_size, MPI_INT, bottom, 0, MPI_COMM_WORLD, \
                        MPI_STATUS_IGNORE));
        //CUDA_RT_CALL(cudaEventRecord(push_top_done, push_top_stream));
        CUDA_RT_CALL(cudaStreamSynchronize(push_bottom_stream));

        MPI_CALL(MPI_Sendrecv(wakeup_phi + layer_offset, layer_size, MPI_INT, bottom, 0, \
                    wakeup_phi_buffer_top, layer_size, MPI_INT, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(wakeup_T + layer_offset, layer_size, MPI_INT, bottom, 0, \
                    wakeup_T_buffer_top, layer_size, MPI_INT, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

        POP_RANGE
        */
        // Activate neighbors
        //CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_top_done, 0));
        //CUDA_RT_CALL(cudaStreamWaitEvent(compute_stream, push_bottom_done, 0));
        launch_neighbor_DBA_kernel(wakeup_phi, wakeup_phi_next, wakeup_T, wakeup_T_next, \
                                (chunk_size + 2), nx, nz, compute_stream);
        
        // Swap new and old fields
        swap(phi_new, phi);
        swap(T_new, T);

        iter++;
    }
    real stop = MPI_Wtime();
    POP_RANGE
    
    if (rank == 0) {
        printf("\nNum GPUs: %d\n", size);
        printf(
            "Time taken: %8.3f s\n",
            (stop - start));
    }
    
    // Let all other processes wait for Rank 0 to finish writing VTK
    //MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));

    // Clean up after calculation
    CURAND_CALL(curandDestroyGenerator(gen));

    // Clean up events and streams
    CUDA_RT_CALL(cudaEventDestroy(reset_sum_done));
    CUDA_RT_CALL(cudaEventDestroy(push_bottom_done));
    CUDA_RT_CALL(cudaEventDestroy(push_top_done));
    CUDA_RT_CALL(cudaEventDestroy(compute_done));
    CUDA_RT_CALL(cudaStreamDestroy(push_bottom_stream));
    CUDA_RT_CALL(cudaStreamDestroy(push_top_stream));
    CUDA_RT_CALL(cudaStreamDestroy(compute_stream));

    // Free host memory
    CUDA_RT_CALL(cudaFreeHost(phi_h));
    CUDA_RT_CALL(cudaFreeHost(T_h));
    CUDA_RT_CALL(cudaFreeHost(wakeup_phi_h));
    CUDA_RT_CALL(cudaFreeHost(wakeup_T_h));
    
    // Free device memory
    CUDA_RT_CALL(cudaFree(phi));
    CUDA_RT_CALL(cudaFree(phi_new));
    CUDA_RT_CALL(cudaFree(T));
    CUDA_RT_CALL(cudaFree(T_new));
    CUDA_RT_CALL(cudaFree(wakeup_phi));
    CUDA_RT_CALL(cudaFree(wakeup_phi_next));
    CUDA_RT_CALL(cudaFree(wakeup_T));
    CUDA_RT_CALL(cudaFree(wakeup_T_next));

    CUDA_RT_CALL(cudaFree(init_rand));
    CUDA_RT_CALL(cudaFree(metric_phi));
    CUDA_RT_CALL(cudaFree(metric_T));

    CUDA_RT_CALL(cudaFreeHost(sum_c_h));
    CUDA_RT_CALL(cudaFree(sum_c_d));
    CUDA_RT_CALL(cudaFreeHost(sum_wakeup_h));
    CUDA_RT_CALL(cudaFree(sum_wakeup_d));

    MPI_CALL(MPI_Finalize());
    return 0;
}