/* 
 * Model B with chemical reactions -- CUDA-Aware MPI version
 * 
 * 11/28/2024
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
#include <cpuid.h>  // GCC-provided

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

//const real PI = 2.0 * asin(1.0);

// Device functions
void launch_initialize(real* __restrict__ const x, real* __restrict__ const y, \
                                real* __restrict__ const z, real* __restrict__ const b, \
                                const real x_init, const real y_init, const real z_init,\
                        const int offset, const int nx, const int my_ny, const int ny, const int nz);

void launch_initialize_DBA(int* __restrict__ const wakeup, int* __restrict__ const wakeup_next, \
                        real* __restrict__ const x, real* __restrict__ const y, \
                        real* __restrict__ const z, real* __restrict__ const b, \
                        const real x_init, const real y_init, const real z_init,\
                        const int offset, const int nx, const int my_ny, const int ny, const int nz);

void launch_mu_kernel(real* __restrict__ const x, real* __restrict__ const y, \
                    real* __restrict__ const z, real* __restrict__ const b, \
                    real* __restrict__ const dfdx, real* __restrict__ const dfdy, \
                    real* __restrict__ const dfdz, \
                    const int iy_start, const int iy_end, const int nx, const int nz, cudaStream_t stream);

void launch_mu_DBA_kernel(int* __restrict__ const wakeup, \
                    real* __restrict__ const x, real* __restrict__ const y, \
                    real* __restrict__ const z, real* __restrict__ const b, \
                    real* __restrict__ const dfdx, real* __restrict__ const dfdy, \
                    real* __restrict__ const dfdz, \
                    real* __restrict__ const metric,\
                    const int iy_start, const int iy_end, const int nx, const int nz, cudaStream_t stream);

void launch_update_kernel(real* __restrict__ const x_old, real* __restrict__ const y_old, \
                        real* __restrict__ const z_old, real* __restrict__ const b, \
                        real* __restrict__ const x_new, real* __restrict__ const y_new, \
                        real* __restrict__ const z_new, \
                        real* __restrict__ const dfdx, real* __restrict__ const dfdy, \
                        real* __restrict__ const dfdz, \
                        real* __restrict__ const sum_c, \
                        const int iy_start, const int iy_end, const int nx, const int nz, \
                        const bool calculate_sum, cudaStream_t stream);

void launch_update_DBA_kernel(int* __restrict__ const wakeup, \
                            real* __restrict__ const x_old, real* __restrict__ const y_old, \
                            real* __restrict__ const z_old, real* __restrict__ const b, \
                            real* __restrict__ const x_new, real* __restrict__ const y_new, \
                            real* __restrict__ const z_new, \
                            real* __restrict__ const dfdx, real* __restrict__ const dfdy, \
                            real* __restrict__ const dfdz, \
                            real* __restrict__ const sum_c, int* __restrict__ const sum_wakeup, \
                            const int iy_start, const int iy_end, const int nx, const int nz, \
                            const bool calculate_sum, cudaStream_t stream);

void launch_wake_DBA_kernel(int* __restrict__ const wakeup, int* __restrict__ const wakeup_next,\
                            real* __restrict__ const metric, \
                            const int nx, const int my_ny, const int nz, cudaStream_t stream);

// Host functions
void integral_c(real* __restrict__ const c, int nx, int iy_start, int iy_end, int nz);

void integral_wakeup(int* __restrict__ const wakeup, int nx, int iy_start, int iy_end, int nz);

void write_output_vtr(real* __restrict__ const x, real* __restrict__ const y, \
                    real* __restrict__ const z, real* __restrict__ const b, \
                    int* __restrict__ const wakeup, int t, int local_rank, \
                    int nx, int iy_start, int iy_end, int nz);
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

    // Print CPU and GPU information
    uint32_t brand[12];
    if (!__get_cpuid_max(0x80000004, NULL)) {
        fprintf(stderr, "Feature not implemented.");
        return 2;
    }
    __get_cpuid(0x80000002, brand+0x0, brand+0x1, brand+0x2, brand+0x3);
    __get_cpuid(0x80000003, brand+0x4, brand+0x5, brand+0x6, brand+0x7);
    __get_cpuid(0x80000004, brand+0x8, brand+0x9, brand+0xa, brand+0xb);
    printf("CPU: %s\n", brand);

    int deviceCount;
    CUDA_RT_CALL(cudaGetDeviceCount(&deviceCount));
    for (int dev = 0; dev < deviceCount; dev++) {
        cudaDeviceProp deviceProp;
        CUDA_RT_CALL(cudaGetDeviceProperties(&deviceProp, dev));
        printf("GPU: %s\n", deviceProp.name);
    }

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

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Allocate host memory
    /////////////////////////////////////////////////////////////////////////////////////////////////
    size_t total_size_count = nx * (chunk_size + 2) * nz;
    size_t total_size = total_size_count * sizeof(real);
    //cout << "size_t total_size = " << total_size << endl;
    real *x_h, *y_h, *z_h, *b_h;
    CUDA_RT_CALL(cudaMallocHost(&x_h, total_size));
    CUDA_RT_CALL(cudaMallocHost(&y_h, total_size));
    CUDA_RT_CALL(cudaMallocHost(&z_h, total_size));
    CUDA_RT_CALL(cudaMallocHost(&b_h, total_size));

    int num_block_x = (nx + BLKXSIZE - 1) / BLKXSIZE;
    int num_block_y = (chunk_size + 2 + BLKYSIZE - 1) / BLKYSIZE;
    int num_block_z = (nz + BLKZSIZE - 1) / BLKZSIZE;
    int total_num_blocks = num_block_x * num_block_y * num_block_z; // we can also use a fixed number of blocks
    int* wakeup_h;
    CUDA_RT_CALL(cudaMallocHost(&wakeup_h, total_num_blocks * sizeof(int)));


    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Allocate device memory
    /////////////////////////////////////////////////////////////////////////////////////////////////

    real* x_old;
    real* x_new;
    real* y_old;
    real* y_new;
    real* z_old;
    real* z_new;
    real* b;
    real* dfdx;
    real* dfdy;
    real* dfdz;
    CUDA_RT_CALL(cudaMalloc(&x_old, total_size));
    CUDA_RT_CALL(cudaMalloc(&x_new, total_size));
    CUDA_RT_CALL(cudaMalloc(&y_old, total_size));
    CUDA_RT_CALL(cudaMalloc(&y_new, total_size));
    CUDA_RT_CALL(cudaMalloc(&z_old, total_size));
    CUDA_RT_CALL(cudaMalloc(&z_new, total_size));
    CUDA_RT_CALL(cudaMalloc(&b, total_size));
    CUDA_RT_CALL(cudaMalloc(&dfdx, total_size));
    CUDA_RT_CALL(cudaMalloc(&dfdy, total_size));
    CUDA_RT_CALL(cudaMalloc(&dfdz, total_size));

    real* init_rand; // store random values for initial conditions
    CUDA_RT_CALL(cudaMalloc(&init_rand, total_size));

    real* metric; // store metric values for activation
    CUDA_RT_CALL(cudaMalloc(&metric, total_size));

    int* wakeup;
    int* wakeup_next;
    CUDA_RT_CALL(cudaMalloc(&wakeup, total_num_blocks * sizeof(int)));
    CUDA_RT_CALL(cudaMalloc(&wakeup_next, total_num_blocks * sizeof(int)));

    cout << "Host memory allocated on CPU rank " << rank << ": " \
             << 4 * total_size/(1024.*1024.*1024.) << " Gigabytes" << endl;
    
    cout << "Device memory allocated on GPU rank " << rank << ": " \
             << 12 * total_size/(1024.*1024.*1024.) << " Gigabytes\n" << endl;
    
    cout << "Total dynamic blocks = " << total_num_blocks << endl;


    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Set initial conditions
    /////////////////////////////////////////////////////////////////////////////////////////////////
    
    curandGenerator_t gen;
    // Create pseudo-random number generator
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    
    // Set seed
    //CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, time(0)+rank)); // random seed
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL + rank)); // default seed 1234ULL

    // Generate nx * (chunk_size + 2) * nz doubles on device for initialization
    //CURAND_CALL(curandGenerateUniformDouble(gen, init_rand, nx * (chunk_size + 2) * nz));
    CURAND_CALL(curandGenerateUniformDouble(gen, x_old, nx * (chunk_size + 2) * nz));
    CURAND_CALL(curandGenerateUniformDouble(gen, y_old, nx * (chunk_size + 2) * nz));
    CURAND_CALL(curandGenerateUniformDouble(gen, z_old, nx * (chunk_size + 2) * nz));
    
    //launch_initialize(x_old, y_old, z_old, b, \
                      0.2, 0.2, 0.01, \
                      iy_start_global - 1, nx, (chunk_size + 2), ny, nz);
    launch_initialize_DBA(wakeup, wakeup_next, x_old, y_old, z_old, b, \
                        0.2, 0.2, 0.01, \
                        iy_start_global - 1, nx, (chunk_size + 2), ny, nz);

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
    MPI_CALL(MPI_Sendrecv(x_old + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0, 
                        x_old + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE));
    MPI_CALL(MPI_Sendrecv(y_old + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0,
                        y_old + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE));
    MPI_CALL(MPI_Sendrecv(z_old + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0,
                        z_old + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE));
    MPI_CALL(MPI_Sendrecv(b + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0,
                        b + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE));
    
    CUDA_RT_CALL(cudaStreamSynchronize(push_bottom_stream));

    MPI_CALL(MPI_Sendrecv(x_old + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, 
                        x_old, nx * nz, MPI_REAL_TYPE, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    MPI_CALL(MPI_Sendrecv(y_old + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, 
                        y_old, nx * nz, MPI_REAL_TYPE, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    MPI_CALL(MPI_Sendrecv(z_old + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, 
                        z_old, nx * nz, MPI_REAL_TYPE, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    MPI_CALL(MPI_Sendrecv(b + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, b, nx * nz,
                        MPI_REAL_TYPE, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    
    POP_RANGE

    CUDA_RT_CALL(cudaDeviceSynchronize());

    if (0 == rank) {
        printf(
            "Simulation: %d iterations on %d x %d x %d mesh with sum check "
            "every %d iterations\n",
            iter_max, nx, ny, nz, nccheck);
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////
    // Create files for writing data
    /////////////////////////////////////////////////////////////////////////////////////////////////

    int iter = 1;
    bool calculate_sum;
    real sum_c = 0.0;
    int sum_wakeup = 0;

    size_t nx_s = (size_t)nx;
    size_t nz_s = (size_t)nz;
    size_t ny_part = (size_t)(ny - iy_start_global);
    size_t sendback_size = std::min(ny_part * nx_s * nz_s, chunk_size * nx_s * nz_s) * sizeof(real);
    
    // Save initial configuration
    //Copy calculated data to CPU host on respective rank
    CUDA_RT_CALL(cudaMemcpy(x_h + (nx * nz), x_old + (nx * nz), \
                        sendback_size, cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaMemcpy(y_h + (nx * nz), y_old + (nx * nz), \
                        sendback_size, cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaMemcpy(z_h + (nx * nz), z_old + (nx * nz), \
                        sendback_size, cudaMemcpyDeviceToHost));
    CUDA_RT_CALL(cudaMemcpy(b_h + (nx * nz), b + (nx * nz), \
                        sendback_size, cudaMemcpyDeviceToHost));
    
    CUDA_RT_CALL(cudaMemcpy(wakeup_h, wakeup, \
                        total_num_blocks * sizeof(int), \
                        cudaMemcpyDeviceToHost));

    // Write initial condition files
    write_output_vtr(x_h, y_h, z_h, b_h, wakeup_h, 0, rank, nx, iy_start_global-1, iy_end_global + 1, nz);
    if (rank == 0) {
        write_output_pvtr(size, 0, nx, ny, nz);
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
        // Calculate mu Halo region first
        /////////////////////////////////////////////////////////////////////////////////////////////////
        CUDA_RT_CALL(cudaStreamWaitEvent(push_top_stream, reset_sum_done, 0));
        //launch_mu_kernel(x_old, y_old, z_old, b, dfdx, dfdy, dfdz, \
                        iy_start, (iy_start + 1), nx, nz, push_top_stream);
        launch_mu_DBA_kernel(wakeup, x_old, y_old, z_old, b, dfdx, dfdy, dfdz, metric, \
                            iy_start, (iy_start + 1), nx, nz, push_top_stream);
        CUDA_RT_CALL(cudaEventRecord(push_top_done, push_top_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(push_bottom_stream, reset_sum_done, 0));
        //launch_mu_kernel(x_old, y_old, z_old, b, dfdx, dfdy, dfdz, \
                        (iy_end - 1), iy_end, nx, nz, push_bottom_stream);
        launch_mu_DBA_kernel(wakeup, x_old, y_old, z_old, b, dfdx, dfdy, dfdz, metric, \
                            (iy_end - 1), iy_end, nx, nz, push_bottom_stream);
        CUDA_RT_CALL(cudaEventRecord(push_bottom_done, push_bottom_stream));

        // Calculate center region
        //launch_mu_kernel(x_old, y_old, z_old, b, dfdx, dfdy, dfdz, \
                        (iy_start + 1), (iy_end - 1), nx, nz, compute_stream);
        launch_mu_DBA_kernel(wakeup, x_old, y_old, z_old, b, dfdx, dfdy, dfdz, metric, \
                            (iy_start + 1), (iy_end - 1), nx, nz, compute_stream);
        CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));

        // Apply periodic boundary conditions
        CUDA_RT_CALL(cudaStreamSynchronize(push_top_stream));
        PUSH_RANGE("MPI", 5)
        MPI_CALL(MPI_Sendrecv(dfdx + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0, \
                              dfdx + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, \
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(dfdy + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0, \
                              dfdy + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, \
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(dfdz + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0, \
                              dfdz + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, \
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(metric + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0, \
                              metric + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, \
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        
        
        CUDA_RT_CALL(cudaStreamSynchronize(push_bottom_stream));
        MPI_CALL(MPI_Sendrecv(dfdx + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, \
                              dfdx, nx * nz, MPI_REAL_TYPE, top, 0, \
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(dfdy + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, \
                              dfdy, nx * nz, MPI_REAL_TYPE, top, 0, \
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(dfdz + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, \
                              dfdz, nx * nz, MPI_REAL_TYPE, top, 0, \
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(metric + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, \
                              metric, nx * nz, MPI_REAL_TYPE, top, 0, \
                              MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        
        POP_RANGE
        
        /////////////////////////////////////////////////////////////////////////////////////////////////
        // Calculate main kernel Halo region
        /////////////////////////////////////////////////////////////////////////////////////////////////
        //CURAND_CALL(curandSetStream(gen, push_top_stream));
        // Generate nx * (chunk_size + 2) * nz doubles on device for initialization
        //CURAND_CALL(curandGenerateUniformDouble(gen, phi_rand, nx * (chunk_size + 2) * nz));

        CUDA_RT_CALL(cudaStreamWaitEvent(push_top_stream, compute_done, 0));
        //launch_update_kernel(x_old, y_old, z_old, b, x_new, y_new, z_new, \
                            dfdx, dfdy, dfdz, sum_c_d, \
                            iy_start, (iy_start + 1), nx, nz, calculate_sum, \
                            push_top_stream);
        launch_update_DBA_kernel(wakeup, x_old, y_old, z_old, b, x_new, y_new, z_new, \
                            dfdx, dfdy, dfdz, sum_c_d, sum_wakeup_d, \
                            iy_start, (iy_start + 1), nx, nz, calculate_sum, \
                            push_top_stream);
        CUDA_RT_CALL(cudaEventRecord(push_top_done, push_top_stream));

        CUDA_RT_CALL(cudaStreamWaitEvent(push_bottom_stream, compute_done, 0));
        //launch_update_kernel(x_old, y_old, z_old, b, x_new, y_new, z_new, \
                            dfdx, dfdy, dfdz, sum_c_d, \
                            (iy_end - 1), iy_end, nx, nz, calculate_sum, \
                            push_bottom_stream);
        launch_update_DBA_kernel(wakeup, x_old, y_old, z_old, b, x_new, y_new, z_new, \
                            dfdx, dfdy, dfdz, sum_c_d, sum_wakeup_d, \
                            (iy_end - 1), iy_end, nx, nz, calculate_sum, \
                            push_bottom_stream);
        CUDA_RT_CALL(cudaEventRecord(push_bottom_done, push_bottom_stream));

        /////////////////////////////////////////////////////////////////////////////////////////////////
        // Calculate center region
        /////////////////////////////////////////////////////////////////////////////////////////////////
        //launch_update_kernel(x_old, y_old, z_old, b, x_new, y_new, z_new, \
                            dfdx, dfdy, dfdz, sum_c_d, \
                            (iy_start + 1), (iy_end - 1), nx, nz, calculate_sum, \
                            compute_stream);
        launch_update_DBA_kernel(wakeup, x_old, y_old, z_old, b, x_new, y_new, z_new, \
                            dfdx, dfdy, dfdz, sum_c_d, sum_wakeup_d, \
                            (iy_start + 1), (iy_end - 1), nx, nz, calculate_sum, \
                            compute_stream);
        CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));

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

        MPI_CALL(MPI_Sendrecv(x_new + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0,
                            x_new + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(y_new + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0,
                            y_new + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(z_new + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0,
                            z_new + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(b + iy_start * nx * nz, nx * nz, MPI_REAL_TYPE, top, 0,
                            b + (iy_end * nx * nz), nx * nz, MPI_REAL_TYPE, bottom, 0, MPI_COMM_WORLD,
                            MPI_STATUS_IGNORE));

        CUDA_RT_CALL(cudaStreamSynchronize(push_bottom_stream));

        MPI_CALL(MPI_Sendrecv(x_new + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, x_new, nx * nz,
                            MPI_REAL_TYPE, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(y_new + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, y_new, nx * nz,
                            MPI_REAL_TYPE, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(z_new + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, z_new, nx * nz,
                            MPI_REAL_TYPE, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
        MPI_CALL(MPI_Sendrecv(b + (iy_end - 1) * nx * nz, nx * nz, MPI_REAL_TYPE, bottom, 0, b, nx * nz,
                            MPI_REAL_TYPE, top, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE));

        POP_RANGE

        /////////////////////////////////////////////////////////////////////////////////////////////////
        // Save configuration and output sum
        /////////////////////////////////////////////////////////////////////////////////////////////////
        if (calculate_sum) {
            CUDA_RT_CALL(cudaStreamSynchronize(compute_stream));
            MPI_CALL(MPI_Allreduce(sum_c_h, &sum_c, 1, MPI_REAL_TYPE, MPI_SUM, MPI_COMM_WORLD));
            MPI_CALL(MPI_Allreduce(sum_wakeup_h, &sum_wakeup, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));

            if (0 == rank) {
                printf("Iteration: %d, Sum: %8.4f, Wakeup: %d\n", iter, sum_c, sum_wakeup);
                fflush(stdout);
                //cout << "Iteration: " << iter << ", Sum: " << sum_c << ", Wakeup: " << sum_wakeup << endl;
            }
            
            // Copy calculated data to CPU host on respective rank
            CUDA_RT_CALL(cudaMemcpy(x_h + (nx * nz), x_new + (nx * nz), \
                            sendback_size, \
                            cudaMemcpyDeviceToHost));
            CUDA_RT_CALL(cudaMemcpy(y_h + (nx * nz), y_new + (nx * nz), \
                            sendback_size, \
                            cudaMemcpyDeviceToHost));
            CUDA_RT_CALL(cudaMemcpy(z_h + (nx * nz), z_new + (nx * nz), \
                            sendback_size, \
                            cudaMemcpyDeviceToHost));
            CUDA_RT_CALL(cudaMemcpy(b_h + (nx * nz), b + (nx * nz), \
                            sendback_size, \
                            cudaMemcpyDeviceToHost));
            CUDA_RT_CALL(cudaMemcpy(wakeup_h, wakeup, \
                        total_num_blocks * sizeof(int), \
                        cudaMemcpyDeviceToHost));

            // Write files on each CPU rank
            if (vtk){
                write_output_vtr(x_h, y_h, z_h, b_h, wakeup_h, iter, rank, nx, iy_start_global-1, iy_end_global + 1, nz);
                if (rank == 0) {
                write_output_pvtr(size, iter, nx, ny, nz);
                }
            }
            
            MPI_CALL(MPI_Barrier(MPI_COMM_WORLD));
        }

        /////////////////////////////////////////////////////////////////////////////////////////////////
        // Update active blocks for next iteration
        /////////////////////////////////////////////////////////////////////////////////////////////////
        launch_wake_DBA_kernel(wakeup, wakeup_next, metric, nx, (chunk_size + 2), nz, compute_stream);
        //CUDA_RT_CALL(cudaEventRecord(compute_done, compute_stream));
        
        // Swap new and old fields
        swap(x_new, x_old);
        swap(y_new, y_old);
        swap(z_new, z_old);

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
    CUDA_RT_CALL(cudaFreeHost(x_h));
    CUDA_RT_CALL(cudaFreeHost(y_h));
    CUDA_RT_CALL(cudaFreeHost(z_h));
    CUDA_RT_CALL(cudaFreeHost(b_h));
    CUDA_RT_CALL(cudaFreeHost(wakeup_h));

    // Free device memory
    CUDA_RT_CALL(cudaFree(x_old));
    CUDA_RT_CALL(cudaFree(x_new));
    CUDA_RT_CALL(cudaFree(y_old));
    CUDA_RT_CALL(cudaFree(y_new));
    CUDA_RT_CALL(cudaFree(z_old));
    CUDA_RT_CALL(cudaFree(z_new));
    CUDA_RT_CALL(cudaFree(b));
    CUDA_RT_CALL(cudaFree(dfdx));
    CUDA_RT_CALL(cudaFree(dfdy));
    CUDA_RT_CALL(cudaFree(dfdz));

    CUDA_RT_CALL(cudaFree(init_rand));
    CUDA_RT_CALL(cudaFree(metric));

    CUDA_RT_CALL(cudaFree(wakeup));
    CUDA_RT_CALL(cudaFree(wakeup_next));

    CUDA_RT_CALL(cudaFreeHost(sum_c_h));
    CUDA_RT_CALL(cudaFree(sum_c_d));
    CUDA_RT_CALL(cudaFreeHost(sum_wakeup_h));
    CUDA_RT_CALL(cudaFree(sum_wakeup_d));

    MPI_CALL(MPI_Finalize());
    return 0;
}