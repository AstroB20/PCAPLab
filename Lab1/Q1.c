#include <mpi.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char *argv[]) {
    int rank, size;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    int x = rank; // x is set to the rank of the process
    double result = pow(x, rank); // Compute x^rank
    
    printf("Process %d: %d^%d = %.2f\n", rank, x, rank, result);
    
    // Finalize MPI
    MPI_Finalize();
    return 0;
}
