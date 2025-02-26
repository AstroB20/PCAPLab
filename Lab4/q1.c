#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Function to compute factorial
long long factorial(int num)
{
    long long fact = 1;
    for (int i = 1; i <= num; i++)
    {
        fact *= i;
    }
    return fact;
}

int main(int argc, char *argv[])
{
    int rank, size, N;
    long long local_factorial, scan_result;

    MPI_Init(&argc, &argv);

    // Error handling for MPI initialization
    int err;
    MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

    err = MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "Error getting MPI size\n");
        MPI_Abort(MPI_COMM_WORLD, err);
    }

    err = MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "Error getting MPI rank\n");
        MPI_Abort(MPI_COMM_WORLD, err);
    }

    if (rank == 0)
    {
        printf("Enter value of N: ");
        scanf("%d", &N);
        if (N < 1 || N > size)
        {
            fprintf(stderr, "N should be between 1 and %d\n", size);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    // Broadcast N to all processes
    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank < N)
    {
        local_factorial = factorial(rank + 1);
    }
    else
    {
        local_factorial = 0;
    }

    // Use MPI_Scan to compute the cumulative sum of factorials
    err = MPI_Scan(&local_factorial, &scan_result, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (err != MPI_SUCCESS)
    {
        fprintf(stderr, "MPI_Scan failed on process %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, err);
    }

    // Print result in the last process
    if (rank == N - 1)
    {
        printf("Sum of factorials up to %d! is %lld\n", N, scan_result);
    }

    MPI_Finalize();
    return 0;
}
