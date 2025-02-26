#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[])
{
    int rank, size, M;
    double local_sum = 0.0, local_avg, total_avg;
    int *ID_array = NULL;
    int *local_array = NULL;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        // Read values for M and the ID array
        printf("Enter the number of elements per process (M): ");
        scanf("%d", &M);

        int total_elements = size * M;
        ID_array = (int *)malloc(total_elements * sizeof(int));

        printf("Enter %d elements for the ID array:\n", total_elements);
        for (int i = 0; i < total_elements; i++)
        {
            scanf("%d", &ID_array[i]);
        }
    }

    // Broadcast M to all processes
    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Allocate memory for local array
    local_array = (int *)malloc(M * sizeof(int));

    // Scatter ID_array among all processes
    MPI_Scatter(ID_array, M, MPI_INT, local_array, M, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local sum and average
    for (int i = 0; i < M; i++)
    {
        local_sum += local_array[i];
    }
    local_avg = local_sum / M;

    // Gather all local averages at the root process
    double *all_averages = NULL;
    if (rank == 0)
    {
        all_averages = (double *)malloc(size * sizeof(double));
    }

    MPI_Gather(&local_avg, 1, MPI_DOUBLE, all_averages, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Compute total average at root process
    if (rank == 0)
    {
        double sum_of_averages = 0.0;
        for (int i = 0; i < size; i++)
        {
            sum_of_averages += all_averages[i];
        }
        total_avg = sum_of_averages / size;

        printf("Total average: %f\n", total_avg);

        free(ID_array);
        free(all_averages);
    }

    free(local_array);
    MPI_Finalize();
    return 0;
}
