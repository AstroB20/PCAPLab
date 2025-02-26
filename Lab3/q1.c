#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    int rank, size, M, *arr = NULL, *sub_arr = NULL;
    float *avg_arr = NULL, total_avg;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        printf("Enter the value of M: ");
        scanf("%d", &M);

        arr = (int *)malloc(size * M * sizeof(int));
        printf("Enter %d elements:\n", size * M);
        for (int i = 0; i < size * M; i++)
        {
            scanf("%d", &arr[i]);
        }
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    sub_arr = (int *)malloc(M * sizeof(int));
    MPI_Scatter(arr, M, MPI_INT, sub_arr, M, MPI_INT, 0, MPI_COMM_WORLD);

    float sub_avg = 0;
    for (int i = 0; i < M; i++)
    {
        sub_avg += sub_arr[i];
    }
    sub_avg /= M;

    avg_arr = (float *)malloc(size * sizeof(float));
    MPI_Gather(&sub_avg, 1, MPI_FLOAT, avg_arr, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        total_avg = 0;
        for (int i = 0; i < size; i++)
        {
            total_avg += avg_arr[i];
        }
        total_avg /= size;
        printf("Total average: %f\n", total_avg);

        free(arr);
        free(avg_arr);
    }

    free(sub_arr);
    MPI_Finalize();
    return 0;
}