#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size, matrix[3][3], element, local_count = 0, total_count;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3)
    {
        if (rank == 0)
            printf("This program requires exactly 3 processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        printf("Enter a 3x3 matrix:\n");
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                scanf("%d", &matrix[i][j]);

        printf("Enter element to search: ");
        scanf("%d", &element);
    }

    MPI_Bcast(&matrix, 9, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&element, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Each process searches one row of the matrix
    for (int j = 0; j < 3; j++)
    {
        if (matrix[rank][j] == element)
            local_count++;
    }

    MPI_Reduce(&local_count, &total_count, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("Element %d found %d times in the matrix.\n", element, total_count);

    MPI_Finalize();
    return 0;
}
