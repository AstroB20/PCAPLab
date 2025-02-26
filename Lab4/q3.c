#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size, matrix[4][4], result[4][4];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 4)
    {
        if (rank == 0)
            printf("This program requires exactly 4 processes.\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0)
    {
        printf("Enter a 4x4 matrix:\n");
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                scanf("%d", &matrix[i][j]);
    }

    MPI_Bcast(&matrix, 16, MPI_INT, 0, MPI_COMM_WORLD);

    for (int j = 0; j < 4; j++)
    {
        result[rank][j] = matrix[rank][j] * (rank + 1);
    }

    MPI_Gather(result[rank], 4, MPI_INT, result, 4, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Output matrix:\n");
        for (int i = 0; i < 4; i++)
        {
            for (int j = 0; j < 4; j++)
                printf("%d ", result[i][j]);
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
}
