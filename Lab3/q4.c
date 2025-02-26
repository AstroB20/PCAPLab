#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
    int rank, size, len;
    char S1[100], S2[100], local_S1[100], local_S2[100], local_result[100], result[100];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        printf("Enter first string S1: ");
        scanf("%s", S1);
        printf("Enter second string S2: ");
        scanf("%s", S2);

        len = strlen(S1);
        if (len != strlen(S2) || len % size != 0)
        {
            printf("Strings must be of the same length and evenly divisible by the number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int chunk_size = len / size;

    MPI_Scatter(S1, chunk_size, MPI_CHAR, local_S1, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(S2, chunk_size, MPI_CHAR, local_S2, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk_size; i++)
    {
        local_result[i * 2] = local_S1[i];
        local_result[i * 2 + 1] = local_S2[i];
    }

    MPI_Gather(local_result, chunk_size * 2, MPI_CHAR, result, chunk_size * 2, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        result[len * 2] = '\0';
        printf("Resultant string: %s\n", result);
    }

    MPI_Finalize();
    return 0;
}
