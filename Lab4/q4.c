#include <stdio.h>
#include <mpi.h>
#include <string.h>

int main(int argc, char *argv[])
{
    int rank, size, N;
    char word[100], result[1000] = "", local_part[100];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0)
    {
        printf("Enter a word: ");
        scanf("%s", word);
        N = strlen(word);

        if (N != size)
        {
            fprintf(stderr, "Number of processes must match word length.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&N, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(word, N, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 0; i <= rank; i++)
    {
        local_part[i] = word[rank];
    }
    local_part[rank + 1] = '\0';

    MPI_Gather(local_part, rank + 1, MPI_CHAR, result, rank + 1, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        printf("Output word: %s\n", result);
    }

    MPI_Finalize();
    return 0;
}
