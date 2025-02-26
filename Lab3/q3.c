#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

int is_vowel(char c)
{
    c = tolower(c);
    return (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u');
}

int main(int argc, char *argv[])
{
    int rank, size, local_count = 0, total_count;
    char str[100], local_str[100];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int len;
    if (rank == 0)
    {
        printf("Enter a string: ");
        scanf("%s", str);
        len = strlen(str);
        if (len % size != 0)
        {
            printf("String length must be evenly divisible by number of processes.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    int chunk_size = len / size;
    MPI_Scatter(str, chunk_size, MPI_CHAR, local_str, chunk_size, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk_size; i++)
    {
        if (!is_vowel(local_str[i]) && isalpha(local_str[i]))
        {
            local_count++;
        }
    }

    MPI_Gather(&local_count, 1, MPI_INT, &total_count, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0)
    {
        int total = 0;
        printf("Non-vowels found by each process:\n");
        for (int i = 0; i < size; i++)
        {
            printf("Process %d: %d\n", i, ((int *)&total_count)[i]);
            total += ((int *)&total_count)[i];
        }
        printf("Total non-vowels: %d\n", total);
    }

    MPI_Finalize();
    return 0;
}
