#include <gpu_rscode.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "encode.h"
#include "decode.h"

void libgpu_rscode_init(int k, int m, uint8_t **pmatrix)
{
    if(k <= 0 && m <= 0) {
        return;
    }
    uint8_t *matrix = malloc(k * m * sizeof(uint8_t));
    GPU_generate_encode_matrix(matrix, k, m);
    *pmatrix = matrix;
}


int libgpu_rscode_encode(uint8_t *generator_matrix, char **data, char **parity, int k, int m, int blocksize)
{
    int i;
    int n = k + m;

    for (i = k; i < n; i++) {
        memset(parity[i - k], 0, blocksize);
    }
    encode_data(generator_matrix, (uint8_t**)data, (uint8_t**)parity,
        k, m, blocksize, 0, 1);
    return 0;
}

int libgpu_rscode_decode(uint8_t *generator_matrix, char **data, char **parity, int k, int m, int *missing, int blocksize)
{
    int n = m + k;
    int *_missing = (int*)malloc(sizeof(int)*n);
    int num_missing = 0;

    memset(_missing, 0, sizeof(int)*n);

    while (missing[num_missing] > -1) {
        _missing[missing[num_missing]] = 1;
        num_missing++;
    }

    if (num_missing > m) {
        free(_missing);
        return -1;
    }
    decode_data(generator_matrix, (uint8_t**)data, (uint8_t**)parity,
        _missing, k, m, blocksize, 0, 1);
    
    free(_missing);
    return 0;
}

int libgpu_rscode_reconstruct(uint8_t *generator_matrix, char **data, char **parity, int k, int m, int *missing, int destination_idx, int blocksize)
{
    return 0;
}
