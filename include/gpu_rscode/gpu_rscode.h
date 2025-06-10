#ifndef LIB_GPU_RSCODE
#define LIB_GPU_RSCODE
#include <stdint.h>


void libgpu_rscode_init(int k, int m, uint8_t **pmatrix);
int libgpu_rscode_encode(uint8_t *generator_matrix, char **data, char **parity, int k, int m, int blocksize);
int libgpu_rscode_decode(uint8_t *generator_matrix, char **data, char **parity, int k, int m, int *missing, int blocksize);
int libgpu_rscode_reconstruct(uint8_t *generator_matrix, char **data, char **parity, int k, int m, int *missing, int destination_idx, int blocksize);

#endif