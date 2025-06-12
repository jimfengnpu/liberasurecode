#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>
#include <gpu_rscode/gpu_rscode.h>

char* gen_random_buffer(int blocksize)
{
    int i;
    char *buf = (char*)malloc(blocksize);

    for (i = 0; i < blocksize; i++) {
        buf[i] = (char)(rand() % 255); 
    }

    return buf;
}

int is_missing(int *missing_idxs, int index_to_check)
{
    int i = 0;
    while (missing_idxs[i] > -1) {
        if (missing_idxs[i] == index_to_check) {
            return 1;
        }
        i++;
    }
    return 0;
}

int test_encode_decode(uint8_t *matrix, int k, int m, int num_missing, int blocksize)
{
    char **data = (char**)malloc(sizeof(char*)*k);
    char **parity = (char**)malloc(sizeof(char*)*m);
    char **missing_bufs = (char**)malloc(sizeof(char*)*num_missing);
    int *missing = (int*)malloc(sizeof(int)*(num_missing+1));
    int n = k + m;
    int i;
    int ret = 1;
    
    srand((unsigned int)time(0));
        
    for (i = 0; i < k; i++) {
        data[i] = gen_random_buffer(blocksize);
    }

    for (i = 0; i < m; i++) {
        parity[i] = (char*)malloc(blocksize);
    }

    for (i = 0; i < num_missing+1; i++) {
        missing[i] = -1;
    }
    
    // Encode
    printf("encode start\n");
    libgpu_rscode_encode(matrix, data, parity, k, m, blocksize);
    printf("encode finish\n");
    // Copy data and parity
    for (i = 0;i < num_missing; i++) {
        int idx = rand() % n;
        while (is_missing(missing, idx)) {
            idx = rand() % n;
        }
        missing_bufs[i] = (char*)malloc(blocksize);
        memcpy(missing_bufs[i], idx < k ? data[idx] : parity[idx - k], blocksize);
        missing[i] = idx;
    }
    
    // Zero missing bufs
    for (i = 0;i < num_missing; i++) {
        if (missing[i] < k) {
            memset(data[missing[i]], 0, blocksize);
        } else {
            memset(parity[missing[i] - k], 0, blocksize);
        }
    }
    
    // Decode and check
    printf("decode start\n");
    libgpu_rscode_decode(matrix, data, parity, k, m, missing, blocksize);
    printf("decode finish\n");
    for (i = 0; i < num_missing; i++) {
        int idx = missing[i];
        if (idx < k) { 
            if (memcmp(data[idx], missing_bufs[i], blocksize)) {
                ret = 0;
            }
        } else if (memcmp(parity[idx - k], missing_bufs[i], blocksize)) {
            ret = 0;
        }
    }

    for (i = 0; i < k; i++) {
        free(data[i]);
    }
    free(data);
    for (i = 0; i < m; i++) {
        free(parity[i]);
    }
    free(parity);
    for (i = 0; i < num_missing; i++) {
        free(missing_bufs[i]);
    }
    free(missing_bufs);
    free(missing);

    return ret;
}

int matrix_dimensions[][2] = { {5, 3}, {-1, -1} };

int main(void) 
{
    int i = 0;
    int blocksize = 128;
    uint8_t *matrix;

    while (matrix_dimensions[i][0] >= 0) {
        int k = matrix_dimensions[i][0], m = matrix_dimensions[i][1];
        matrix = NULL;
        printf("Start testcase k=%d m=%d\n", k, m);
        libgpu_rscode_init(k, m, &matrix);

        if (!matrix) {
            fprintf(stderr, "Error init matrix for k=%d, m=%d\n", k, m);
            return 1;
        }

        int enc_dec_res = test_encode_decode(matrix, k, m, m, blocksize);
        if (!enc_dec_res) {
            fprintf(stderr, "Error running encode/decode test for k=%d, m=%d, bs=%d\n", k, m, blocksize);
            return 1;
        }
        
        free(matrix);
        i++;
    }
}