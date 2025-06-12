/*
 * =====================================================================================
 *
 *       Filename:  encode.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  12/05/2012 10:42:32 PM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Shuai YUAN (yszheda AT gmail.com), 
 *        Company:  
 *
 * =====================================================================================
 */

#include "encode.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "matrix.h"


struct ThreadDataType {
    int id;
    int nativeBlockNum;
    int parityBlockNum;
    int chunkSize;
    int gridDimXSize;
    int streamNum;
    char* fileName;
    uint8_t* dataBuf;
    uint8_t* codeBuf;
    uint8_t* encodingMatrix;
};	/* ----------  end of struct ThreadDataType  ---------- */

typedef struct ThreadDataType ThreadDataType;

static pthread_barrier_t barrier;

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  encode
 *  Description:  encode the given buffer of data chunks in the GPU with <id>
 * =====================================================================================
 */
void encode(uint8_t *dataBuf, uint8_t *codeBuf, uint8_t *encodingMatrix, int id, int nativeBlockNum, int parityBlockNum, int chunkSize, int gridDimXSize, int streamNum)
{
    uint8_t *encodingMatrix_d;	//device
    int matrixSize = parityBlockNum * nativeBlockNum * sizeof(uint8_t);
    checkCudaErrors(cudaMalloc((void **)&encodingMatrix_d, matrixSize));
    checkCudaErrors(cudaMemcpy(encodingMatrix_d, encodingMatrix, matrixSize, cudaMemcpyHostToDevice));
    // Use cuda stream to encode the file
    // to achieve computation and comunication overlapping
    // Use DFS way
    int streamMinChunkSize = chunkSize / streamNum;
    cudaStream_t stream[streamNum];
    for (int i = 0; i < streamNum; i++)
    {
        checkCudaErrors(cudaStreamCreate(&stream[i]));
    }

    uint8_t *dataBuf_d[streamNum];		//device
    uint8_t *codeBuf_d[streamNum];		//device
    for (int i = 0; i < streamNum; i++)
    {
        int streamChunkSize = streamMinChunkSize;
        if (i == streamNum - 1)
        {
            streamChunkSize = chunkSize - i * streamMinChunkSize;
        }

        int dataSize = nativeBlockNum * streamChunkSize * sizeof(uint8_t);
        int codeSize = parityBlockNum * streamChunkSize * sizeof(uint8_t);

        checkCudaErrors(cudaMalloc((void **)&dataBuf_d[i], dataSize));
        checkCudaErrors(cudaMalloc((void **)&codeBuf_d[i], codeSize));
    }

    for (int i = 0; i < streamNum; i++)
    {
        int streamChunkSize = streamMinChunkSize;
        if (i == streamNum - 1)
        {
            streamChunkSize = chunkSize - i * streamMinChunkSize;
        }
        for (int j = 0; j < nativeBlockNum; j++)
        {
            checkCudaErrors(cudaMemcpyAsync(dataBuf_d[i] + j * streamChunkSize,
                    dataBuf + j * chunkSize + i * streamMinChunkSize,
                    streamChunkSize * sizeof(uint8_t),
                    cudaMemcpyHostToDevice,
                    stream[i]));
        }

        encode_chunk(dataBuf_d[i], encodingMatrix_d, codeBuf_d[i], nativeBlockNum, parityBlockNum, streamChunkSize, gridDimXSize, stream[i]);

        for (int j = 0; j < parityBlockNum; j++)
        {
            checkCudaErrors(cudaMemcpyAsync(codeBuf + j * chunkSize + i * streamMinChunkSize,
                    codeBuf_d[i] + j * streamChunkSize,
                    streamChunkSize * sizeof(uint8_t),
                    cudaMemcpyDeviceToHost,
                    stream[i]));
        }
    }

    for (int i = 0; i < streamNum; i++)
    {
        checkCudaErrors(cudaFree(dataBuf_d[i]));
        checkCudaErrors(cudaFree(codeBuf_d[i]));
    }
    checkCudaErrors(cudaFree(encodingMatrix_d));

    for (int i = 0; i < streamNum; i++)
    {
        checkCudaErrors(cudaStreamDestroy(stream[i]));
    }
}

static void* GPU_thread_func(void * args)
{
    ThreadDataType* thread_data = (ThreadDataType *) args;
    checkCudaErrors(cudaSetDevice(thread_data->id));

    struct timespec start, end;
    pthread_barrier_wait(&barrier);
    clock_gettime(CLOCK_REALTIME, &start);
    pthread_barrier_wait(&barrier);

    encode(thread_data->dataBuf,
            thread_data->codeBuf,
            thread_data->encodingMatrix,
            thread_data->id,
            thread_data->nativeBlockNum,
            thread_data->parityBlockNum,
            thread_data->chunkSize,
            thread_data->gridDimXSize,
            thread_data->streamNum);

    pthread_barrier_wait(&barrier);
    clock_gettime(CLOCK_REALTIME, &end);
    if (thread_data->id == 0)
    {
        double totalTime = (double) (end.tv_sec - start.tv_sec) * 1000
            + (double) (end.tv_nsec - start.tv_nsec) / (double) 1000000L;
        printf("Total GPU encoding time using multiple devices: %fms\n", totalTime);
    }

    // if (thread_data->id == 0)
    // {
    //     char *fileName = thread_data->fileName;
    //     int totalSize = thread_data->totalSize;
    //     char metadata_file_name[strlen(fileName) + 15];
    //     sprintf(metadata_file_name, "%s.METADATA", fileName);
    //     write_metadata(metadata_file_name, totalSize, parityBlockNum, nativeBlockNum, encodingMatrix);
    // }
    // NOTE: Pageable Host Memory is preferred here since the encodingMatrix is small
    // cudaFreeHost(encodingMatrix);

    return NULL;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  encode_data
 *  Description:  encode the input file <fileName> with the given settings
 * =====================================================================================
 */
extern "C"
void encode_data(uint8_t *generator_matrix, uint8_t **dataBuf, uint8_t **codeBuf, int nativeBlockNum, int parityBlockNum, int chunkSize, int gridDimXSize, int streamNum)
{

    cudaDeviceProp deviceProperties;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProperties, 0));
    int maxGridDimXSize = min(deviceProperties.maxGridSize[0], deviceProperties.maxGridSize[1]);
    if (gridDimXSize > maxGridDimXSize || gridDimXSize <= 0)
    {
        // printf("Valid grid size: (0, %d]\n", maxGridDimXSize);
        gridDimXSize = maxGridDimXSize;
    }

    int GPU_num;
    checkCudaErrors(cudaGetDeviceCount(&GPU_num));

    void* threads = malloc(GPU_num * sizeof(pthread_t));
    ThreadDataType* thread_data = (ThreadDataType *) malloc(GPU_num * sizeof(ThreadDataType));

    uint8_t *dataBufPerDevice[GPU_num];
    uint8_t *codeBufPerDevice[GPU_num];
    uint8_t *encodingMatrx;
    size_t matrixSize = nativeBlockNum*parityBlockNum*sizeof(uint8_t);
    checkCudaErrors(cudaMallocHost((void **)&encodingMatrx, matrixSize));
    checkCudaErrors(cudaMemcpy(encodingMatrx, generator_matrix, matrixSize, cudaMemcpyHostToHost));
    pthread_barrier_init(&barrier, NULL, GPU_num);

    int minChunkSizePerDevice = chunkSize / GPU_num;
    for (int i = 0; i < GPU_num; ++i)
    {
        checkCudaErrors(cudaSetDevice(i));

        thread_data[i].id = i;
        thread_data[i].nativeBlockNum = nativeBlockNum;
        thread_data[i].parityBlockNum = parityBlockNum;
        int deviceChunkSize = minChunkSizePerDevice;
        if (i == GPU_num - 1)
        {
            deviceChunkSize = chunkSize - i * minChunkSizePerDevice;
        }
        thread_data[i].chunkSize = deviceChunkSize;
        thread_data[i].gridDimXSize = gridDimXSize;
        thread_data[i].streamNum = streamNum;

        int deviceDataSize = nativeBlockNum * deviceChunkSize * sizeof(uint8_t);
        int deviceCodeSize = parityBlockNum * deviceChunkSize * sizeof(uint8_t);
        checkCudaErrors(cudaMallocHost((void **)&dataBufPerDevice[i], deviceDataSize));
        checkCudaErrors(cudaMallocHost((void **)&codeBufPerDevice[i], deviceCodeSize));
        for (int j = 0; j < nativeBlockNum; ++j)
        {
            // Pinned Host Memory
            checkCudaErrors(cudaMemcpy(dataBufPerDevice[i] + j * deviceChunkSize,
                    dataBuf[j] + i * minChunkSizePerDevice,
                    deviceChunkSize,
                    cudaMemcpyHostToHost));
        }
        thread_data[i].dataBuf = dataBufPerDevice[i];
        thread_data[i].codeBuf = codeBufPerDevice[i];
        thread_data[i].encodingMatrix = encodingMatrx;

        pthread_create(&((pthread_t*) threads)[i], NULL, GPU_thread_func, (void *) &thread_data[i]);
    }

    for (int i = 0; i < GPU_num; ++i)
    {
        pthread_join(((pthread_t*) threads)[i], NULL);
    }

    for (int i = 0; i < GPU_num; ++i)
    {
        int deviceChunkSize = minChunkSizePerDevice;
        if (i == GPU_num - 1) {
            deviceChunkSize = chunkSize - i * minChunkSizePerDevice;
        }

        for (int j = 0; j < parityBlockNum; ++j)
        {
            // Pinned Host Memory
            checkCudaErrors(cudaMemcpy(codeBuf[j] + i * minChunkSizePerDevice,
                    codeBufPerDevice[i] + j * deviceChunkSize,
                    deviceChunkSize,
                    cudaMemcpyHostToHost));
        }

        // Pinned Host Memory
        checkCudaErrors(cudaFreeHost(dataBufPerDevice[i]));
        checkCudaErrors(cudaFreeHost(codeBufPerDevice[i]));
    }

    pthread_barrier_destroy(&barrier);
    checkCudaErrors(cudaFreeHost(encodingMatrx));
    checkCudaErrors(cudaDeviceReset());

}

extern "C"
void GPU_generate_encode_matrix(uint8_t *encodingMatrix, int nativeBlockNum, int parityBlockNum) {
    uint8_t *encodingMatrix_d;	//device
    int matrixSize = parityBlockNum * nativeBlockNum * sizeof(uint8_t);
    checkCudaErrors(cudaMalloc((void **)&encodingMatrix_d, matrixSize));

    // record event
    const int maxBlockDimSize = 16;
    int blockDimX = min(parityBlockNum, maxBlockDimSize);
    int blockDimY = min(nativeBlockNum, maxBlockDimSize);
    int gridDimX = (int) ceil((float) parityBlockNum / blockDimX);
    int gridDimY = (int) ceil((float) nativeBlockNum / blockDimY);
    dim3 grid(gridDimX, gridDimY);
    dim3 block(blockDimX, blockDimY);
    gen_encoding_matrix<<<grid, block>>>(encodingMatrix_d, parityBlockNum, nativeBlockNum);
    checkCudaErrors(cudaMemcpy(encodingMatrix, encodingMatrix_d, matrixSize, cudaMemcpyDeviceToHost));
}