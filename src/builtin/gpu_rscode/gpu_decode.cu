
/*
 * =====================================================================================
 *
 *       Filename:  decode.cu
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  12/05/2012 10:50:55 PM
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Shuai YUAN (yszheda AT gmail.com),
 *        Company:  
 *
 * =====================================================================================
 */

#include "decode.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "helper_cuda.h"
#include "matrix.h"
#include "cpu_decode.h"


struct ThreadDataType {
    int id;
    int nativeBlockNum;
    int parityBlockNum;
    int chunkSize;
    int totalSize;
    int gridDimXSize;
    int streamNum;
    uint8_t* dataBuf;
    uint8_t* codeBuf;
    uint8_t* decodingMatrix;
};	/* ----------  end of struct ThreadDataType  ---------- */

typedef struct ThreadDataType ThreadDataType;

static pthread_barrier_t barrier;

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  show_square_matrix_debug
 *  Description:  show the content of a square matrix
 *  Used only for debugging
 * =====================================================================================
 */
#ifdef DEBUG
void show_squre_matrix_debug(uint8_t *matrix, int size)
{
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < size; j++)
        {
            printf("%d ", matrix[i*size+j]);
        }
        printf("\n");
    }
}
#endif
/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  copy_matrix
 *  Description:  copy the row with <srcRowIndex> from the matrix <src>
 *  to the row with <desRowIndex> of the matrix <des>
 * =====================================================================================
 */
void copy_matrix(uint8_t *src, uint8_t *des, int srcRowIndex, int desRowIndex, int rowSize)
{
    for (int i = 0; i < rowSize; i++)
    {
        des[desRowIndex * rowSize + i] = src[srcRowIndex * rowSize + i];
    }
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  decode
 *  Description:  decode the given buffer of code chunks in the GPU with <id>
 * =====================================================================================
 */
void decode(uint8_t *dataBuf, uint8_t *codeBuf, uint8_t *decodingMatrix, int id, int nativeBlockNum, int parityBlockNum, int chunkSize, int gridDimXSize, int streamNum)
{
    float totalCommunicationTime = 0;
    // compute total execution time
    float totalTime;
    cudaEvent_t totalStart, totalStop;
    // create event
    checkCudaErrors(cudaEventCreate(&totalStart));
    checkCudaErrors(cudaEventCreate(&totalStop));
    checkCudaErrors(cudaEventRecord(totalStart));

    // compute step execution time
    float stepTime;
    cudaEvent_t stepStart, stepStop;
    // create event
    checkCudaErrors(cudaEventCreate(&stepStart));
    checkCudaErrors(cudaEventCreate(&stepStop));

    int matrixSize = nativeBlockNum * nativeBlockNum * sizeof(uint8_t);
    uint8_t *decodingMatrix_d;	//device
    checkCudaErrors(cudaMalloc((void **) &decodingMatrix_d, matrixSize));

    // record event
    checkCudaErrors(cudaEventRecord(stepStart));
    checkCudaErrors(cudaMemcpy(decodingMatrix_d, decodingMatrix, matrixSize, cudaMemcpyHostToDevice));
    // record event and synchronize
    checkCudaErrors(cudaEventRecord(stepStop));
    checkCudaErrors(cudaEventSynchronize(stepStop));
    // get event elapsed time
    checkCudaErrors(cudaEventElapsedTime(&stepTime, stepStart, stepStop));
    printf("Device%d: Copy decoding matrix from CPU to GPU: %fms\n", id, stepTime);
    totalCommunicationTime += stepTime;

    // NOTE: use CUDA stream to decode the file
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
        int codeSize = nativeBlockNum * streamChunkSize * sizeof(uint8_t);

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
            checkCudaErrors(cudaMemcpyAsync(codeBuf_d[i] + j * streamChunkSize,
                    codeBuf + j * chunkSize + i * streamMinChunkSize,
                    streamChunkSize * sizeof(uint8_t),
                    cudaMemcpyHostToDevice,
                    stream[i]));
        }

        stepTime = decode_chunk(dataBuf_d[i], decodingMatrix_d, codeBuf_d[i], nativeBlockNum, parityBlockNum, streamChunkSize, gridDimXSize, stream[i]);

        for (int j = 0; j < nativeBlockNum; j++)
        {
            checkCudaErrors(cudaMemcpyAsync(dataBuf + j * chunkSize + i * streamMinChunkSize,
                    dataBuf_d[i] + j * streamChunkSize,
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
    checkCudaErrors(cudaFree(decodingMatrix_d));

    // record event and synchronize
    checkCudaErrors(cudaEventRecord(totalStop));
    checkCudaErrors(cudaEventSynchronize(totalStop));
    // get event elapsed time
    checkCudaErrors(cudaEventElapsedTime(&totalTime, totalStart, totalStop));
    printf("Device%d: Total GPU decoding time: %fms\n", id, totalTime);

    for(int i = 0; i < streamNum; i++)
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

    decode(thread_data->dataBuf,
            thread_data->codeBuf,
            thread_data->decodingMatrix,
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
        printf("Total GPU decoding time using multiple devices: %fms\n", totalTime);
    }
    return NULL;
}

/* 
 * ===  FUNCTION  ======================================================================
 *         Name:  decode_file
 *  Description:  decode the original input file <fileName> with the given settings
 * =====================================================================================
 */
extern "C"
void decode_data(uint8_t *generatorMatrix, uint8_t **data, uint8_t **parity, int *missing, int nativeBlockNum, int parityBlockNum, int chunkSize, int gridDimXSize, int streamNum)
{

    int matrixSize;
    uint8_t *encodingMatrix;	//host
    uint8_t **availableData;

    matrixSize = nativeBlockNum * nativeBlockNum;
    encodingMatrix = (uint8_t*) malloc(matrixSize);
    availableData = (uint8_t**) malloc(sizeof(uint8_t*) * nativeBlockNum);
    // NOTE: Pinned host memory is expensive for allocation,
    // so pageable host memory is used here.

    int i = 0;

    for (int index = 0; index < nativeBlockNum + parityBlockNum && i < nativeBlockNum; index++)
    {
        if(!missing[index]) {
            availableData[i] = (index < nativeBlockNum)? data[index]: parity[index - nativeBlockNum];
            copy_matrix(generatorMatrix, encodingMatrix, index, i, nativeBlockNum);
            i++;
        }
    }

    cudaDeviceProp deviceProperties;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProperties, 0));
    int maxGridDimXSize = min(deviceProperties.maxGridSize[0], deviceProperties.maxGridSize[1]);
    if (gridDimXSize > maxGridDimXSize || gridDimXSize <= 0)
    {
        printf("Valid grid size: (0, %d]\n", maxGridDimXSize);
        gridDimXSize = maxGridDimXSize;
    }

    uint8_t *decodingMatrix;
    // Pageable Host Memory is preferred here since the decodingMatrix is small
    decodingMatrix = (uint8_t*) malloc(matrixSize);
    CPU_invert_matrix(encodingMatrix, decodingMatrix, nativeBlockNum);

    int GPU_num;
    checkCudaErrors(cudaGetDeviceCount(&GPU_num));

    void* threads = malloc(GPU_num * sizeof(pthread_t));
    ThreadDataType* thread_data = (ThreadDataType *) malloc(GPU_num * sizeof(ThreadDataType));

    uint8_t *dataBufPerDevice[GPU_num];
    uint8_t *codeBufPerDevice[GPU_num];
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
        int deviceCodeSize = nativeBlockNum * deviceChunkSize * sizeof(uint8_t);
        checkCudaErrors(cudaMallocHost((void **)&dataBufPerDevice[i], deviceDataSize));
        checkCudaErrors(cudaMallocHost((void **)&codeBufPerDevice[i], deviceCodeSize));
        for (int j = 0; j < nativeBlockNum; ++j)
        {
            // Pinned Host Memory
            checkCudaErrors(cudaMemcpy(codeBufPerDevice[i] + j * deviceChunkSize,
                    availableData[j] + i * minChunkSizePerDevice,
                    deviceChunkSize,
                    cudaMemcpyHostToHost));
        }
        thread_data[i].dataBuf = dataBufPerDevice[i];
        thread_data[i].codeBuf = codeBufPerDevice[i];
        thread_data[i].decodingMatrix = decodingMatrix;

        pthread_create(&((pthread_t*) threads)[i], NULL, GPU_thread_func, (void *) &thread_data[i]);
    }

    for (int i = 0; i < GPU_num; ++i)
    {
        pthread_join(((pthread_t*) threads)[i], NULL);
    }

    for (int i = 0; i < GPU_num; ++i)
    {
        int deviceChunkSize = minChunkSizePerDevice;
        if (i == GPU_num - 1)
        {
            deviceChunkSize = chunkSize - i * minChunkSizePerDevice;
        }

        for (int j = 0; j < nativeBlockNum; ++j)
        {
            // Pinned Host Memory
            if(missing[j]){
                checkCudaErrors(cudaMemcpy(data[j] + i * minChunkSizePerDevice,
                    dataBufPerDevice[i] + j * deviceChunkSize,
                    deviceChunkSize,
                    cudaMemcpyHostToHost));
            }
            
        }

        // Pinned Host Memory
        checkCudaErrors(cudaFreeHost(dataBufPerDevice[i]));
        checkCudaErrors(cudaFreeHost(codeBufPerDevice[i]));
    }

    pthread_barrier_destroy(&barrier);
    checkCudaErrors(cudaDeviceReset());

    // NOTE: Pinned host memory is expensive for deallocation,
    // so pageable host memory is used here.
    free(decodingMatrix);
    free(encodingMatrix);
    free(availableData);
}
