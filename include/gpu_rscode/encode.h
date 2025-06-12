/*
 * =====================================================================================
 *
 *       Filename:  encode.h
 *
 *    Description:  
 *
 *        Version:  1.0
 *        Created:  12/25/2012 04:33:06 PM
 *       Revision:  none
 *       Compiler:  nvcc 
 *
 *         Author:  Shuai YUAN (yszheda AT gmail.com), 
 *        Company:  
 *
 * =====================================================================================
 */
#ifndef _ENCODE_H_
#define _ENCODE_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif
void GPU_generate_encode_matrix(uint8_t *encodingMatrix, int nativeBlockNum, int parityBlockNum);
void encode_data(uint8_t *generator_matrix, uint8_t **dataBuf, uint8_t **codeBuf, int nativeBlockNum, int parityBlockNum, int chunkSize, int gridDimXSize, int streamNum);
#ifdef __cplusplus
}
#endif


#endif
