//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */

#include <stdio.h>

typedef unsigned int uint;

#define BLOCK_WIDTH 256

#define BITS_PER_DIGIT 4 // 16 possible values per digit
#define VALUES_PER_DIGIT (1 << BITS_PER_DIGIT)
#define DIGIT_MASK ((1 << BITS_PER_DIGIT) - 1)

#define HISTOGRAM_ELEMS_PER_THREAD 255

// Static variable doesn't work with cudaMemcpy... weird
//__device__ uint d_digitOffsets[VALUES_PER_DIGIT];
    
// Compute a histogram of the number of occurences of every possible value
// of the digit
__global__ void histogramKernel(uint* const d_inputVals,
                                const size_t numElems,
                                const uint digit,
                                uint* d_blockHistograms) {
    extern __shared__ uint sharedHistograms[];
        
    const size_t absThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint digitShift = BITS_PER_DIGIT * digit;

    // Compute local histogram
    uint localHistogram[VALUES_PER_DIGIT];

    for (int i = 0; i < VALUES_PER_DIGIT; i++) {
        localHistogram[i] = 0;
    }

    // Offset into the input array
    const uint threadInputOffset = absThreadIdx * HISTOGRAM_ELEMS_PER_THREAD;

    for (size_t i = 0; i < HISTOGRAM_ELEMS_PER_THREAD; i++) {
        int inputOffset = threadInputOffset + i;

        if (inputOffset < numElems) {
            uint digitValue = (d_inputVals[inputOffset] >> digitShift) & DIGIT_MASK;
            localHistogram[digitValue]++;
        }
    }

    // Write local histogram to shared mem
    for (int i = 0; i < VALUES_PER_DIGIT; i++) {
        sharedHistograms[threadIdx.x * VALUES_PER_DIGIT + i] = localHistogram[i];
    }

    __syncthreads();

    // Reduce shared histograms to a single one
    // (Slow due to lots of branching, oh well)
    for (int s = blockDim.x/2; s > 0; s /= 2) {
        // TODO parallelize the for (i=...) loop
        if (threadIdx.x < s) {
            for (int i = 0; i < VALUES_PER_DIGIT; i++) {
                sharedHistograms[threadIdx.x * VALUES_PER_DIGIT + i] +=
                    sharedHistograms[(threadIdx.x+s) * VALUES_PER_DIGIT + i];
            }
        }

        __syncthreads();
    }

    // Write block histogram to global memory
    if (threadIdx.x < VALUES_PER_DIGIT) {
        d_blockHistograms[blockIdx.x * VALUES_PER_DIGIT + threadIdx.x] = 
            sharedHistograms[threadIdx.x];
    }
}

// The number of processors needed to handle totalElems elements, given
// that each processor handles elemsPerProcessor elements.
int get_enough_processors(int totalElems, int elemsPerProcessor) {
    return totalElems / elemsPerProcessor +
        (totalElems % elemsPerProcessor == 0 ? 0 : 1);
}

void reference_histogram(uint* const d_inputVals,
                         const size_t numElems,
                         const uint digit) {
    uint hist[VALUES_PER_DIGIT];

    size_t allocSize = sizeof(uint) * numElems;
    uint* h_inputVals = (uint*) malloc(allocSize);
    checkCudaErrors(cudaMemcpy(h_inputVals,
                               d_inputVals,
                               allocSize,
                               cudaMemcpyDeviceToHost));

    for (int i = 0; i < VALUES_PER_DIGIT; i++) {
        hist[i] = 0;
    }

    for (size_t i = 0; i < numElems; i++) {
        uint digitValue = (h_inputVals[i] >> (BITS_PER_DIGIT * digit)) & DIGIT_MASK;
        hist[digitValue]++;
    }

    printf("\nReference histogram for digit=%d :\n", digit);
    int total = 0;
    for (int i = 0; i < VALUES_PER_DIGIT; i++) {
        printf("%02x : %d\n", i, hist[i]);
        total += hist[i];
    }
    printf("Total: %d\n", total);
    printf("\n");
}

__global__ void rearrangeKernel(uint* const d_inputVals,
                                uint* const d_inputPos,
                                uint* const d_outputVals,
                                uint* const d_outputPos,
                                const size_t numElems,
                                uint* const d_digitOffsets,
                                const uint digit) {
    uint targetDigitValue = threadIdx.x;
    uint offset = d_digitOffsets[targetDigitValue];

    for (size_t i = 0; i < numElems; i++) {
        const uint digitShift = BITS_PER_DIGIT * digit;
        const uint digitValue = (d_inputVals[i] >> digitShift) & DIGIT_MASK;

        if (digitValue == targetDigitValue) {
            d_outputVals[offset] = d_inputVals[i];
            d_outputPos[offset] = d_inputPos[i];
            offset++;
        }
    }

    /*if (targetDigitValue < 6) {
        printf("Thread %d final offset: %u vs. %u\n",
               threadIdx.x, offset, d_digitOffsets[targetDigitValue+1]);
               }*/
}

void radix_sort_pass(uint* const d_inputVals,
                     uint* const d_inputPos,
                     uint* const d_outputVals,
                     uint* const d_outputPos,
                     const size_t numElems,
                     uint digit,
                     size_t numHistThreads,
                     size_t numHistBlocks,
                     uint* d_blockHistograms,
                     uint* h_blockHistograms,
                     uint* d_digitOffsets) {
    int sharedMemSize = sizeof(uint) * VALUES_PER_DIGIT * BLOCK_WIDTH;
    //printf("Allocating %d bytes of shared memory\n", sharedMemSize);

    histogramKernel<<<numHistBlocks, BLOCK_WIDTH, sharedMemSize>>>
        (d_inputVals, numElems, digit, d_blockHistograms);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    const int blockHistSize = sizeof(uint) * VALUES_PER_DIGIT * numHistBlocks;
    checkCudaErrors(cudaMemcpy(h_blockHistograms,
                               d_blockHistograms,
                               blockHistSize,
                               cudaMemcpyDeviceToHost));

    uint histogram[VALUES_PER_DIGIT];
    memset(histogram, 0, sizeof(uint) * VALUES_PER_DIGIT);

    for (size_t i = 0; i < numHistBlocks; i++) {
        for (size_t j = 0; j < VALUES_PER_DIGIT; j++) {
            histogram[j] += h_blockHistograms[i * VALUES_PER_DIGIT + j];
        }
    }

    /*int total = 0;
    for (int i = 0; i < VALUES_PER_DIGIT; i++) {
        printf("%02x : %d\n", i, histogram[i]);
        total += histogram[i];
    }
    printf("Total: %d\n", total);
    reference_histogram(d_inputVals, numElems, digit);*/

    // Compute histogram prefix sum. It's a small histogram so we do it on the host.
    uint h_digitOffsets[VALUES_PER_DIGIT];
    h_digitOffsets[0] = 0;

    for (int i = 1; i < VALUES_PER_DIGIT; i++) {
        h_digitOffsets[i] = h_digitOffsets[i-1] + histogram[i-1];
    }

    checkCudaErrors(cudaMemcpy(d_digitOffsets,
                               h_digitOffsets,
                               sizeof(uint) * VALUES_PER_DIGIT,
                               cudaMemcpyHostToDevice));

    // Rearrange values according to offsets
    rearrangeKernel<<<1, VALUES_PER_DIGIT>>>(d_inputVals,
                                             d_inputPos,
                                             d_outputVals,
                                             d_outputPos,
                                             numElems,
                                             d_digitOffsets,
                                             digit);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

void your_sort(uint* const d_inputVals,
               uint* const d_inputPos,
               uint* const d_outputVals,
               uint* const d_outputPos,
               const size_t numElems)
{ 
    printf("numElems = %d\n", (int) numElems);

    size_t numHistThreads = get_enough_processors(numElems, HISTOGRAM_ELEMS_PER_THREAD);
    size_t numHistBlocks = get_enough_processors(numHistThreads, BLOCK_WIDTH);

    printf("Histogram: %d threads, %d blocks\n",
           (int) numHistThreads,
           (int) numHistBlocks);

    uint* d_blockHistograms;
    uint* h_blockHistograms;
    const int blockHistSize = sizeof(uint) * VALUES_PER_DIGIT * numHistBlocks;
    checkCudaErrors(cudaMalloc((void**) &d_blockHistograms, blockHistSize));
    h_blockHistograms = (uint*) malloc(blockHistSize);

    uint* d_digitOffsets;
    checkCudaErrors(cudaMalloc((void**) &d_digitOffsets,
                               sizeof(uint) * VALUES_PER_DIGIT));

    assert(sizeof(uint) % BITS_PER_DIGIT == 0);
    uint maxDigit = sizeof(uint) * 8 / BITS_PER_DIGIT;

    uint* d_myInputVals = d_inputVals;
    uint* d_myInputPos = d_inputPos;
    uint* d_myOutputVals = d_outputVals;
    uint* d_myOutputPos = d_outputPos;

    for (uint digit = 0; digit < maxDigit; digit++) {
        radix_sort_pass(d_myInputVals,
                        d_myInputPos,
                        d_myOutputVals,
                        d_myOutputPos,
                        numElems,
                        digit,
                        numHistThreads,
                        numHistBlocks,
                        d_blockHistograms,
                        h_blockHistograms,
                        d_digitOffsets);

        // Swap buffers
        if (digit < maxDigit - 1) {
            uint* tmpVals = d_myInputVals;
            uint* tmpPos = d_myInputPos;
            d_myInputVals = d_myOutputVals;
            d_myInputPos = d_myOutputPos;
            d_myOutputVals = tmpVals;
            d_myOutputPos = tmpPos;
        }
    }

    if (d_myOutputVals != d_outputVals) {
        // Last output buffer was the original input, so we need to copy
        size_t len = sizeof(uint) * numElems;
        checkCudaErrors(cudaMemcpy(d_outputVals, d_inputVals,
                                   len, cudaMemcpyDeviceToDevice));
        checkCudaErrors(cudaMemcpy(d_outputPos, d_inputPos,
                                   len, cudaMemcpyDeviceToDevice));
    }

    size_t len = sizeof(uint) * numElems;
    uint* h_outputVals = (uint*) malloc(len);
    uint* h_inputVals = (uint*) malloc(len);
    checkCudaErrors(cudaMemcpy(h_outputVals, d_outputVals, len, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_inputVals, d_inputVals, len, cudaMemcpyDeviceToHost));

    /*printf("First few inputs:\n");
    for (int i = 0; i < 5; i++) {
        printf("%u\t", h_inputVals[i]);
    }
    printf(" ...\n");

    printf("First few outputs:\n");
    for (int i = 0; i < 5; i++) {
        printf("%u\t", h_outputVals[i]);
    }
    printf(" ...\n");

    printf("Last few outputs:\n");
    for (int i = 0; i < 5; i++) {
        printf("%u\t", h_outputVals[numElems - 1 - i]);
    }
    printf(" ...\n");*/

    checkCudaErrors(cudaFree(d_blockHistograms));
    checkCudaErrors(cudaFree(d_digitOffsets));
    free(h_blockHistograms);
}
