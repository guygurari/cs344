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

#define BLOCK_WIDTH 1024

#define BITS_PER_DIGIT 4 // 16 possible values per digit
#define VALUES_PER_DIGIT (1 << BITS_PER_DIGIT)
#define DIGIT_MASK ((1 << BITS_PER_DIGIT) - 1)

#define HISTOGRAM_ELEMS_PER_THREAD 1024
    
// Compute a histogram of the number of occurences of every possible value
// of the digit
__global__ void histogramKernel(const uint* const d_inputVals,
                                const size_t numElems,
                                const uint digit,
                                uint* const d_threadHistograms,
                                const uint numThreads) {

    //extern __shared__ uint sharedHistograms[];
        
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

    // Write local histogram to global mem
    for (int i = 0; i < VALUES_PER_DIGIT; i++) {
        //sharedHistograms[threadIdx.x * VALUES_PER_DIGIT + i] = localHistogram[i];

        // First write all the histograms of digitValue=0
        // Then histograms of digitValue=1
        // etc.
        d_threadHistograms[i * numThreads + absThreadIdx] = localHistogram[i];
    }
}

// Add-scan the given input with Hillis-Steele, then convert to exclusive scan.
// Run in a single block so we can synchronize.
__global__ void exclusiveScanKernel(uint* const d_input,
                                    const uint len,
                                    const uint numReduceThreads) {
    // How many elements should this thread process
    const uint lenToProcess =
        len / numReduceThreads + (len % numReduceThreads == 0 ? 0 : 1);

    /*if (threadIdx.x == 0) {
        printf("len = %d\n", (int) len);
        printf("numReduceThreads = %d\n", (int) numReduceThreads);
        printf("lenToProcess = %d\n", (int) lenToProcess);
        }*/

    const uint threadOffset = threadIdx.x * lenToProcess;

    const uint workspaceLen = 512;

    if (workspaceLen < lenToProcess) {
        printf("ERROR: Workspace isn't large enough!\n");
        return;
    }

    // Inclusive scan
    for (int s = 1; s < len; s *= 2) {
        uint values[workspaceLen];
        
        for (int i = threadOffset; i < threadOffset + lenToProcess; i++) {
            if (i < len) {
                values[i-threadOffset] = (i-s >= 0) ? d_input[i-s] : 0;
            }
        }

        __syncthreads();

        for (uint i = threadOffset; i < threadOffset + lenToProcess; i++) {
            if (i < len) {
                d_input[i] += values[i-threadOffset];
            }
        }

        __syncthreads();
    }

    // Turn into exclusive scan
    uint values[workspaceLen];

    for (int i = threadOffset; i < threadOffset + lenToProcess; i++) {
        if (i < len) {
            values[i-threadOffset] = (i-1 >= 0) ? d_input[i-1] : 0;
        }
    }

    __syncthreads();

    for (int i = threadOffset; i < threadOffset + lenToProcess; i++) {
        if (i < len) {
            d_input[i] = values[i-threadOffset];
        }
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

// Write the elements of the subsequence to their proper output positions
// as indicated by d_threadOffsets. Works on the same data subsets as 
// histogramKernel.
__global__ void rearrangeKernel(uint* const d_inputVals,
                                uint* const d_inputPos,
                                uint* const d_outputVals,
                                uint* const d_outputPos,
                                const size_t numElems,
                                const uint* const d_threadOffsets,
                                const uint numThreads,
                                const uint digit) {

    const size_t absThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint digitShift = BITS_PER_DIGIT * digit;

    // Read this thread's initial offsets (offset per digit value)
    uint offsets[VALUES_PER_DIGIT];

    for (uint i = 0; i < VALUES_PER_DIGIT; i++) {
        offsets[i] = d_threadOffsets[i * numThreads + absThreadIdx];
    }

    // Offset into the input array
    const uint threadInputOffset = absThreadIdx * HISTOGRAM_ELEMS_PER_THREAD;

    for (int i = threadInputOffset;
         i < threadInputOffset + HISTOGRAM_ELEMS_PER_THREAD;
         i++) {

        if (i < numElems) {
            uint digitValue = (d_inputVals[i] >> digitShift) & DIGIT_MASK;
            d_outputVals[offsets[digitValue]] = d_inputVals[i];
            d_outputPos[offsets[digitValue]] = d_inputPos[i];
            offsets[digitValue]++;
        }
    }
}

void radix_sort_pass(uint* const d_inputVals,
                     uint* const d_inputPos,
                     uint* const d_outputVals,
                     uint* const d_outputPos,
                     const size_t numElems,
                     uint digit,
                     size_t numHistBlocks,
                     uint* d_threadHistograms) {
    //int sharedMemSize = sizeof(uint) * VALUES_PER_DIGIT * BLOCK_WIDTH;
    //printf("Allocating %d bytes of shared memory\n", sharedMemSize);

    const uint numHistThreads = numHistBlocks * BLOCK_WIDTH;

    //printf("Launching Histogram\n"); fflush(stdout);
    histogramKernel<<<numHistBlocks, BLOCK_WIDTH>>>
        (d_inputVals, numElems, digit, d_threadHistograms, numHistThreads);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    //printf("Launching Scan\n"); fflush(stdout);
    const int numReduceThreads = 1024;
    exclusiveScanKernel<<<1, numReduceThreads>>>(d_threadHistograms,
                                                 numHistThreads * VALUES_PER_DIGIT,
                                                 numReduceThreads);

        
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    const uint* const d_threadOffsets = d_threadHistograms;

    // Rearrange values according to offsets
    rearrangeKernel<<<numHistBlocks, BLOCK_WIDTH>>>(d_inputVals,
                                                    d_inputPos,
                                                    d_outputVals,
                                                    d_outputPos,
                                                    numElems,
                                                    d_threadOffsets,
                                                    numHistThreads,
                                                    digit);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

/*
 * This implementation is very slow. The last rearrangement step has one thread
 * pre digit value, and each thread has to loop over the whole input.
 * We can parallelize as follows: instead of computing one big histogram for the
 * whole input, we should keep the histograms for each subsequence. So we have a 2D
 * array of histograms: per digit value and per subsequence. Then the offsets (the
 * prefix sums) should again be computed per subsequence. So each subsequence knows
 * when all the previous subsequences ended, and can output its elements independently.
 */
void your_sort(uint* const d_inputVals,
               uint* const d_inputPos,
               uint* const d_outputVals,
               uint* const d_outputPos,
               const size_t numElems)
{ 
    printf("numElems = %d\n", (int) numElems);

    size_t numHistThreads = get_enough_processors(numElems, HISTOGRAM_ELEMS_PER_THREAD);
    size_t numHistBlocks = get_enough_processors(numHistThreads, BLOCK_WIDTH);
    numHistThreads = numHistBlocks * BLOCK_WIDTH;

    printf("Histogram: %d threads, %d blocks\n",
           (int) numHistThreads,
           (int) numHistBlocks);

    uint* d_threadHistograms;
    const int threadHistsSize = sizeof(uint) * VALUES_PER_DIGIT * numHistThreads;
    printf("Allocating %d bytes of global memory\n", threadHistsSize);
    checkCudaErrors(cudaMalloc((void**) &d_threadHistograms, threadHistsSize));

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
                        numHistBlocks,
                        d_threadHistograms);

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

    checkCudaErrors(cudaFree(d_threadHistograms));
    //checkCudaErrors(cudaFree(d_digitOffsets));
    //free(h_blockHistograms);
}
