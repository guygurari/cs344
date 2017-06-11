/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/


//#include "reference_calc.cpp"
#include "utils.h"
#include <float.h>
#include <stdio.h>

/*
 * Find the minimum and maximum of d_logLuminance using reduce.
 * Step 1: Each block reduces a data of length blockWidth, and writes
 *         it to global memory.
 * Shared memory: should be array of floats of length 2*blockWidth.
 */
__global__ void min_max_kernel(const float* const d_logLuminance,
                            const size_t length,
                            float* d_min_logLum,
                            float* d_max_logLum) {
    // Shared working memory
    extern __shared__ float sh_logLuminance[];

    int blockWidth = blockDim.x;
    int x = blockDim.x * blockIdx.x + threadIdx.x;

    float* min_logLuminance = sh_logLuminance;
    float* max_logLuminance = sh_logLuminance + blockWidth;

    // Copy this block's chunk of the data to shared memory
    // We copy twice so we compute min and max at the same time
    if (x < length) {
        min_logLuminance[threadIdx.x] = d_logLuminance[x];
        max_logLuminance[threadIdx.x] = min_logLuminance[threadIdx.x];
    }
    else {
        // Pad if we're out of range
        min_logLuminance[threadIdx.x] =  FLT_MAX;
        max_logLuminance[threadIdx.x] = -FLT_MAX;
    }

    __syncthreads();

    // Reduce
    for (int s = blockWidth/2; s > 0; s /= 2) {
        if (threadIdx.x < s) {
            if (min_logLuminance[threadIdx.x + s] < min_logLuminance[threadIdx.x]) {
                min_logLuminance[threadIdx.x] = min_logLuminance[threadIdx.x + s];
            }

            if (max_logLuminance[threadIdx.x + s] > max_logLuminance[threadIdx.x]) {
                max_logLuminance[threadIdx.x] = max_logLuminance[threadIdx.x + s];
            }

            // Same speed
            /*min_logLuminance[threadIdx.x] = fmin(min_logLuminance[threadIdx.x],
                                                 min_logLuminance[threadIdx.x + s]);
            max_logLuminance[threadIdx.x] = fmax(max_logLuminance[threadIdx.x],
            max_logLuminance[threadIdx.x + s]);*/
        }

        __syncthreads();
    }

    // Write to global memory
    if (threadIdx.x == 0) {
        d_min_logLum[blockIdx.x] = min_logLuminance[0];
        d_max_logLum[blockIdx.x] = max_logLuminance[0];
    }
}

size_t get_num_blocks(size_t inputLength, size_t threadsPerBlock) {
    return inputLength / threadsPerBlock +
        ((inputLength % threadsPerBlock == 0) ? 0 : 1);
}

/*
* Compute min, max over the data by first reducing on the device, then
* doing the final reducation on the host.
*/
void compute_min_max(const float* const d_logLuminance,
                    float& min_logLum,
                    float& max_logLum,
                    const size_t numPixels) {
    // Compute min, max
    //printf("\n=== computing min/max ===\n");
    const size_t blockWidth = 1024;
    size_t numBlocks = get_num_blocks(numPixels, blockWidth);

    //printf("Num min/max blocks = %d\n", numBlocks);

    float* d_min_logLum;
    float* d_max_logLum;
    int alloc_size = sizeof(float) * numBlocks;
    checkCudaErrors(cudaMalloc(&d_min_logLum, alloc_size));
    checkCudaErrors(cudaMalloc(&d_max_logLum, alloc_size));

    min_max_kernel<<<numBlocks, blockWidth, sizeof(float) * blockWidth * 2>>>
        (d_logLuminance, numPixels, d_min_logLum, d_max_logLum);

    float* h_min_logLum = (float*) malloc(alloc_size);
    float* h_max_logLum = (float*) malloc(alloc_size);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaMemcpy(h_min_logLum, d_min_logLum, alloc_size, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(h_max_logLum, d_max_logLum, alloc_size, cudaMemcpyDeviceToHost));

    min_logLum = FLT_MAX;
    max_logLum = -FLT_MAX;

    // Reduce over the block results
    // (would be a bit faster to do it on the GPU, but it's just 96 numbers)
    for (int i = 0; i < numBlocks; i++) {
        if (h_min_logLum[i] < min_logLum) {
            min_logLum = h_min_logLum[i];
        }
        if (h_max_logLum[i] > max_logLum) {
            max_logLum = h_max_logLum[i];
        }
    }

    //printf("min_logLum = %.2f\nmax_logLum = %.2f\n", min_logLum, max_logLum);

    checkCudaErrors(cudaFree(d_min_logLum));
    checkCudaErrors(cudaFree(d_max_logLum));
    free(h_min_logLum);
    free(h_max_logLum);
}

void compute_min_max_on_host(const float* const d_logLuminance, size_t numPixels) {
    int alloc_size = sizeof(float) * numPixels;
    float* h_logLuminance = (float*) malloc(alloc_size);
    checkCudaErrors(cudaMemcpy(h_logLuminance, d_logLuminance, alloc_size, cudaMemcpyDeviceToHost));
    float host_min_logLum = FLT_MAX;
    float host_max_logLum = -FLT_MAX;
    for (int i = 0; i < numPixels; i++) {
        if (h_logLuminance[i] < host_min_logLum) {
            host_min_logLum = h_logLuminance[i];
        }
        if (h_logLuminance[i] > host_max_logLum) {
            host_max_logLum = h_logLuminance[i];
        }
    }
    printf("host_min_logLum = %.2f\nhost_max_logLum = %.2f\n",
           host_min_logLum, host_max_logLum);
    free(h_logLuminance);
}

__global__ void histogram_kernel(const float* const d_logLuminance,
                                 int* d_histogram,
                                 const float min_logLum,
                                 const float lumRange,
                                 const size_t numBins/*,
                                                       const size_t pixelsPerThread*/) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int bin = (d_logLuminance[x] - min_logLum) / lumRange * numBins;

    if (bin > numBins - 1) {
        bin = numBins - 1;
    }

    atomicAdd(&d_histogram[bin], 1);
}

int* compute_histogram(const float* const d_logLuminance,
                       const float min_logLum,
                       const float max_logLum,
                       const size_t numPixels,
                       const size_t numBins) {
    printf("\n=== computing histogram ===\n");
    float lumRange = max_logLum - min_logLum;
    printf("min_logLum=%.2f range=%.2f numBins=%d\n", min_logLum, lumRange, numBins);

    int* d_histogram;
    int alloc_size = sizeof(int) * numBins;
    checkCudaErrors(cudaMalloc((void**) &d_histogram, alloc_size));
    checkCudaErrors(cudaMemset((void*) d_histogram, 0, alloc_size));

    const size_t blockWidth = 1024;
    size_t numBlocks = get_num_blocks(numPixels, blockWidth);

    histogram_kernel<<<numBlocks, blockWidth>>>(d_logLuminance,
                                                d_histogram,
                                                min_logLum,
                                                lumRange,
                                                numBins);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // Chekck
    /*int* h_histogram = (int*) malloc(alloc_size);
    checkCudaErrors(cudaMemcpy(h_histogram, d_histogram, alloc_size, cudaMemcpyDeviceToHost));
    int n = 0;
    for (int i = 0; i < numBins; i++) {
        n += h_histogram[i];
    }
    printf("Total number of histogram entries: %d\nNumber of pixels: %d\n", n, numPixels);
    free(h_histogram);*/

    return d_histogram;
}

/*
 * Use Hillis-Steele to scan.
 */
__global__ void cdf_kernel(const int* d_histogram, 
                           unsigned int* const d_cdf,
                           const size_t numBins) {
    extern __shared__ int sh_histogram[];
    int x = threadIdx.x;

    // Copy histogram to shared memory
    sh_histogram[x] = d_histogram[x];

    __syncthreads();

    for (int s = 1; s < numBins; s *= 2) {
        // Read the value at x-s
        int value = (x-s >= 0) ? sh_histogram[x-s] : 0;
        __syncthreads();

        // Add the value at x-s to x
        sh_histogram[x] += value;
        __syncthreads();
    }

    // Copy to d_cdf, converting to an exclusive scan
    if (x == numBins - 1) {
        d_cdf[0] = 0;
    }
    else {
        d_cdf[x+1] = sh_histogram[x];
    }
}

void compute_cdf(const int* d_histogram, 
                 unsigned int* const d_cdf,
                 const size_t numBins) {
    cdf_kernel<<<1, numBins, sizeof(int) * numBins>>>
        (d_histogram, d_cdf, numBins);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}

#define TEST_SIZE 10
void test_compute_cdf() {
    int h_data[TEST_SIZE];

    for (int i = 0; i < TEST_SIZE; i++) h_data[i] = i+1;

    int* d_data;
    int alloc_size = sizeof(int) * TEST_SIZE;
    checkCudaErrors(cudaMalloc((void**) &d_data, alloc_size));
    checkCudaErrors(cudaMemcpy(d_data, h_data, alloc_size, cudaMemcpyHostToDevice));

    int* d_scan;
    checkCudaErrors(cudaMalloc((void**) &d_scan, alloc_size));

    cdf_kernel<<<1, TEST_SIZE, alloc_size>>>(d_data, (unsigned int*) d_scan, TEST_SIZE);

    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    int h_scan[TEST_SIZE];
    checkCudaErrors(cudaMemcpy(h_scan, d_scan, alloc_size, cudaMemcpyDeviceToHost));

    printf("Test: ");
    for (int i = 0; i < TEST_SIZE; i++) printf("%d : %d\t", i, h_scan[i]);
    printf("\n");
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float& min_logLum,
                                  float& max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
    /*printf("numRows=%d\nnumCols=%d\nnumRows*numCols=%d\nnumBins=%d\n",
      numRows, numCols, numRows*numCols, numBins);*/
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

    const size_t numPixels = numRows * numCols;
    compute_min_max(d_logLuminance, min_logLum, max_logLum, numPixels);

    // Find min/max on host for checking
    //compute_min_max_on_host(d_logLuminance, numPixels);

    const int* d_histogram = compute_histogram(d_logLuminance,
                                               min_logLum, max_logLum,
                                               numPixels,
                                               numBins);

    //test_compute_cdf();

    compute_cdf(d_histogram, d_cdf, numBins);

    checkCudaErrors(cudaFree((void*) d_histogram));
}
