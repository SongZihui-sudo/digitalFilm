#ifndef TRILINEAR_KERNEL_H
#define TRILINEAR_KERNEL_H

#include <cuda_runtime.h>

__global__ void triLinearForwardKernel( const int nthreads,
                                  const float* lut,
                                  const float* image,
                                  float* output,
                                  const int dim,
                                  const int shift,
                                  const float binsize,
                                  const int width,
                                  const int height,
                                  const int batch );

int triLinearForwardLaucher( const float* lut,
                             const float* image,
                             float* output,
                             const int lut_dim,
                             const int shift,
                             const float binsize,
                             const int width,
                             const int height,
                             const int batch,
                             cudaStream_t stream );

__global__ void triLinearBackwardKernel( const int nthreads,
                                   const float* image,
                                   const float* image_grad,
                                   float* lut_grad,
                                   const int dim,
                                   const int shift,
                                   const float binsize,
                                   const int width,
                                   const int height,
                                   const int batch );

int triLinearBackwardLaucher( const float* image,
                              const float* image_grad,
                              float* lut_grad,
                              const int lut_dim,
                              const int shift,
                              const float binsize,
                              const int width,
                              const int height,
                              const int batch,
                              cudaStream_t stream );

#endif
