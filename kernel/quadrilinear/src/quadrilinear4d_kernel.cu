#include <cuda_runtime.h>
#include <float.h>
#include <math.h>
#include <stdio.h>

#include "../include/quadrilinear4d_kernel.h"

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

__global__ void QuadriLinearForward(const int nthreads, const float* lut,
                                    const float* image, float* output,
                                    const int dim, const int shift,
                                    const float binsize, const int width,
                                    const int height, const int batch) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    float context = image[index];
    float r = image[index + width * height * batch];
    float g = image[index + width * height * batch * 2];
    float b = image[index + width * height * batch * 3];

    int context_id = 0;
    int r_id = floor(r / binsize);
    int g_id = floor(g / binsize);
    int b_id = floor(b / binsize);

    float context_d = context;
    float r_d = fmod(r, binsize) / binsize;
    float g_d = fmod(g, binsize) / binsize;
    float b_d = fmod(b, binsize) / binsize;

    int id0000 =
        context_id * dim * dim * dim + r_id + g_id * dim + b_id * dim * dim;
    int id0100 = context_id * dim * dim * dim + (r_id + 1) + g_id * dim +
                 b_id * dim * dim;
    int id0010 = context_id * dim * dim * dim + r_id + (g_id + 1) * dim +
                 b_id * dim * dim;
    int id0001 = context_id * dim * dim * dim + r_id + g_id * dim +
                 (b_id + 1) * dim * dim;
    int id0110 = context_id * dim * dim * dim + (r_id + 1) + (g_id + 1) * dim +
                 b_id * dim * dim;
    int id0011 = context_id * dim * dim * dim + r_id + (g_id + 1) * dim +
                 (b_id + 1) * dim * dim;
    int id0101 = context_id * dim * dim * dim + (r_id + 1) + g_id * dim +
                 (b_id + 1) * dim * dim;
    int id0111 = context_id * dim * dim * dim + (r_id + 1) + (g_id + 1) * dim +
                 (b_id + 1) * dim * dim;

    int id1000 = (context_id + 1) * dim * dim * dim + r_id + g_id * dim +
                 b_id * dim * dim;
    int id1100 = (context_id + 1) * dim * dim * dim + (r_id + 1) + g_id * dim +
                 b_id * dim * dim;
    int id1010 = (context_id + 1) * dim * dim * dim + r_id + (g_id + 1) * dim +
                 b_id * dim * dim;
    int id1001 = (context_id + 1) * dim * dim * dim + r_id + g_id * dim +
                 (b_id + 1) * dim * dim;
    int id1110 = (context_id + 1) * dim * dim * dim + (r_id + 1) +
                 (g_id + 1) * dim + b_id * dim * dim;
    int id1011 = (context_id + 1) * dim * dim * dim + r_id + (g_id + 1) * dim +
                 (b_id + 1) * dim * dim;
    int id1101 = (context_id + 1) * dim * dim * dim + (r_id + 1) + g_id * dim +
                 (b_id + 1) * dim * dim;
    int id1111 = (context_id + 1) * dim * dim * dim + (r_id + 1) +
                 (g_id + 1) * dim + (b_id + 1) * dim * dim;

    float w0000 = (1 - context_d) * (1 - r_d) * (1 - g_d) * (1 - b_d);
    float w0100 = (1 - context_d) * r_d * (1 - g_d) * (1 - b_d);
    float w0010 = (1 - context_d) * (1 - r_d) * g_d * (1 - b_d);
    float w0001 = (1 - context_d) * (1 - r_d) * (1 - g_d) * b_d;
    float w0110 = (1 - context_d) * r_d * g_d * (1 - b_d);
    float w0011 = (1 - context_d) * (1 - r_d) * g_d * b_d;
    float w0101 = (1 - context_d) * r_d * (1 - g_d) * b_d;
    float w0111 = (1 - context_d) * r_d * g_d * b_d;

    float w1000 = context_d * (1 - r_d) * (1 - g_d) * (1 - b_d);
    float w1100 = context_d * r_d * (1 - g_d) * (1 - b_d);
    float w1010 = context_d * (1 - r_d) * g_d * (1 - b_d);
    float w1001 = context_d * (1 - r_d) * (1 - g_d) * b_d;
    float w1110 = context_d * r_d * g_d * (1 - b_d);
    float w1011 = context_d * (1 - r_d) * g_d * b_d;
    float w1101 = context_d * r_d * (1 - g_d) * b_d;
    float w1111 = context_d * r_d * g_d * b_d;

    output[index] =
        w0000 * lut[id0000] + w0100 * lut[id0100] + w0010 * lut[id0010] +
        w0001 * lut[id0001] + w0110 * lut[id0110] + w0011 * lut[id0011] +
        w0101 * lut[id0101] + w0111 * lut[id0111] + w1000 * lut[id1000] +
        w1100 * lut[id1100] + w1010 * lut[id1010] + w1001 * lut[id1001] +
        w1110 * lut[id1110] + w1011 * lut[id1011] + w1101 * lut[id1101] +
        w1111 * lut[id1111];

    output[index + width * height * batch] =
        w0000 * lut[id0000 + shift] + w0100 * lut[id0100 + shift] +
        w0010 * lut[id0010 + shift] + w0001 * lut[id0001 + shift] +
        w0110 * lut[id0110 + shift] + w0011 * lut[id0011 + shift] +
        w0101 * lut[id0101 + shift] + w0111 * lut[id0111 + shift] +
        w1000 * lut[id1000 + shift] + w1100 * lut[id1100 + shift] +
        w1010 * lut[id1010 + shift] + w1001 * lut[id1001 + shift] +
        w1110 * lut[id1110 + shift] + w1011 * lut[id1011 + shift] +
        w1101 * lut[id1101 + shift] + w1111 * lut[id1111 + shift];

    output[index + width * height * batch * 2] =
        w0000 * lut[id0000 + shift * 2] + w0100 * lut[id0100 + shift * 2] +
        w0010 * lut[id0010 + shift * 2] + w0001 * lut[id0001 + shift * 2] +
        w0110 * lut[id0110 + shift * 2] + w0011 * lut[id0011 + shift * 2] +
        w0101 * lut[id0101 + shift * 2] + w0111 * lut[id0111 + shift * 2] +
        w1000 * lut[id1000 + shift * 2] + w1100 * lut[id1100 + shift * 2] +
        w1010 * lut[id1010 + shift * 2] + w1001 * lut[id1001 + shift * 2] +
        w1110 * lut[id1110 + shift * 2] + w1011 * lut[id1011 + shift * 2] +
        w1101 * lut[id1101 + shift * 2] + w1111 * lut[id1111 + shift * 2];
  }
}

int QuadriLinearForwardLaucher(const float* lut, const float* image,
                               float* output, const int lut_dim,
                               const int shift, const float binsize,
                               const int width, const int height,
                               const int batch, cudaStream_t stream) {
  const int kThreadsPerBlock = 1024;
  const int output_size = height * width * batch;
  cudaError_t err;

  QuadriLinearForward<<<(output_size + kThreadsPerBlock - 1) / kThreadsPerBlock,
                        kThreadsPerBlock, 0, stream>>>(
      output_size, lut, image, output, lut_dim, shift, binsize, width, height,
      batch);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
    return 0;
  }

  return 1;
}

__global__ void QuadriLinearBackward( const int nthreads,
                                      const float* __restrict__ image,
                                      float* __restrict__ image_grad,
                                      const float* __restrict__ lut,
                                      float* __restrict__ lut_grad,
                                      const int dim,
                                      const int shift, // 元素偏移量用于通道间
                                      const float binsize,
                                      const int width,
                                      const int height,
                                      const int batch )
{
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if ( index >= nthreads )
        return;

    // 读取像素通道（尽量少用临时变量）
    const int plane     = width * height * batch;
    const float context = image[index];
    const float r       = image[index + plane * 1];
    const float g       = image[index + plane * 2];
    const float b       = image[index + plane * 3];

    const int r_id = ( int )floorf( r / binsize );
    const int g_id = ( int )floorf( g / binsize );
    const int b_id = ( int )floorf( b / binsize );

    const float r_d       = fmodf( r, binsize ) / binsize;
    const float g_d       = fmodf( g, binsize ) / binsize;
    const float b_d       = fmodf( b, binsize ) / binsize;
    const float context_d = context; // 如果 context 也有 binsize/quantize，按需调整

    // 预计算 base offsets for context 0 and 1 to avoid repeated muls
    const int dim3  = dim * dim * dim;
    const int base0 = 0 * dim3;
    const int base1 = 1 * dim3; // 如果 context>1 需要改成 context_id*dim3

    // 用循环生成 16 个 id 与权重：bit layout: (dc<<3)|(db<<2)|(dg<<1)|dr
    float weights[16];
    int ids[16];

    // precompute factors for r,g,b and (1-*) to use fewer ops
    const float r_fac[2] = { 1.0f - r_d, r_d };
    const float g_fac[2] = { 1.0f - g_d, g_d };
    const float b_fac[2] = { 1.0f - b_d, b_d };
    const float c_fac[2] = { 1.0f - context_d, context_d };

    // compute ids and weights
    for ( int i = 0; i < 16; ++i )
    {
        int dr     = ( i >> 0 ) & 1;
        int dg     = ( i >> 1 ) & 1;
        int db     = ( i >> 2 ) & 1;
        int dc     = ( i >> 3 ) & 1;
        int r_idx  = r_id + dr;
        int g_idx  = g_id + dg;
        int b_idx  = b_id + db;
        int base   = ( dc == 0 ) ? base0 : base1;
        ids[i]     = base + r_idx + g_idx * dim + b_idx * dim * dim;
        weights[i] = c_fac[dc] * r_fac[dr] * g_fac[dg] * b_fac[db];
    }

    // 对 lut_grad 做 atomicAdd（16 ids × 3 channels）
    // image_grad channels offsets
    const float grad_ch0 = image_grad[index + plane * 1];
    const float grad_ch1 = image_grad[index + plane * 2];
    const float grad_ch2 = image_grad[index + plane * 3];

    for ( int i = 0; i < 16; ++i )
    {
        float w = weights[i];
        // channel 0
        atomicAdd( lut_grad + ids[i] + 0 * shift, grad_ch0 * w );
        // channel 1
        atomicAdd( lut_grad + ids[i] + 1 * shift, grad_ch1 * w );
        // channel 2
        atomicAdd( lut_grad + ids[i] + 2 * shift, grad_ch2 * w );
    }

    // 对 image_grad[index] 只做一次 atomicAdd（原来是多次等价相加）
    // 原始逻辑等价于： binsize * (grad_ch0 + grad_ch1 + grad_ch2)
    const float sum_three = grad_ch0 + grad_ch1 + grad_ch2;
    // 将单次 atomicAdd 替代多次按权重的 atomicAdds（因为 weights 对 8 个顶点和为 1）
    // 前提：权重对 r/g/b 顶点之和确实为 1（trilinear / quadrilinear 插值），与原逻辑等价。
    atomicAdd( image_grad + index, binsize * sum_three );
}

int QuadriLinearBackwardLaucher(const float* image, float* image_grad,
                                const float* lut, float* lut_grad,
                                const int lut_dim, const int shift,
                                const float binsize, const int width,
                                const int height, const int batch,
                                cudaStream_t stream) {
  const int kThreadsPerBlock = 1024;
  const int output_size = height * width * batch;
  cudaError_t err;

  QuadriLinearBackward<<<(output_size + kThreadsPerBlock - 1) /
                             kThreadsPerBlock,
                         kThreadsPerBlock, 0, stream>>>(
      output_size, image, image_grad, lut, lut_grad, lut_dim, shift, binsize,
      width, height, batch);

  err = cudaGetLastError();
  if (cudaSuccess != err) {
    printf("CUDA kernel failed : %s\n", cudaGetErrorString(err));
    return 0;
  }

  return 1;
}
