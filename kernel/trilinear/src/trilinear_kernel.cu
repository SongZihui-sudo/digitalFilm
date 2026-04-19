#include <cstdio>

#include <trilinear_kernel.h>

#define CUDA_1D_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)

__global__ void triLinearForwardKernel(
    const int nthreads,
    const float* __restrict__ lut,
    const float* __restrict__ image,
    float* __restrict__ output,
    const int dim,
    const int shift,
    const float binsize,
    const int width,
    const int height,
    const int batch)
{
    const int HW = width * height;
    const int channel_stride = HW;                 // 在 NCHW 中单个 batch 内同通道偏移
    const int batch_stride = 3 * HW;               // 每个 batch 占用的元素数（假设 3 通道）
    const int lut_size_per_channel = dim * dim * dim;

    CUDA_1D_KERNEL_LOOP(global_index, nthreads)
    {
        if (global_index >= nthreads) return; // 额外保险

        // global_index ranges [0, batch*HW)
        const int b = global_index / HW;
        const int p = global_index % HW; // pixel index in H*W

        // compute base offset in image arranged as NCHW flattened:
        // layout: for batch b, channels are contiguous per-batch: base = b * (C*HW)
        const int base = b * (3 * HW) + p; // + c*HW will pick channel

        // load RGB
        const float r = image[base + 0 * HW];
        const float g = image[base + 1 * HW];
        const float bval = image[base + 2 * HW];

        // compute ids safely
        int r_id = (int)floorf(r / binsize);
        int g_id = (int)floorf(g / binsize);
        int b_id = (int)floorf(bval / binsize);

        // clamp ids into [0, dim-1]
        if (r_id < 0) r_id = 0;
        if (g_id < 0) g_id = 0;
        if (b_id < 0) b_id = 0;
        if (r_id >= dim) r_id = dim - 1;
        if (g_id >= dim) g_id = dim - 1;
        if (b_id >= dim) b_id = dim - 1;

        // compute fractional part robustly: avoid fmod weirdness
        float r_frac = (r - (float)r_id * binsize) / binsize;
        float g_frac = (g - (float)g_id * binsize) / binsize;
        float b_frac = (bval - (float)b_id * binsize) / binsize;

        // if we are at the upper boundary (id == dim-1), force frac to 0 and next id == id (so no +1 out-of-bound)
        int r_id_p1 = r_id + 1;
        int g_id_p1 = g_id + 1;
        int b_id_p1 = b_id + 1;
        if (r_id == dim - 1) { r_frac = 0.0f; r_id_p1 = r_id; }
        if (g_id == dim - 1) { g_frac = 0.0f; g_id_p1 = g_id; }
        if (b_id == dim - 1) { b_frac = 0.0f; b_id_p1 = b_id; }

        // compute linear LUT indices (per-channel: 0..dim^3-1)
        auto idx3 = [dim](int rr, int gg, int bb)->int {
            return rr + gg * dim + bb * dim * dim;
        };

        const int id000 = idx3(r_id, g_id, b_id);
        const int id100 = idx3(r_id_p1, g_id, b_id);
        const int id010 = idx3(r_id, g_id_p1, b_id);
        const int id110 = idx3(r_id_p1, g_id_p1, b_id);
        const int id001 = idx3(r_id, g_id, b_id_p1);
        const int id101 = idx3(r_id_p1, g_id, b_id_p1);
        const int id011 = idx3(r_id, g_id_p1, b_id_p1);
        const int id111 = idx3(r_id_p1, g_id_p1, b_id_p1);

        // compute weights
        const float w000 = (1.0f - r_frac) * (1.0f - g_frac) * (1.0f - b_frac);
        const float w100 = (r_frac)       * (1.0f - g_frac) * (1.0f - b_frac);
        const float w010 = (1.0f - r_frac) * (g_frac)       * (1.0f - b_frac);
        const float w110 = (r_frac)       * (g_frac)       * (1.0f - b_frac);
        const float w001 = (1.0f - r_frac) * (1.0f - g_frac) * (b_frac);
        const float w101 = (r_frac)       * (1.0f - g_frac) * (b_frac);
        const float w011 = (1.0f - r_frac) * (g_frac)       * (b_frac);
        const float w111 = (r_frac)       * (g_frac)       * (b_frac);

        // ensure lut indexing safe: each id must be in [0, dim^3-1]
        // compute base_lut pointers
        const int lut_base_r = 0;
        const int lut_base_g = shift;         // your original semantics
        const int lut_base_b = shift * 2;

        // safety: check boundaries before reading (optional in release, useful for debug)
        // (you can assert or just clamp index)
        auto safe_read_lut = [&](const float* arr, int idx)->float {
            if (idx < 0) idx = 0;
            if (idx >= lut_size_per_channel) idx = lut_size_per_channel - 1;
            return arr[idx];
        };

        // read LUT values and combine
        // IMPORTANT: if your lut is actually arranged differently, change indexing accordingly.
        float out_r = 0.0f;
        out_r += w000 * lut[lut_base_r + id000];
        out_r += w100 * lut[lut_base_r + id100];
        out_r += w010 * lut[lut_base_r + id010];
        out_r += w110 * lut[lut_base_r + id110];
        out_r += w001 * lut[lut_base_r + id001];
        out_r += w101 * lut[lut_base_r + id101];
        out_r += w011 * lut[lut_base_r + id011];
        out_r += w111 * lut[lut_base_r + id111];

        float out_g = 0.0f;
        out_g += w000 * lut[lut_base_g + id000];
        out_g += w100 * lut[lut_base_g + id100];
        out_g += w010 * lut[lut_base_g + id010];
        out_g += w110 * lut[lut_base_g + id110];
        out_g += w001 * lut[lut_base_g + id001];
        out_g += w101 * lut[lut_base_g + id101];
        out_g += w011 * lut[lut_base_g + id011];
        out_g += w111 * lut[lut_base_g + id111];

        float out_b = 0.0f;
        out_b += w000 * lut[lut_base_b + id000];
        out_b += w100 * lut[lut_base_b + id100];
        out_b += w010 * lut[lut_base_b + id010];
        out_b += w110 * lut[lut_base_b + id110];
        out_b += w001 * lut[lut_base_b + id001];
        out_b += w101 * lut[lut_base_b + id101];
        out_b += w011 * lut[lut_base_b + id011];
        out_b += w111 * lut[lut_base_b + id111];

        // write to output with same NCHW flattened layout: base + c*HW
        output[base + 0 * HW] = out_r;
        output[base + 1 * HW] = out_g;
        output[base + 2 * HW] = out_b;
    }
}

int triLinearForwardLaucher( const float* lut,
                             const float* image,
                             float* output,
                             const int lut_dim,
                             const int shift,
                             const float binsize,
                             const int width,
                             const int height,
                             const int batch,
                             cudaStream_t stream )
{
    const int kThreadsPerBlock = 1024;
    const int output_size      = height * width * batch;
    cudaError_t err;

    triLinearForwardKernel<<<( output_size + kThreadsPerBlock - 1 ) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
    output_size, lut, image, output, lut_dim, shift, binsize, width, height, batch );

    err = cudaGetLastError( );
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}

// Backward (safe) version
__global__ void triLinearBackwardKernel(
    const int nthreads,
    const float* __restrict__ image,
    const float* __restrict__ image_grad,
    float* __restrict__ lut_grad,
    const int dim,
    const int shift,
    const float binsize,
    const int width,
    const int height,
    const int batch)
{
    const int HW = width * height;
    const int lut_size_per_channel = dim * dim * dim;

    CUDA_1D_KERNEL_LOOP(global_index, nthreads)
    {
        if (global_index >= nthreads) return;

        const int b = global_index / HW;
        const int p = global_index % HW;
        const int base = b * (3 * HW) + p;

        const float r = image[base + 0 * HW];
        const float g = image[base + 1 * HW];
        const float bval = image[base + 2 * HW];

        int r_id = (int)floorf(r / binsize);
        int g_id = (int)floorf(g / binsize);
        int b_id = (int)floorf(bval / binsize);

        if (r_id < 0) r_id = 0;
        if (g_id < 0) g_id = 0;
        if (b_id < 0) b_id = 0;
        if (r_id >= dim) r_id = dim - 1;
        if (g_id >= dim) g_id = dim - 1;
        if (b_id >= dim) b_id = dim - 1;

        float r_frac = (r - (float)r_id * binsize) / binsize;
        float g_frac = (g - (float)g_id * binsize) / binsize;
        float b_frac = (bval - (float)b_id * binsize) / binsize;

        int r_id_p1 = r_id + 1;
        int g_id_p1 = g_id + 1;
        int b_id_p1 = b_id + 1;
        if (r_id == dim - 1) { r_frac = 0.0f; r_id_p1 = r_id; }
        if (g_id == dim - 1) { g_frac = 0.0f; g_id_p1 = g_id; }
        if (b_id == dim - 1) { b_frac = 0.0f; b_id_p1 = b_id; }

        auto idx3 = [dim](int rr, int gg, int bb)->int {
            return rr + gg * dim + bb * dim * dim;
        };

        const int id000 = idx3(r_id, g_id, b_id);
        const int id100 = idx3(r_id_p1, g_id, b_id);
        const int id010 = idx3(r_id, g_id_p1, b_id);
        const int id110 = idx3(r_id_p1, g_id_p1, b_id);
        const int id001 = idx3(r_id, g_id, b_id_p1);
        const int id101 = idx3(r_id_p1, g_id, b_id_p1);
        const int id011 = idx3(r_id, g_id_p1, b_id_p1);
        const int id111 = idx3(r_id_p1, g_id_p1, b_id_p1);

        const float w000 = (1.0f - r_frac) * (1.0f - g_frac) * (1.0f - b_frac);
        const float w100 = (r_frac)       * (1.0f - g_frac) * (1.0f - b_frac);
        const float w010 = (1.0f - r_frac) * (g_frac)       * (1.0f - b_frac);
        const float w110 = (r_frac)       * (g_frac)       * (1.0f - b_frac);
        const float w001 = (1.0f - r_frac) * (1.0f - g_frac) * (b_frac);
        const float w101 = (r_frac)       * (1.0f - g_frac) * (b_frac);
        const float w011 = (1.0f - r_frac) * (g_frac)       * (b_frac);
        const float w111 = (r_frac)       * (g_frac)       * (b_frac);

        // accumulate into lut_grad: ensure indices within bounds
        auto safe_atomic_add = [&](float* base_ptr, int idx, float val){
            if (idx < 0) idx = 0;
            if (idx >= lut_size_per_channel) idx = lut_size_per_channel - 1;
            atomicAdd(base_ptr + idx, val);
        };

        // bases
        float* lut_r_base = lut_grad + 0;
        float* lut_g_base = lut_grad + shift;
        float* lut_b_base = lut_grad + shift * 2;

        // read grads for three channels from image_grad with NCHW layout
        const float grad_r = image_grad[base + 0 * HW];
        const float grad_g = image_grad[base + 1 * HW];
        const float grad_b = image_grad[base + 2 * HW];

        safe_atomic_add(lut_r_base, id000, grad_r * w000);
        safe_atomic_add(lut_r_base, id100, grad_r * w100);
        safe_atomic_add(lut_r_base, id010, grad_r * w010);
        safe_atomic_add(lut_r_base, id110, grad_r * w110);
        safe_atomic_add(lut_r_base, id001, grad_r * w001);
        safe_atomic_add(lut_r_base, id101, grad_r * w101);
        safe_atomic_add(lut_r_base, id011, grad_r * w011);
        safe_atomic_add(lut_r_base, id111, grad_r * w111);

        safe_atomic_add(lut_g_base, id000, grad_g * w000);
        safe_atomic_add(lut_g_base, id100, grad_g * w100);
        safe_atomic_add(lut_g_base, id010, grad_g * w010);
        safe_atomic_add(lut_g_base, id110, grad_g * w110);
        safe_atomic_add(lut_g_base, id001, grad_g * w001);
        safe_atomic_add(lut_g_base, id101, grad_g * w101);
        safe_atomic_add(lut_g_base, id011, grad_g * w011);
        safe_atomic_add(lut_g_base, id111, grad_g * w111);

        safe_atomic_add(lut_b_base, id000, grad_b * w000);
        safe_atomic_add(lut_b_base, id100, grad_b * w100);
        safe_atomic_add(lut_b_base, id010, grad_b * w010);
        safe_atomic_add(lut_b_base, id110, grad_b * w110);
        safe_atomic_add(lut_b_base, id001, grad_b * w001);
        safe_atomic_add(lut_b_base, id101, grad_b * w101);
        safe_atomic_add(lut_b_base, id011, grad_b * w011);
        safe_atomic_add(lut_b_base, id111, grad_b * w111);
    }
}

int triLinearBackwardLaucher( const float* image,
                              const float* image_grad,
                              float* lut_grad,
                              const int lut_dim,
                              const int shift,
                              const float binsize,
                              const int width,
                              const int height,
                              const int batch,
                              cudaStream_t stream )
{
    const int kThreadsPerBlock = 1024;
    const int output_size      = height * width * batch;
    cudaError_t err;

    triLinearBackwardKernel<<<( output_size + kThreadsPerBlock - 1 ) / kThreadsPerBlock, kThreadsPerBlock, 0, stream>>>(
    output_size, image, image_grad, lut_grad, lut_dim, shift, binsize, width, height, batch );

    err = cudaGetLastError( );
    if ( cudaSuccess != err )
    {
        fprintf( stderr, "cudaCheckError() failed : %s\n", cudaGetErrorString( err ) );
        exit( -1 );
    }

    return 1;
}
