#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <trilinear_kernel.h>
#include <trilinear_cuda.h>

int trilinear_forward_cuda( torch::Tensor lut,
                            torch::Tensor image,
                            torch::Tensor output,
                            int lut_dim,
                            int shift,
                            float binsize,
                            int width,
                            int height,
                            int batch )
{
    // Grab the input tensor
    float* lut_flat    = lut.data_ptr< float >( );
    float* image_flat  = image.data_ptr< float >( );
    float* output_flat = output.data_ptr< float >( );

    triLinearForwardLaucher(
        lut_flat, image_flat, output_flat, lut_dim, shift, binsize, width, height, batch, at::cuda::getCurrentCUDAStream( ) 
    );

    return 1;
}

int trilinear_backward_cuda( torch::Tensor image,
                             torch::Tensor image_grad,
                             torch::Tensor lut_grad,
                             int lut_dim,
                             int shift,
                             float binsize,
                             int width,
                             int height,
                             int batch )
{
    // Grab the input tensor
    float* image_grad_flat = image_grad.data_ptr< float >( );
    float* image_flat      = image.data_ptr< float >( );
    float* lut_grad_flat   = lut_grad.data_ptr< float >( );

    triLinearBackwardLaucher(
        image_flat, image_grad_flat, lut_grad_flat, lut_dim, shift, binsize, width, height, batch, at::cuda::getCurrentCUDAStream( ) 
    );

    return 1;
}