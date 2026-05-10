#include <trilinear_cpu.h>
#include <quadrilinear4d.h>

#ifdef WITH_CUDA
#include <trilinear_cuda.h>
#include <quadrilinear4d_cuda.h>
#endif

int trilinear_forward( torch::Tensor lut,
                       torch::Tensor image,
                       torch::Tensor output,
                       int lut_dim,
                       int shift,
                       float binsize,
                       int width,
                       int height,
                       int batch )
{
#ifdef WITH_CUDA
    if (lut.is_cuda())
        return trilinear_forward_cuda(lut, image, output, lut_dim, shift, binsize, width, height, batch);
#endif
    return trilinear_forward_cpu(lut, image, output, lut_dim, shift, binsize, width, height, batch);
}

int trilinear_backward( torch::Tensor image,
                        torch::Tensor image_grad,
                        torch::Tensor lut_grad,
                        int lut_dim,
                        int shift,
                        float binsize,
                        int width,
                        int height,
                        int batch )
{
#ifdef WITH_CUDA
    if (image.is_cuda())
        return trilinear_backward_cuda(image, image_grad, lut_grad, lut_dim, shift, binsize, width, height, batch);
#endif
    return trilinear_backward_cpu(image, image_grad, lut_grad, lut_dim, shift, binsize, width, height, batch);
}

int quadrilinear4d_forward(torch::Tensor lut,
                           torch::Tensor image, torch::Tensor output,
                           int lut_dim,
                           int shift,
                           float binsize,
                           int width,
                           int height,
                           int batch)
{
#ifdef WITH_CUDA
    if (image.is_cuda())
        return quadrilinear4d_forward_cuda(lut, image, output, lut_dim, shift, binsize, width, height, batch);
#endif
    return quadrilinear4d_forward(lut, image, output, lut_dim, shift, binsize, width, height, batch);
}

int quadrilinear4d_backward( torch::Tensor image,
                             torch::Tensor image_grad,
                             torch::Tensor lut,
                             torch::Tensor lut_grad,
                             int lut_dim,
                             int shift,
                             float binsize,
                             int width,
                             int height,
                             int batch )
{
#ifdef WITH_CUDA
    if (image.is_cuda())
        return quadrilinear4d_backward_cuda(image, image_grad, lut, lut_grad, lut_dim, shift, binsize, width, height, batch);
#endif
    return quadrilinear4d_backward(image, image_grad, lut, lut_grad, lut_dim, shift, binsize, width, height, batch);
}

PYBIND11_MODULE( TORCH_EXTENSION_NAME, m )
{
    m.def( "trilinear_forward", &trilinear_forward, "trilinear forward" );
    m.def( "trilinear_backward", &trilinear_backward, "trilinear_backward" );
    m.def("quadrilinear4d_forward", &quadrilinear4d_forward, "quadrilinear4d_forward");
    m.def( "quadrilinear4d_backward", &quadrilinear4d_backward, "quadrilinear4d_backward");
}
