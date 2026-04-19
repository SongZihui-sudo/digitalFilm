#ifndef QUADRILINEAR_CUDA_H
#define QUADRILINEAR_CUDA_H

#include <torch/extension.h>

int quadrilinear4d_forward_cuda(torch::Tensor lut, torch::Tensor image, torch::Tensor output,
                           int lut_dim, int shift, float binsize, int width, int height, int batch);

int quadrilinear4d_backward_cuda(torch::Tensor image, torch::Tensor image_grad, torch::Tensor lut, torch::Tensor lut_grad,
                            int lut_dim, int shift, float binsize, int width, int height, int batch);

#endif
