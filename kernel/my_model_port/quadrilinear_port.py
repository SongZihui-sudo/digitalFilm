import torch
import my_model_kernel


class quadrilinearPort(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lut: torch.Tensor, x: torch.Tensor):
        x = x.contiguous()
        output = x.new(x.size()[0], 3, x.size()[2], x.size()[3])
        dim = lut.size()[-1]
        shift = 2 * dim**3
        binsize = 1.000001 / (dim - 1)
        W = x.size(2)
        H = x.size(3)
        batch = x.size(0)
        assert 1 == my_model_kernel.quadrilinear4d_forward(
            lut, x, output, dim, shift, binsize, W, H, batch
        )
        int_package = torch.IntTensor([dim, shift, W, H, batch])
        float_package = torch.FloatTensor([binsize])
        variables = [lut, x, int_package, float_package]
        ctx.save_for_backward(*variables)
        return lut, output

    @staticmethod
    def backward(ctx, lut_grad: torch.Tensor, x_grad: torch.Tensor): # type: ignore
        x_grad = x_grad.contiguous()
        output_grad = x_grad.new(
            x_grad.size()[0], 6, x_grad.size()[2], x_grad.size()[3]
        ).fill_(0)
        output_grad[:, 3:, :, :] = x_grad
        lut, x, int_package, float_package = ctx.saved_variables
        dim, shift, W, H, batch = int_package
        dim, shift, W, H, batch = int(dim), int(shift), int(W), int(H), int(batch)
        binsize = float(float_package[0])
        assert 1 == my_model_kernel.quadrilinear4d_backward(
            x, output_grad, lut, lut_grad, dim, shift, binsize, W, H, batch
        )
        return lut_grad, output_grad
    