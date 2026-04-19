import torch
import my_model_kernel


'''
    三线性插值 py 接口
'''
class trilinearPort(torch.autograd.Function):
        @staticmethod
        def forward(ctx, lut: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
            image = image.contiguous()
            output: torch.Tensor = torch.zeros_like(image, device=image.device)
            
            lut_dim: int = lut.size(-1)
            lut_shift: int = lut_dim ** 3
            lut_binsize: float = 1.0001 / (lut_dim - 1)
            W, H, B = image.size(2), image.size(3), image.size(0)
            
            # 保存上下文
            ctx.save_for_backward(lut, image)
            ctx.lut_dim, ctx.lut_shift, ctx.lut_binsize = lut_dim, lut_shift, lut_binsize
            ctx.W, ctx.H, ctx.B = W, H, B
            
            if B == 1:
                my_model_kernel.trilinear_forward(lut, image, output,
                                                  lut_dim, lut_shift, lut_binsize, W, H, B)
            else:
                output = output.permute(1,0,2,3).contiguous()
                my_model_kernel.trilinear_forward(
                    lut,
                    image.permute(1,0,2,3).contiguous(),
                    output,
                    lut_dim, lut_shift, lut_binsize, W, H, B
                )
                output = output.permute(1,0,2,3).contiguous()
            
            return output

        @staticmethod
        def backward(ctx, grad_output: torch.Tensor): # type: ignore
            lut, image = ctx.saved_tensors
            grad_LUT = torch.zeros_like(lut, device=lut.device)      # 和 lut 一致
            
            if ctx.B == 1:
                my_model_kernel.trilinear_backward(
                    image,
                    grad_output.contiguous(),
                    grad_LUT,
                    ctx.lut_dim,
                    ctx.lut_shift,
                    ctx.lut_binsize,
                    ctx.W,
                    ctx.H,
                    ctx.B
                )
            else:
                my_model_kernel.trilinear_backward(
                    image.permute(1,0,2,3).contiguous(),
                    grad_output.permute(1,0,2,3).contiguous(),
                    grad_LUT,
                    ctx.lut_dim,
                    ctx.lut_shift,
                    ctx.lut_binsize,
                    ctx.W,
                    ctx.H,
                    ctx.B
                )
            
            return grad_LUT, grad_output
    