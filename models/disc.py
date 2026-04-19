import torch
import functools
import numpy
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.utils import get_filter, get_pad_layer


class downSample(torch.nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super().__init__()
        
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(numpy.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(numpy.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes) # type: ignore

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return torch.nn.functional.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1]) # type: ignore


class nLayerDiscriminator(torch.nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self,
                 in_channels: int,
                 ndf = 64,
                 n_layers: int= 3,
                 norm_layer = torch.nn.BatchNorm2d,
                 no_antialias = 0.0,
                 weight_norm="spectral"): # type: ignore
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
            weight_norm     -- weight_norm
        """
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == torch.nn.InstanceNorm2d # type: ignore
        else:
            use_bias = norm_layer == torch.nn.InstanceNorm2d

        if weight_norm == 'spectral':
            weight_norm = torch.nn.utils.spectral_norm # type: ignore
        else:
            def weight_norm(x): return x

        kw = 4
        padw = 1
        if(no_antialias):
            sequence = [torch.nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=2, padding=padw),
                        torch.nn.LeakyReLU(0.2, True)]
        else:
            sequence = [torch.nn.Conv2d(in_channels, ndf, kernel_size=kw, stride=1, padding=padw),
                        torch.nn.LeakyReLU(0.2, True), downSample(ndf)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if(no_antialias):
                sequence += [
                    torch.nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    torch.nn.LeakyReLU(0.2, True)
                ]
            else:
                sequence += [
                    torch.nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    torch.nn.LeakyReLU(0.2, True),
                    downSample(ndf * nf_mult)
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            torch.nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            torch.nn.LeakyReLU(0.2, True)
        ]

        for i, layer in enumerate(sequence):
            if isinstance(layer, torch.nn.Conv2d):
                sequence[i] = weight_norm(layer)

        self.enc = torch.nn.Sequential(*sequence)
        # output 1 channel prediction map
        self.final_conv = weight_norm(torch.nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))


    def forward(self, input):
        """Standard forward."""
        final_ft = self.enc(input)
        dout = self.final_conv(final_ft)
        
        return dout


class patchDiscriminator(nLayerDiscriminator):
    """Defines a PatchGAN discriminator"""

    def __init__(self, in_channels, ndf=64, n_layers=3, norm_layer=torch.nn.BatchNorm2d, no_antialias=False, patch_size: int = 16):
        super().__init__(in_channels, ndf, n_layers, norm_layer, no_antialias)

        self.__patch_size: int = patch_size

    def forward(self, input):
        B, C, H, W = input.size(0), input.size(1), input.size(2), input.size(3)
        Y = H // self.__patch_size
        X = W // self.__patch_size
        input = input.view(B, C, Y, self.__patch_size, X, self.__patch_size)
        input = input.permute(0, 2, 4, 1, 3, 5).contiguous().view(B * Y * X, C, self.__patch_size, self.__patch_size)
        
        return super().forward(input)
    