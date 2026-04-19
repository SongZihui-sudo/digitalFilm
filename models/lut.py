import torch
from typing import cast

from kernel.my_model_port import trilinear_port


def identity4d_tensor(dim, num_context_bins=2):
    """
    Creates an identity 4D LUT tensor of shape (3, num_context_bins, dim, dim, dim).
    Matches the channel ordering and grid generation approach of the 3D version.
    """
    # Create normalized steps from 0 to 1
    step = torch.linspace(0, 1, steps=dim)

    # Create 3D grids for each channel using same pattern as 3D version
    LUT = torch.empty(3, dim, dim, dim)
    # Red channel varies along first axis
    LUT[0] = step.unsqueeze(0).unsqueeze(0).expand(dim, dim, dim)
    # Green channel varies along second axis
    LUT[1] = step.unsqueeze(-1).unsqueeze(0).expand(dim, dim, dim)
    # Blue channel varies along third axis
    LUT[2] = step.unsqueeze(-1).unsqueeze(-1).expand(dim, dim, dim)

    # Add extra dimension for context bins and replicate
    LUT = LUT.unsqueeze(1).expand(3, num_context_bins, dim, dim, dim).clone()

    return LUT


class CLUT4D(torch.nn.Module):
    """
    This module learns a collection of basis 4D LUT atoms and combines them
    (using a predicted weight vector) to produce a fused 4D LUT.

    The learned parameter tensor has shape:
        (num, 3, num_context_bins, dim, dim, dim)
    where:
      - num is the number of basis atoms,
      - 3 is for the RGB channels,
      - num_context_bins corresponds to the extra dimension (e.g., for a context channel),
      - dim is the resolution along each spatial/color axis.
    """

    def __init__(self, num, dim=17, num_context_bins=2):
        """
        Args:
            num (int): number of basis 4D LUT atoms.
            dim (int): resolution of the LUT grid along each dimension.
            num_context_bins (int): number of context bins.
        """
        super(CLUT4D, self).__init__()
        self.num = num
        self.dim = dim
        self.num_context_bins = num_context_bins
        # Learn a tensor of shape (num, 3, num_context_bins, dim, dim, dim)
        self.LUTs = torch.nn.Parameter(torch.zeros(num, 3, num_context_bins, dim, dim, dim))
        # Initialize the LUT atoms with small random values
        torch.nn.init.uniform_(self.LUTs, -0.1, 0.1)

    def combine(self, weight, identity_lut):
        """
        Combine the learned LUT atoms using the predicted weight vector,
        and then add the identity LUT as a residual.

        Args:
            weight (Tensor): predicted weight vector of shape (batch, num)
            identity_lut (Tensor): identity 4D LUT of shape (3, num_context_bins, dim, dim, dim)

        Returns:
            Tensor: fused 4D LUT of shape (batch, 3, num_context_bins, dim, dim, dim)
        """
        # Flatten the learned LUTs: shape (num, 3*num_context_bins*dim^3)
        LUTs_flat = self.LUTs.view(
            self.num, -1
        )  # shape: (num, N), where N = 3*num_context_bins*dim^3
        # Weight the LUT atoms:
        #   (batch, num) x (num, N) -> (batch, N)
        fused_flat = torch.matmul(weight, LUTs_flat)
        # Reshape back to (batch, 3, num_context_bins, dim, dim, dim)
        fused_lut = fused_flat.view(
            -1, 3, self.num_context_bins, self.dim, self.dim, self.dim
        )
        # Add the identity LUT (unsqueezed to match batch dimension)
        fused_lut = fused_lut + identity_lut.unsqueeze(0)
        # Clamp to the valid range [0, 1]
        fused_lut = torch.clamp(fused_lut, 0, 1)
        
        return fused_lut

    def forward(self, weight, identity_lut, tvmn_module=None):
        """
        Args:
            weight (Tensor): predicted weight vector, shape (batch, num)
            identity_lut (Tensor): identity 4D LUT of shape (3, num_context_bins, dim, dim, dim)
            tvmn_module (callable, optional): a module/function to compute TV/monotonicity loss.

        Returns:
            fused_lut (Tensor): fused 4D LUT for each batch element, shape (batch, 3, num_context_bins, dim, dim, dim)
            tvmn: the TV/monotonicity loss (or 0 if tvmn_module is None)
        """
        fused_lut = self.combine(weight, identity_lut)
        tvmn = 0
        if tvmn_module is not None:
            tvmn = tvmn_module(fused_lut)
        return fused_lut, tvmn


class TV_4D(torch.nn.Module):
    def __init__(self, dim=17, num_context_bins=2):
        super(TV_4D, self).__init__()
        self.num_context_bins = num_context_bins

        self.weight_r = torch.ones(3, num_context_bins, dim, dim, dim - 1, dtype=torch.float)
        self.weight_r[:, :, :, :, (0, dim - 2)] *= 2.0
        self.weight_g = torch.ones(3, num_context_bins, dim, dim - 1, dim, dtype=torch.float)
        self.weight_g[:, :, :, (0, dim - 2), :] *= 2.0
        self.weight_b = torch.ones(3, num_context_bins, dim - 1, dim, dim, dtype=torch.float)
        self.weight_b[:, :, (0, dim - 2), :, :] *= 2.0

        if self.num_context_bins > 1:
            self.weight_c = torch.ones(
                3, num_context_bins - 1, dim, dim, dim, dtype=torch.float
            )
            if num_context_bins - 1 > 0:
                self.weight_c[:, (0, num_context_bins - 2), :, :, :] *= 2.0

        self.relu = torch.nn.ReLU()

    def forward(self, lut):
        self.weight_r = self.weight_r.to(lut.device)
        self.weight_g = self.weight_g.to(lut.device)
        self.weight_b = self.weight_b.to(lut.device)
        self.weight_c = self.weight_c.to(lut.device)
        
        # Handle batch dimension
        dif_context = lut[:, :, :-1, :, :, :] - lut[:, :, 1:, :, :, :]
        dif_r = lut[:, :, :, :, :, :-1] - lut[:, :, :, :, :, 1:]
        dif_g = lut[:, :, :, :, :-1, :] - lut[:, :, :, :, 1:, :]
        dif_b = lut[:, :, :, :-1, :, :] - lut[:, :, :, 1:, :, :]

        tv = (
            torch.mean(torch.mul((dif_r**2), self.weight_r))
            + torch.mean(torch.mul((dif_g**2), self.weight_g))
            + torch.mean(torch.mul((dif_b**2), self.weight_b))
        )
        mn = (
            torch.mean(self.relu(dif_r))
            + torch.mean(self.relu(dif_g))
            + torch.mean(self.relu(dif_b))
            + torch.mean(self.relu(dif_context))
        )

        if self.num_context_bins > 1:
            tv += torch.mean(torch.mul((dif_context**2), self.weight_c))
            mn += torch.mean(self.relu(dif_context))

        return tv, mn

class Generator3DLUT_identity(torch.nn.Module):
    def __init__(self, basis_lut_path: list[str], dim: int = 33):
        super().__init__()
        
        if dim == 33:
            file = open(basis_lut_path[0],'r')
        elif dim == 64:
            file = open(basis_lut_path[1],'r')
        else:
            raise FileNotFoundError(f"3D LUT dimensions supported: 33, 64. Unsupported: {dim}.")
        
        LUT: list[str] = file.readlines()
        self.LUT = torch.zeros(3, dim, dim, dim, dtype=torch.float)

        for i in range(0,dim):
            for j in range(0,dim):
                for k in range(0,dim):
                    n: int = i * dim*dim + j * dim + k
                    x: list[str] = LUT[n].split()
                    self.LUT[0, i, j, k] = float(x[0])
                    self.LUT[1, i, j, k] = float(x[1])
                    self.LUT[2, i, j, k] = float(x[2])
                    
        self.LUT = torch.nn.Parameter(self.LUT.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, trilinear_port.trilinearPort.apply(self.LUT, x))

class Generator3DLUT_zero(torch.nn.Module):
    def __init__(self, dim: int = 33):
        super().__init__()

        self.LUT = torch.zeros(3, dim, dim, dim, dtype=torch.float)
        self.LUT = torch.nn.Parameter(self.LUT.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, trilinear_port.trilinearPort.apply(self.LUT, x))

class Generator3DLUT_rand(torch.nn.Module):
    def __init__(self, dim: int = 33):
        super().__init__()
        
        self.LUT = torch.randn(3, dim, dim, dim, dtype=torch.float) * 0.01
        self.LUT = torch.nn.Parameter(self.LUT.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, trilinear_port.trilinearPort.apply(self.LUT, x))

class TV3D(torch.nn.Module):
    def __init__(self, dim: int = 33):
        super().__init__()

        self.__weight_r: torch.Tensor = torch.ones(3,dim,dim,dim-1, dtype=torch.float)
        self.__weight_r[:,:,:,(0,dim-2)] *= 2.0
        
        self.__weight_g: torch.Tensor = torch.ones(3,dim,dim-1,dim, dtype=torch.float)
        self.__weight_g[:,:,(0,dim-2),:] *= 2.0
        
        self.__weight_b: torch.Tensor = torch.ones(3,dim-1,dim,dim, dtype=torch.float)
        self.__weight_b[:,(0,dim-2),:,:] *= 2.0
        
        self.relu = torch.nn.LeakyReLU()

    def forward(self, LUT: Generator3DLUT_zero | Generator3DLUT_identity) -> tuple[torch.Tensor, torch.Tensor]:
        lut: torch.Tensor = LUT.LUT
        self.__weight_r = self.__weight_r.to(lut.device)
        self.__weight_g = self.__weight_g.to(lut.device)
        self.__weight_b = self.__weight_b.to(lut.device)
        
        dif_r: torch.Tensor = lut[:,:,:,:-1] - lut[:,:,:,1:]
        dif_g: torch.Tensor = lut[:,:,:-1,:] - lut[:,:,1:,:]
        dif_b: torch.Tensor = lut[:,:-1,:,:] - lut[:,1:,:,:]
        
        tv: torch.Tensor = torch.mean(torch.mul((dif_r ** 2), self.__weight_r)) + \
                           torch.mean(torch.mul((dif_g ** 2), self.__weight_g)) + \
                           torch.mean(torch.mul((dif_b ** 2), self.__weight_b))
        mn: torch.Tensor = torch.mean(self.relu(dif_r)) + torch.mean(self.relu(dif_g)) + \
                           torch.mean(self.relu(dif_b))

        return tv, mn
    