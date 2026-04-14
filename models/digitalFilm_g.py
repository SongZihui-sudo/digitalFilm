import torch
from typing import Optional

from kernel.my_model_port.trilinear_port import trilinearPort
from kernel.my_model_port.quadrilinear_port import quadrilinearPort


def identity3d_tensor(dim: int) -> torch.Tensor:
    """
    Create identity 3D LUT with shape (3, dim, dim, dim)
    """
    step = torch.linspace(0.0, 1.0, steps=dim)
    lut = torch.empty(3, dim, dim, dim)

    # Keep same convention as your previous code
    lut[0] = step.unsqueeze(0).unsqueeze(0).expand(dim, dim, dim)
    lut[1] = step.unsqueeze(-1).unsqueeze(0).expand(dim, dim, dim)
    lut[2] = step.unsqueeze(-1).unsqueeze(-1).expand(dim, dim, dim)

    return lut


def identity4d_tensor(dim: int, num_context_bins: int = 2) -> torch.Tensor:
    """
    Create identity 4D LUT with shape (3, num_context_bins, dim, dim, dim)
    """
    step = torch.linspace(0.0, 1.0, steps=dim)
    lut = torch.empty(3, dim, dim, dim)

    lut[0] = step.unsqueeze(0).unsqueeze(0).expand(dim, dim, dim)
    lut[1] = step.unsqueeze(-1).unsqueeze(0).expand(dim, dim, dim)
    lut[2] = step.unsqueeze(-1).unsqueeze(-1).expand(dim, dim, dim)

    lut = lut.unsqueeze(1).expand(3, num_context_bins, dim, dim, dim).clone()
    return lut


class TV3D(torch.nn.Module):
    def __init__(self, dim: int = 33):
        super().__init__()
        self.weight_r = torch.ones(3, dim, dim, dim - 1, dtype=torch.float)
        self.weight_r[:, :, :, (0, dim - 2)] *= 2.0

        self.weight_g = torch.ones(3, dim, dim - 1, dim, dtype=torch.float)
        self.weight_g[:, :, (0, dim - 2), :] *= 2.0

        self.weight_b = torch.ones(3, dim - 1, dim, dim, dtype=torch.float)
        self.weight_b[:, (0, dim - 2), :, :] *= 2.0

        self.relu = torch.nn.ReLU()

    def forward(self, lut: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.weight_r = self.weight_r.to(lut.device)
        self.weight_g = self.weight_g.to(lut.device)
        self.weight_b = self.weight_b.to(lut.device)

        dif_r = lut[:, :, :, :-1] - lut[:, :, :, 1:]
        dif_g = lut[:, :, :-1, :] - lut[:, :, 1:, :]
        dif_b = lut[:, :-1, :, :] - lut[:, 1:, :, :]

        tv = (
            torch.mean((dif_r ** 2) * self.weight_r) +
            torch.mean((dif_g ** 2) * self.weight_g) +
            torch.mean((dif_b ** 2) * self.weight_b)
        )
        mn = (
            torch.mean(self.relu(dif_r)) +
            torch.mean(self.relu(dif_g)) +
            torch.mean(self.relu(dif_b))
        )
        return tv, mn


class TV4D(torch.nn.Module):
    def __init__(self, dim: int = 17, num_context_bins: int = 2):
        super().__init__()
        self.num_context_bins = num_context_bins

        self.weight_r = torch.ones(3, num_context_bins, dim, dim, dim - 1, dtype=torch.float)
        self.weight_r[:, :, :, :, (0, dim - 2)] *= 2.0

        self.weight_g = torch.ones(3, num_context_bins, dim, dim - 1, dim, dtype=torch.float)
        self.weight_g[:, :, :, (0, dim - 2), :] *= 2.0

        self.weight_b = torch.ones(3, num_context_bins, dim - 1, dim, dim, dtype=torch.float)
        self.weight_b[:, :, (0, dim - 2), :, :] *= 2.0

        if num_context_bins > 1:
            self.weight_c = torch.ones(3, num_context_bins - 1, dim, dim, dim, dtype=torch.float)
            self.weight_c[:, (0, num_context_bins - 2), :, :, :] *= 2.0
        else:
            self.weight_c = None

        self.relu = torch.nn.ReLU()

    def forward(self, lut: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # lut: [B, 3, Ctx, D, D, D]
        self.weight_r = self.weight_r.to(lut.device)
        self.weight_g = self.weight_g.to(lut.device)
        self.weight_b = self.weight_b.to(lut.device)
        if self.weight_c is not None:
            self.weight_c = self.weight_c.to(lut.device)

        dif_context = lut[:, :, :-1, :, :, :] - lut[:, :, 1:, :, :, :]
        dif_r = lut[:, :, :, :, :, :-1] - lut[:, :, :, :, :, 1:]
        dif_g = lut[:, :, :, :, :-1, :] - lut[:, :, :, :, 1:, :]
        dif_b = lut[:, :, :, :-1, :, :] - lut[:, :, :, 1:, :, :]

        tv = (
            torch.mean((dif_r ** 2) * self.weight_r) +
            torch.mean((dif_g ** 2) * self.weight_g) +
            torch.mean((dif_b ** 2) * self.weight_b)
        )
        mn = (
            torch.mean(self.relu(dif_r)) +
            torch.mean(self.relu(dif_g)) +
            torch.mean(self.relu(dif_b))
        )

        if self.weight_c is not None:
            tv = tv + torch.mean((dif_context ** 2) * self.weight_c)
            mn = mn + torch.mean(self.relu(dif_context))

        return tv, mn


class GlobalFeatureNet(torch.nn.Module):
    """
    A tiny CNN to predict LUT mixing weights.
    """
    def __init__(self, in_channels: int = 3, dim: int = 32, out_dim: int = 8):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, dim, 3, 1, 1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(dim, dim, 3, 2, 1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(dim, dim * 2, 3, 2, 1),
            torch.nn.ReLU(inplace=True),

            torch.nn.Conv2d(dim * 2, dim * 2, 3, 2, 1),
            torch.nn.ReLU(inplace=True),

            torch.nn.AdaptiveAvgPool2d(1),
        )

        self.head = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(dim * 2, dim * 2),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(dim * 2, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        out = self.head(feat)
        return out


class BasisLUT3D(torch.nn.Module):
    """
    Learn N basis 3D LUTs and combine them using predicted weights.
    """
    def __init__(self, num_basis: int = 8, dim: int = 33):
        super().__init__()
        self.num_basis = num_basis
        self.dim = dim

        self.luts = torch.nn.Parameter(torch.zeros(num_basis, 3, dim, dim, dim))
        torch.nn.init.uniform_(self.luts, -0.05, 0.05)

        identity = identity3d_tensor(dim)
        self.register_buffer("identity_lut", identity)

    def combine(self, weight: torch.Tensor) -> torch.Tensor:
        # weight: [B, num_basis]
        lut_flat = self.luts.view(self.num_basis, -1)              # [N, M]
        fused_flat = torch.matmul(weight, lut_flat)                # [B, M]
        fused = fused_flat.view(-1, 3, self.dim, self.dim, self.dim)
        fused = fused + self.identity_lut.unsqueeze(0)
        fused = torch.clamp(fused, 0.0, 1.0)
        return fused

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return self.combine(weight)


class BasisLUT4D(torch.nn.Module):
    """
    Learn N basis 4D LUTs and combine them using predicted weights.
    """
    def __init__(self, num_basis: int = 8, dim: int = 17, num_context_bins: int = 2):
        super().__init__()
        self.num_basis = num_basis
        self.dim = dim
        self.num_context_bins = num_context_bins

        self.luts = torch.nn.Parameter(
            torch.zeros(num_basis, 3, num_context_bins, dim, dim, dim)
        )
        torch.nn.init.uniform_(self.luts, -0.05, 0.05)

        identity = identity4d_tensor(dim, num_context_bins)
        self.register_buffer("identity_lut", identity)

    def combine(self, weight: torch.Tensor) -> torch.Tensor:
        # weight: [B, num_basis]
        lut_flat = self.luts.view(self.num_basis, -1)
        fused_flat = torch.matmul(weight, lut_flat)
        fused = fused_flat.view(
            -1, 3, self.num_context_bins, self.dim, self.dim, self.dim
        )
        fused = fused + self.identity_lut.unsqueeze(0)
        fused = torch.clamp(fused, 0.0, 1.0)
        return fused

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        return self.combine(weight)


class digitalFilm_G(torch.nn.Module):
    """
    Lightweight digital-to-film generator based on:
      - basis 3D LUT mixture
      - optional basis 4D LUT mixture
      - optional residual blending
      - optional LUT regularization output

    Input:
      x: [B,3,H,W] in [0,1]

    Output:
      dict with:
        "out": final image
        "lut3d": fused 3D LUT or None
        "lut4d": fused 4D LUT or None
        "tv3d": tv regularization term
        "mn3d": monotonicity regularization term
        "tv4d": tv regularization term
        "mn4d": monotonicity regularization term
        "weight3d": predicted basis weights
        "weight4d": predicted basis weights
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

        self.use_3d = getattr(config, "use_3d", True)
        self.use_4d = getattr(config, "use_4d", False)

        self.num_basis_3d = getattr(config, "num_basis_3d", 8)
        self.num_basis_4d = getattr(config, "num_basis_4d", 8)

        self.lut3d_dim = getattr(config, "lut3d_dim", 33)
        self.lut4d_dim = getattr(config, "lut4d_dim", 17)
        self.num_context_bins = getattr(config, "num_context_bins", 2)

        self.feature_dim = getattr(config, "feature_dim", 32)
        self.residual = getattr(config, "residual", True)
        self.learn_blend = getattr(config, "learn_blend", True)
        self.clamp_output = getattr(config, "clamp_output", True)

        if self.use_3d:
            self.weight_net_3d = GlobalFeatureNet(
                in_channels=3,
                dim=self.feature_dim,
                out_dim=self.num_basis_3d
            )
            self.lut3d_module = BasisLUT3D(
                num_basis=self.num_basis_3d,
                dim=self.lut3d_dim
            )
            self.tv3d_module = TV3D(self.lut3d_dim)

        if self.use_4d:
            self.weight_net_4d = GlobalFeatureNet(
                in_channels=3,
                dim=self.feature_dim,
                out_dim=self.num_basis_4d
            )
            self.lut4d_module = BasisLUT4D(
                num_basis=self.num_basis_4d,
                dim=self.lut4d_dim,
                num_context_bins=self.num_context_bins
            )
            self.tv4d_module = TV4D(
                dim=self.lut4d_dim,
                num_context_bins=self.num_context_bins
            )

        if self.learn_blend:
            blend_out_dim = 0
            if self.use_3d:
                blend_out_dim += 1
            if self.use_4d:
                blend_out_dim += 1

            self.blend_net = GlobalFeatureNet(
                in_channels=3,
                dim=max(16, self.feature_dim // 2),
                out_dim=blend_out_dim if blend_out_dim > 0 else 1
            )

    def _predict_context(self, x: torch.Tensor) -> torch.Tensor:
        """
        Build a 4D-LUT input tensor with 3 RGB channels + extra context channels.
        Here we use luminance as context and duplicate/pack to fit your kernel's 6-ch input.

        quadrilinear backward code implies x has 6 channels:
          first 3 channels: context-related / auxiliary input
          last 3 channels: RGB to be transformed

        So here we build:
          [ctx1, ctx2, ctx3, R, G, B]
        For simplicity:
          ctx1 = luminance
          ctx2 = max_rgb
          ctx3 = saturation proxy
        """
        r = x[:, 0:1]
        g = x[:, 1:2]
        b = x[:, 2:3]

        luminance = 0.299 * r + 0.587 * g + 0.114 * b
        max_rgb = torch.max(x, dim=1, keepdim=True)[0]
        min_rgb = torch.min(x, dim=1, keepdim=True)[0]
        sat_proxy = max_rgb - min_rgb

        quad_input = torch.cat([luminance, max_rgb, sat_proxy, r, g, b], dim=1)
        return quad_input

    def _apply_3d_lut_batch(self, lut: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply per-sample 3D LUTs.
        lut: [B,3,D,D,D]
        x:   [B,3,H,W]
        """
        B = x.size(0)
        outs = []
        for i in range(B):
            out_i = trilinearPort.apply(lut[i], x[i:i+1])
            outs.append(out_i)
        return torch.cat(outs, dim=0)

    def _apply_4d_lut_batch(self, lut: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Apply per-sample 4D LUTs.
        lut: [B,3,Ctx,D,D,D]
        x:   [B,3,H,W]
        """
        quad_input = self._predict_context(x)  # [B,6,H,W]
        outs = []
        for i in range(x.size(0)):
            _, out_i = quadrilinearPort.apply(lut[i], quad_input[i:i+1])
            outs.append(out_i)
        return torch.cat(outs, dim=0)

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4 and x.size(1) == 3, "Input must be [B,3,H,W]"
        input_x = x

        out_3d: Optional[torch.Tensor] = None
        out_4d: Optional[torch.Tensor] = None
        lut3d: Optional[torch.Tensor] = None
        lut4d: Optional[torch.Tensor] = None
        weight3d: Optional[torch.Tensor] = None
        weight4d: Optional[torch.Tensor] = None

        tv3d = x.new_tensor(0.0)
        mn3d = x.new_tensor(0.0)
        tv4d = x.new_tensor(0.0)
        mn4d = x.new_tensor(0.0)

        # 3D branch
        if self.use_3d:
            logits3d = self.weight_net_3d(x)
            weight3d = torch.softmax(logits3d, dim=1)
            lut3d = self.lut3d_module(weight3d)
            out_3d = self._apply_3d_lut_batch(lut3d, x)

            # regularization over fused LUTs
            tv3d_list = []
            mn3d_list = []
            for i in range(lut3d.size(0)):
                cur_tv, cur_mn = self.tv3d_module(lut3d[i])
                tv3d_list.append(cur_tv)
                mn3d_list.append(cur_mn)
            tv3d = torch.stack(tv3d_list).mean()
            mn3d = torch.stack(mn3d_list).mean()

        # 4D branch
        if self.use_4d:
            logits4d = self.weight_net_4d(x)
            weight4d = torch.softmax(logits4d, dim=1)
            lut4d = self.lut4d_module(weight4d)
            out_4d = self._apply_4d_lut_batch(lut4d, x)

            tv4d, mn4d = self.tv4d_module(lut4d)

        # fusion
        if self.use_3d and self.use_4d:
            if self.learn_blend:
                blend_logits = self.blend_net(x)
                blend = torch.softmax(blend_logits, dim=1)  # [B,2]
                w3 = blend[:, 0:1].unsqueeze(-1).unsqueeze(-1)
                w4 = blend[:, 1:2].unsqueeze(-1).unsqueeze(-1)
                out = w3 * out_3d + w4 * out_4d
            else:
                out = 0.5 * out_3d + 0.5 * out_4d

        elif self.use_3d:
            out = out_3d
        elif self.use_4d:
            out = out_4d
        else:
            out = x

        if self.residual:
            # residual style blending, often more stable for film rendering
            out = 0.7 * out + 0.3 * input_x

        if self.clamp_output:
            out = torch.clamp(out, 0.0, 1.0)

        return {
            "out": out,
            "lut3d": lut3d,
            "lut4d": lut4d,
            "tv3d": tv3d,
            "mn3d": mn3d,
            "tv4d": tv4d,
            "mn4d": mn4d,
            "weight3d": weight3d,
            "weight4d": weight4d,
        }

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

    from options.options import everyThingOptions

    inp = torch.randn([16, 3, 256, 256])
    options_path: str = "D:/projects/digitalFilm/options/digitalFilm.yaml"
    cur_options: everyThingOptions = everyThingOptions(options_path)
    cur_options.load_config()
    g = digitalFilm_G(cur_options)
    g.forward(inp)
