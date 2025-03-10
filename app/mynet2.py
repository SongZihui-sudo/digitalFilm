import torch
from torch import nn
from torchvision import models
    
class ResNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, pretrained=True):
        super(ResNetGenerator, self).__init__()

        # 加载预训练的 ResNet-18
        resnet = models.resnet18()
        resnet.load_state_dict(torch.load("./resnet18-5c106cde.pth", weights_only=False))
        
        # 编码器：获取 ResNet 的各层输出，构造 U-Net 风格的跳跃连接
        self.input_conv = nn.Sequential(
            resnet.conv1,   # 输出: 64通道, H/2 x W/2
            resnet.bn1,
            resnet.relu
        )
        self.maxpool = resnet.maxpool   # H/4 x W/4
        self.encoder1 = resnet.layer1   # 输出: 64通道, H/4 x W/4
        self.encoder2 = resnet.layer2   # 输出: 128通道, H/8 x W/8
        self.encoder3 = resnet.layer3   # 输出: 256通道, H/16 x W/16
        self.encoder4 = resnet.layer4   # 输出: 512通道, H/32 x W/32

        # 解码器：使用上采样（双线性插值）+ 卷积来恢复细节，并与跳跃连接特征融合
        self.up1 = self._up_block(512, 256)  # H/32 -> H/16
        self.up2 = self._up_block(512, 128)  # 256(from up1)+256(encoder3) -> H/16 -> H/8
        self.up3 = self._up_block(256, 64)   # 128(from up2)+128(encoder2) -> H/8 -> H/4
        self.up4 = self._up_block(128, 64)   # 64(from up3)+64(encoder1) -> H/4 -> H/2
        self.up5 = self._up_block(128, 64)   # 64(from up4)+64(from input_conv) -> H/2 -> H

        self.final_conv = nn.Conv2d(64, output_nc, kernel_size=1)

        # 添加感知损失模块 (VGG16 特征提取)
        self.vgg = models.vgg16()
        self.vgg.load_state_dict(torch.load("./vgg16-397923af.pth", weights_only=False))
        self.vgg  = self.vgg.features[:16].eval()
        for param in self.vgg.parameters():
            param.requires_grad = False  # 冻结VGG权重

    def _up_block(self, in_channels, out_channels):
        """
        上采样块：先上采样（双线性），再卷积+ReLU
        """
        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        return block

    def add_film_grain(self, x, noise_strength=0.1):
        """
        给图像Tensor添加高斯噪声，模拟胶片颗粒效果。
        
        :param x: 输入图像Tensor
        :param noise_strength: 噪声强度 (0 - 1)，控制颗粒大小
        :return: 添加了噪声的图像Tensor
        """
        batch_size, channels, height, width = x.shape
        mean = 0
        sigma = noise_strength * 255  # 控制噪声的标准差
        
        # 生成噪声并将其与图像叠加
        noise = torch.normal(mean, sigma, size=(batch_size, channels, height, width), dtype=torch.float32, device=x.device)
        noisy_image = x + noise  # 添加噪声
        return noisy_image

    def perceptual_loss(self, real, fake):
        """
        计算感知损失，基于 VGG 特征提取。
        """
        real_features = self.vgg(real)
        fake_features = self.vgg(fake)
        return torch.nn.functional.mse_loss(real_features, fake_features)

    def forward(self, x):
        h = x.shape[2]
        w = x.shape[3]

        # 编码器部分
        x0 = self.input_conv(x)       # x0: [B, 64, H/2, W/2]
        x1 = self.maxpool(x0)         # x1: [B, 64, H/4, W/4]
        x1 = self.encoder1(x1)        # x1: [B, 64, H/4, W/4]
        x2 = self.encoder2(x1)        # x2: [B, 128, H/8, W/8]
        x3 = self.encoder3(x2)        # x3: [B, 256, H/16, W/16]
        x4 = self.encoder4(x3)        # x4: [B, 512, H/32, W/32]

        # 解码器部分：逐步上采样并与编码器对应层特征拼接
        d1 = self.up1(x4)             # d1: [B, 256, H/16, W/16]
        d1 = torch.cat([d1, x3], dim=1)  # [B, 256+256=512, H/16, W/16]

        d2 = self.up2(d1)             # d2: [B, 128, H/8, W/8]
        d2 = torch.cat([d2, x2], dim=1)  # [B, 128+128=256, H/8, W/8]

        d3 = self.up3(d2)             # d3: [B, 64, H/4, W/4]
        d3 = torch.cat([d3, x1], dim=1)  # [B, 64+64=128, H/4, W/4]

        d4 = self.up4(d3)             # d4: [B, 64, H/2, W/2]
        d4 = torch.cat([d4, x0], dim=1)  # [B, 64+64=128, H/2, W/2]

        d5 = self.up5(d4)             # d5: [B, 64, H, W]

        out = self.final_conv(d5)     # out: [B, output_nc, H, W]
        out = torch.tanh(out)

        out = torch.nn.functional.interpolate(out, size=(h, w), mode="bilinear")  # 恢复到原始大小

        # 添加胶片颗粒效果
        out = self.add_film_grain(out, noise_strength=0.0001)

        return out, self.perceptual_loss(x, out)