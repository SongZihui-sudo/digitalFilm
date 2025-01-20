import torch.nn as nn
import torch

class ImageToImageCNN(nn.Module):
  def __init__(self):
    super(ImageToImageCNN, self).__init__()
    # 编码器部分 (使用卷积层)
    self.encoder1 = nn.Sequential(
      nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 输入为 RGB 图像 (3 通道)
      nn.ReLU()
    )
    self.encoder2 = nn.Sequential(
      nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
      nn.ReLU()
    )
    self.encoder3 = nn.Sequential(
      nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
    )

    # 解码器部分 (使用反卷积层)
    self.decoder1 = nn.Sequential(
      nn.ConvTranspose2d(256, 128, kernel_size=8, stride=4, padding=2, output_padding=0),
      nn.ReLU()
    )
    self.decoder2 = nn.Sequential(
      nn.ConvTranspose2d(192, 64, kernel_size=4, stride=2, padding=1, output_padding=0),
      nn.ReLU()
    )
    self.decoder3 = nn.Sequential(
      nn.ConvTranspose2d(67, 3, kernel_size=2, stride=2, padding=0, output_padding=0),
      nn.Tanh(),
      nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False)
    )

    self.apply(self.init_weights)

  def init_weights(self, m):
    """ 使用 Kaiming 初始化卷积层的权重 """
    if isinstance(m, nn.Conv2d):
      nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
      if m.bias is not None:
        nn.init.zeros_(m.bias)

  def forward(self, x):
    enc1 = self.encoder1(x)
    enc2 = self.encoder2(enc1)
    enc3 = self.encoder3(enc2)
    dec1 = self.decoder1(enc3)
    dec2 = self.decoder2(torch.cat([dec1, enc1], dim=1))
    # print(dec2.shape)
    dec3 = self.decoder3(torch.cat([dec2, x], dim=1))
    # print(dec3.shape)
    return dec3