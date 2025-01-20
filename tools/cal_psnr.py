import torch
import math

def psnr(target, prediction, max_pixel=1.0):
  """计算 PSNR (峰值信噪比)

  Args:
      target (Tensor): 目标图像（真实图像）
      prediction (Tensor): 生成的图像
      max_pixel (float): 图像的最大像素值，默认是1.0 (对于[0, 1]范围的图像)

  Returns:
      float: PSNR值（单位：dB）
  """
  mse = torch.nn.functional.mse_loss(target, prediction)
  if mse == 0:
      return float('inf')  # 如果 MSE 为 0，意味着两图完全相同
  return 20 * math.log10(max_pixel / math.sqrt(mse))

def evaluate_psnr(data_loader, device, model, max_pixel=1.0):
  total_psnr = 0.0
  num_images = 0

  # 遍历数据集
  for generated_images, original_images in data_loader:
    # 确保输入输出图像在相同的设备上（例如：GPU）
    original_images = original_images.to(device)
    generated_images = generated_images.to(device)

    # 计算每一对图像的 PSNR
    for orig, gen in zip(original_images, generated_images):
      orig = orig.unsqueeze(0).cuda()  # 增加批量维度
      output = model(orig)
      psnr_value = psnr(gen, output[0], max_pixel=max_pixel)
      total_psnr += psnr_value
      num_images += 1

  # 计算整个数据集的平均 PSNR
  avg_psnr = total_psnr / num_images if num_images > 0 else 0.0
  return avg_psnr