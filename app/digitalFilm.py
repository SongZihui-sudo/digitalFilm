import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import getopt
import tkinter as tk
from tkinter import filedialog
import numpy as np

import mynet

transform = transform = transforms.Compose([
  transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    model = mynet.FilmStyleTransfer()
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("[INFO] Open model successfully!")
    return model

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image)
    print("[INFO] Open image successfully!")
    return image

def postprocess_image(tensor):
    """将模型输出的张量转换为 PIL 图像"""
    tensor = tensor.squeeze(0)  # 去掉 batch 维度
    tensor = tensor.clamp(0, 1)  # 确保值在 [0, 1] 范围内
    transform = transforms.ToPILImage()  # 转换为 PIL 图像
    image = transform(tensor)
    return image

def adjust_white_balance(image):
    """
    自动调整彩色图片的白平衡（基于灰度世界算法）
    :param image: PIL Image对象
    :return: 调整后的PIL Image对象
    """
    # 将图像转换为RGB模式（确保不是灰度图）
    image = image.convert("RGB")
    
    # 转换为NumPy数组以便处理
    img_array = np.array(image).astype(np.float32)
    
    # 计算每个通道的平均值
    avg_r = np.mean(img_array[:, :, 0])
    avg_g = np.mean(img_array[:, :, 1])
    avg_b = np.mean(img_array[:, :, 2])
    
    # 灰度世界假设：三个通道的平均值应该相等
    # 计算增益系数（避免除零）
    target_avg = (avg_r + avg_g + avg_b) / 3.0
    gain_r = target_avg / avg_r if avg_r != 0 else 1.0
    gain_g = target_avg / avg_g if avg_g != 0 else 1.0
    gain_b = target_avg / avg_b if avg_b != 0 else 1.0
    
    # 应用增益调整每个通道
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * gain_r, 0, 255)
    img_array[:, :, 1] = np.clip(img_array[:, :, 1] * gain_g, 0, 255)
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * gain_b, 0, 255)
    
    # 转换回PIL Image并返回
    return Image.fromarray(img_array.astype(np.uint8))

def save_image(image, save_path):
    """
    保存图片时自动调整白平衡
    :param image: PIL Image对象
    :param save_path: 保存路径
    """
    # 调整白平衡
    balanced_image = adjust_white_balance(image)
    # 保存图片
    balanced_image.save(save_path)
    print(f"[INFO] Image saved successfully to {save_path}")


def process_images(model, image):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0)
        image = image.to(device)
        output= model(image)
        output = postprocess_image(output)
        return output
           
def main(argv):
    model_path = "./kodark_gold_200.pt"
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"gvhi:o:m:",["ifile=","ofile="])
    except getopt.GetoptError:
        print("python digitalFilm.py [-v/-h/-g] -i <input> -o <ouput> -m <model>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("python digitalFilm.py [-v/-h/-g] -i <input> -o <ouput> -m <model>")
            sys.exit()
        elif opt == '-g':
            root = tk.Tk()
            root.withdraw()
            inputfile = filedialog.askopenfilename()
            outputfile = filedialog.asksaveasfilename()
        elif opt == '-v':
            print("digitalFilm v0.0.1 SongZihui-sudo 1751122876@qq.com")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-o", "--ofile"):
            outputfile = arg
        elif opt in ("-m", "-model"):
            model_path = arg

    if inputfile != '' and outputfile != '':
        image = load_image(inputfile)
        model = load_model(model_path)
        output_image = process_images(model, image)
        save_image(output_image, outputfile)
    else:
        print("digitalFilm.py -i <inputfile> -o <outputfile>")
        print("input path and output path.")

if __name__ == "__main__":
    main(sys.argv[1:])