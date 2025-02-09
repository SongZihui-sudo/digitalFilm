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
    if image_path[-4:] == ".dng":
        pass
    else:
        print(1)
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

def save_image(image, save_path):
    """
    保存图片时自动调整白平衡
    :param image: PIL Image对象
    :param save_path: 保存路径
    """
    # 保存图片
    image.save(save_path)
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