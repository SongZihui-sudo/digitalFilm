import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import getopt

import mynet

transform = transform = transforms.Compose([
  # transforms.Resize((200, 320)),
  transforms.ToTensor()
])

def load_model(model_path):
    model = mynet.FilmStyleTransfer()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def save_image(image, save_path):
    image.save(save_path)
    print("[INFO] Image save successfully!")

def process_images(model, image):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).cuda()
        output= model(image)
        output = postprocess_image(output)
        return output
           
def main(argv):
    model_path = "./final_gold_200_training_600.pt"
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,"vhi:o:m:",["ifile=","ofile="])
    except getopt.GetoptError:
        print("digitalFilm.py -i <inputfile> -o <outputfile>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("digitalFilm.py -i <inputfile> -o <outputfile>")
            sys.exit()
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