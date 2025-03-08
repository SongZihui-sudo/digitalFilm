import torch
import torchvision.transforms as transforms
from PIL import Image
import sys
import getopt
import tkinter as tk
from tkinter import filedialog
import numpy as np
import rawpy
from torchvision.utils import save_image
import torchvision

import mynet
import mynet2

transform = transform = transforms.Compose([
  transforms.ToTensor()
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path):
    if model_path[-4:] == ".pth":
        model = mynet2.ResNetGenerator(3,3)
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        model = mynet.FilmStyleTransfer()
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("[INFO] Open model successfully!")
    return model

def load_image(image_path):
    print(image_path[-4:])
    if image_path[-4:] == ".dng":
        with rawpy.imread(image_path) as raw:
            image = raw.postprocess()
    else:
        image = Image.open(image_path).convert("RGB")
    if transform:
        image = transform(image)

    print("[INFO] Open image successfully!")
    
    return image

def process_images(model, image):
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0)
        image = image.to(device)
        output, _ = model(image)
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
        print(f"[INFO] Image saved successfully to {outputfile}")
    else:
        print("digitalFilm.py -i <inputfile> -o <outputfile>")
        print("input path and output path.")

if __name__ == "__main__":
    main(sys.argv[1:])