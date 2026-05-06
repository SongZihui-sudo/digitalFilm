import gradio as gr
import torch
import torchvision.transforms as transforms
import os

from models.digitalFilm_v2 import digitalFilmv2
from options.options import everyThingOptions
import subprocess, sys, os

kernel_dir = os.path.join(os.path.dirname(__file__), "kernel")
if not os.path.exists(os.path.join(kernel_dir, "build")):
    subprocess.check_call([sys.executable, "-m", "pip", "install", kernel_dir, "--no-build-isolation"])

MAX_WIDTH = 2457
MAX_HEIGHT = 1843
transform = transforms.Compose([
    transforms.Resize((MAX_HEIGHT, MAX_WIDTH)),
    transforms.ToTensor()
])

models = {
    "kodak_gold_200.pth"
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_config(options_path):
    cur_options: everyThingOptions = everyThingOptions(options_path)
    cur_options.load_config()
    return cur_options

def load_model(model_path):
    options = load_config("./options/digitalFilm.yaml")
    model = digitalFilmv2(0, options.opt.global_config, options.opt.model_config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print("[INFO] Open model successfully!")
    return model

def process_images(image, model_choice):
    width, height = image.size

    if width > MAX_WIDTH and height > MAX_HEIGHT:
        raise gr.Error("Image too large!")

    image = transform(image)
    print(os.path.join("checkpoints", model_choice))
    model = load_model(os.path.join("checkpoints", model_choice))
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0)
        image = image.to(device)
        output = model.g(image)["out"]
        output = output.squeeze().cpu().clamp(0, 1)
        output = transforms.ToPILImage()(output)
        return output
    
def main():
    with gr.Blocks(title="DigitalFilm App") as demo:
        image_input = gr.Image(type="pil", label="Upload Image")
        model_choice = gr.Dropdown(models, label="Select Model", allow_custom_value=False)
        image_output = gr.Image(type="pil", label="Generated Image")
        run_button = gr.Button("Run Model")
        run_button.click(process_images, inputs=[image_input, model_choice], outputs=image_output)

    demo.launch()

if __name__ == "__main__":
    main()