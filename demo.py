import gradio as gr
import torch
import torchvision.transforms as transforms
import os
import gc
from PIL import Image

from models.digitalFilm_v2 import digitalFilmv2
from options.options import everyThingOptions

torch.set_num_threads(2)
torch.set_num_interop_threads(1)

SCALE_RATIO = 0.6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def adaptive_resize(image: Image.Image) -> Image.Image:
    w, h = image.size
    new_w = max(round(w * SCALE_RATIO), 32)
    new_h = max(round(h * SCALE_RATIO), 32)
    new_w = ((new_w + 31) // 32) * 32
    new_h = ((new_h + 31) // 32) * 32
    return image.resize((new_w, new_h), Image.BILINEAR)

transform = transforms.Compose([
    transforms.Lambda(adaptive_resize),
    transforms.ToTensor()
])

def load_config(options_path):
    cur_options = everyThingOptions(options_path)
    cur_options.load_config()
    return cur_options

def load_model(model_path):
    options = load_config("./options/digitalFilm.yaml")
    model = digitalFilmv2(0, options.opt.global_config, options.opt.model_config)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    print("[INFO] Open model successfully!")
    return model

@torch.inference_mode()
def process_images(image, model_choice):
    image = transform(image)
    print(os.path.join("checkpoints", model_choice))
    model = load_model(os.path.join("checkpoints", model_choice))
    model.eval()
    image = image.unsqueeze(0)
    image = image.to(device)
    output = model.g(image)["out"]
    output = output.squeeze().cpu().clamp(0, 1)
    output = transforms.ToPILImage()(output)
    gc.collect()
    return output

# ----- 胶片模型元数据 -----

MODELS_DIR = "checkpoints"

FILM_INFO = {
    "kodak_gold_200.pth": {
        "name": "Kodak Gold 200",
        "desc": "暖色调、色彩饱满，经典日光胶片风格",
        "image": "https://github.com/user-attachments/assets/25d75466-0378-444c-8410-ae5196e51d94",
        "emoji": "🌅",
    },
}

PRETRAINED_EXCLUDE = {"resnet18", "resnet50", "vgg16", "vgg19", "inception", "dino", "alexnet"}

def scan_models():
    discovered = {}
    for f in sorted(os.listdir(MODELS_DIR)):
        if not (f.endswith(".pth") and os.path.isfile(os.path.join(MODELS_DIR, f))):
            continue
        stem = os.path.splitext(f)[0].lower().split("-")[0].split("_")[0]
        if stem in PRETRAINED_EXCLUDE:
            continue
        if f in FILM_INFO:
            discovered[f] = FILM_INFO[f]
        else:
            name = f.replace(".pth", "").replace("_", " ").title()
            discovered[f] = {"name": name, "desc": f"{name} 胶片风格", "image": "", "emoji": "🎞️"}
    return discovered

ALL_MODELS = scan_models()
DEFAULT_MODEL = "kodak_gold_200.pth" if "kodak_gold_200.pth" in ALL_MODELS else list(ALL_MODELS.keys())[0]
DEFAULT_NAME = ALL_MODELS.get(DEFAULT_MODEL, {}).get("name", DEFAULT_MODEL)

CARD_COLORS = [
    "linear-gradient(135deg, #f093fb 0%, #f5576c 100%)",
    "linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)",
    "linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)",
    "linear-gradient(135deg, #fa709a 0%, #fee140 100%)",
    "linear-gradient(135deg, #a18cd1 0%, #fbc2eb 100%)",
    "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
]

BASE_CSS = """
#model-key-input { position: absolute; left: -9999px; opacity: 0; height: 1px; width: 1px; overflow: hidden; }
.cards-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 18px; padding: 4px; }
.model-card {
    border: 2px solid #e5e7eb; border-radius: 14px; overflow: hidden; cursor: pointer;
    transition: all 0.25s ease; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,0.06);
}
.model-card:hover {
    transform: translateY(-6px); box-shadow: 0 12px 28px rgba(99,102,241,0.18);
    border-color: #6366f1;
}
.model-card.selected { border-color: #6366f1; box-shadow: 0 0 0 3px rgba(99,102,241,0.3); }
.model-card-img { height: 130px; display: flex; align-items: center; justify-content: center; font-size: 52px; user-select: none; overflow: hidden; position: relative; }
.model-card-img img { width: 100%; height: 100%; object-fit: cover; display: block; }
.model-card-body { padding: 14px 16px; background: #ffffff; }
.model-card-name { font-weight: 700; font-size: 16px; color: #111827 !important; margin-bottom: 6px; }
.model-card-desc { font-size: 14px; color: #374151 !important; line-height: 1.5; }
"""

def build_cards_html(selected_key=None):
    cards = []
    for i, (key, info) in enumerate(ALL_MODELS.items()):
        sel = " selected" if key == selected_key else ""
        bg = CARD_COLORS[i % len(CARD_COLORS)]
        img_url = info.get("image", "").strip()
        if img_url:
            img_block = f'<img src="{img_url}" alt="{info["name"]}" />'
        else:
            img_block = info.get("emoji", "🎞️")
        cards.append(f"""
    <div class="model-card{sel}" data-key="{key}" onclick="selectCard('{key}')">
        <div class="model-card-img" style="background:{bg}">{img_block}</div>
        <div class="model-card-body">
            <div class="model-card-name">{info["name"]}</div>
            <div class="model-card-desc">{info["desc"]}</div>
        </div>
    </div>""")
    return f"""<style>{BASE_CSS}</style>
<div class="cards-grid">{"".join(cards)}</div>
<script>
function selectCard(key) {{
    document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
    const card = document.querySelector(`.model-card[data-key="${{key}}"]`);
    if (card) card.classList.add('selected');
    const h = document.getElementById('model-key-input');
    if (h) {{
        const ta = h.querySelector('textarea') || h.querySelector('input');
        if (ta) {{
            const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
            setter.call(ta, key);
            ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
        }}
        const confirmBtn = document.querySelector('#model-confirm-btn button');
        if (confirmBtn) confirmBtn.click();
    }}
}}
</script>"""

def on_confirm_model(key):
    info = ALL_MODELS.get(key, {})
    return key, info.get("name", key), gr.update(visible=True), gr.update(visible=False), build_cards_html(key)

def on_open_selector():
    return gr.update(visible=False), gr.update(visible=True)

def on_close_selector():
    return gr.update(visible=True), gr.update(visible=False)

def main():
    with gr.Blocks(title="DigitalFilm App", theme=gr.themes.Soft()) as demo:
        model_state = gr.State(DEFAULT_MODEL)
        model_key_input = gr.Textbox(value=DEFAULT_MODEL, elem_id="model-key-input", container=False, label="")
        confirm_model_btn = gr.Button("Confirm", visible=False, elem_id="model-confirm-btn")

        with gr.Column(visible=True) as main_page:
            gr.Markdown("""
# 🎞️ DigitalFilm
### AI 胶片风格模拟
""")
            with gr.Row():
                cur_model_display = gr.Textbox(
                    label="当前胶片", value=DEFAULT_NAME,
                    interactive=False, scale=3, container=True
                )
                select_btn = gr.Button("📂 更换胶片", variant="secondary", scale=1, min_width=140)

            with gr.Row(equal_height=True):
                image_input = gr.Image(type="pil", label="📷 上传图片")
                image_output = gr.Image(type="pil", label="🎨 生成结果")

            with gr.Row():
                run_btn = gr.Button("🚀 开始处理", variant="primary", size="lg", scale=2)
                gr.Column(scale=1)

        with gr.Column(visible=False) as model_page:
            gr.Markdown("## 📸 选择胶片模型")
            with gr.Row():
                close_btn = gr.Button("✕ 返回", variant="secondary", size="sm")
            model_cards = gr.HTML(build_cards_html(DEFAULT_MODEL))

        select_btn.click(fn=on_open_selector, outputs=[main_page, model_page])
        close_btn.click(fn=on_close_selector, outputs=[main_page, model_page])
        confirm_model_btn.click(
            fn=on_confirm_model,
            inputs=[model_key_input],
            outputs=[model_state, cur_model_display, main_page, model_page, model_cards]
        )
        run_btn.click(process_images, inputs=[image_input, model_state], outputs=image_output)

    demo.launch(server_name="0.0.0.0", server_port=7860)

if __name__ == "__main__":
    main()
