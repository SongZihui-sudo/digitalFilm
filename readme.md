# DigitalFilm

> 
> End-to-End film aesthetics simulation system built on physical decomposition and neural networks

DigitalFilm is a full-stack film style simulation solution that combines differentiable physical rendering pipelines with neural network training to deliver authentic film texture to digital imagery. It ships with a professional web editing workstation and developer-native MCP integration, serving both everyday color grading workflows and advanced custom extension scenarios.

![Output](./example/digital_output.svg)

## ✨ Core Features

- 🎞️ **Physics-Driven Film Emulation**: The v2 pipeline decomposes film imaging into differentiable stages — exposure, dye coupling, halation diffusion, tone response, and grain — with fully interpretable parameters and intuitive manual tuning
- ⚡ **Lightweight Efficient Inference**: Replaces large generative networks with matrices, 1D/3D LUTs for low VRAM footprint and real-time rendering on consumer GPUs
- 🖥️ **Full-Featured Web Workstation**: Complete image editing application covering base color adjustment, film presets, depth-of-field simulation and quality enhancement
- 🔌 **Native MCP Protocol Support**: Built-in Model Context Protocol service for seamless integration with AI agents and intelligent workflows
- 🧠 **End-to-End Trainable**: Supports custom dataset training for proprietary film styles; near-identity initialization ensures stable training even on small datasets

---

## 📋 Prerequisites

The following runtime environments are required to run this project:

- Python 3.10+
- Go 1.20+
- Node.js 16+ (for frontend development / build)
- pnpm / npm / yarn

Install Python dependencies:

```
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Run Demo

Test the core film emulation capability without launching the full stack:

```
python demo.py
```

### 2. Test MCP Service

Verify MCP protocol connectivity:

```
npx @modelcontextprotocol/inspector --transport stdio -- python app/mcp_server.py
```

### 3. Launch Full Web Application

It is recommended to start the full-stack services in the following order:

#### Step 1: Start Image Processing Service

Handles model inference, image editing and style generation:

```
python app/image_server.py
```

#### Step 2: Start Master Backend

Manages projects, image metadata, editing parameters, user permissions and frontend data exchange:

```
cd app/master_backend
go run .
# Or compile and run
go build -o master_backend
./master_backend
```

#### Step 3: Start Static Asset Backend

Hosts uploaded original images and rendered results, and provides HTTP access endpoints:

```
cd app/static_backend
go run .
# Or compile and run
go build -o static_backend
./static_backend
```

#### Step 4: Start Frontend Application

```
cd app/frontend
npm run dev
# Or with pnpm
pnpm dev
```

Once all services are up, access the DigitalFilm editing workstation in your browser.

> 
> 💡 Admin initialization: After first deployment, run the following SQL to grant admin privileges to the first user
> 
> 
> ```
> UPDATE users SET is_admin = 1 WHERE username = 'your_username';
> ```

---

## 📂 Project Structure

```
DigitalFilm/
├── app/                    # Full-stack web application code
│   ├── frontend/           # Frontend UI (Vue + Vite + TypeScript)
│   ├── master_backend/     # Master backend: projects, images, parameters, user management
│   ├── static_backend/     # Static asset backend: hosts originals and outputs
│   ├── image_server.py     # Python image processing & model inference service
│   └── mcp_server.py       # MCP protocol service
├── options/                # Model training configuration files
├── example/                # Sample images and demo assets
├── pipeline.py             # Model training entry point
└── ...
```

---

## 🎞️ digitalFilm v2 Physical Rendering Pipeline

![digitalFilmv2](./example/digitalFILm.png)

digitalFilm v2 is the core technical module of the project. It adopts a **physically decomposed differentiable rendering pipeline** design, addressing the limitations of the v1 fully-convolutional GAN approach: high VRAM cost, poor interpretability, and unstable training on small datasets.

### Design Philosophy

Inspired by computational optics film simulation concepts (credits to the Phos project), the pipeline decomposes the physical film imaging process into independent differentiable stages, each corresponding to a real-world film formation step. All modules are initialized near identity mapping and learn residual style offsets, ensuring physical interpretability and greatly improving training stability on small datasets.

### Full Pipeline Flow

The pipeline is divided into two processing domains: **linear domain** and **curve domain**.

#### Linear Domain (Optical Physics Stage)

1. **ExposureModule**: Per-channel exposure gain and bias adjustment, simulating film exposure compensation
2. **Spectral Dye Mixing**: Two selectable modes
   - `linear` lightweight mode: 3×3 spectral mixing matrix + residual 3D LUT, balancing speed and basic color coupling
   - `density` physics mode (Phos-style): linear dye crosstalk in density domain + compact density 3D LUT, more faithful to the subtractive mixing physics of film dyes
3. **PyramidBloom**: Multi-scale Gaussian pyramid simulates light scattering in the emulsion layer, with wavelength-dependent scattering radius (red scatters farthest, blue closest) to reproduce authentic film highlight halation

#### Curve Domain (Photochemical Processing Stage)

4. **ToneResponseCurve**: Per-channel 1D LUT emulates the film H&D characteristic curve, delivering the classic toe-linear-shoulder tone response of analog film
5. **Residual Color Correction**: Compact 3D LUT captures non-linear color crossover effects beyond linear matrix capability
6. **Grain**: Multi-scale luminance-modulated noise with heavier grain in shadows, reproducing organic film grain texture

### Technical Advantages

- Fully differentiable end-to-end pipeline, supporting end-to-end training
- Parameters carry clear physical meaning, facilitating manual fine-tuning and style control
- Far lower computational cost than fully convolutional generative models, enabling real-time inference
- Near-identity initialization prevents training collapse on small datasets

---

## 🖥️ Web Editing Application

### Core Functions

- **Base Color Tools**: Full parameter control over exposure, contrast, highlights/shadows, color temperature/tint, and saturation
- **Film Style Processing**: Built-in multiple digitalFilm v2 film presets, with independent controls for grain intensity and halation strength, plus advanced color grading
- **Depth of Field Simulation**: DeepAnything2-based depth estimation for large-aperture bokeh effect
- **Quality Enhancement**: OSEDiff-powered image restoration, detail enhancement and noise reduction

---

## 🔌 MCP Service

The project includes a built-in MCP (Model Context Protocol) service, which seamlessly integrates DigitalFilm's film emulation and image editing capabilities into MCP-compatible AI applications and intelligent agent systems for building automated smart imaging workflows.

Launch command:

```
python app/mcp_server.py
```

---

## 🧠 Model Training

The training entry point is `pipeline.py`. Customize training parameters by modifying configuration files under the `options/` directory:

- Dataset path and data loading strategy
- Training hyperparameters: batch size, learning rate, iterations
- Model structure: LUT dimensions, basis count
- Feature toggles: 3D/4D LUT, residual connection, blending mode

Start training:

```
python pipeline.py
```

---

## 🗺️ Roadmap

✅ **Completed**

- Physically decomposed digitalFilm v2 rendering pipeline
- Depth of field simulation
- Image quality enhancement
- MCP protocol service support

📌 **Planned**

- History system: full undo/redo for editing steps
- Preset management: save, import and share custom presets
- Batch processing: apply styles and color grades to multiple images
- Advanced color tools: curves, levels, independent HSL adjustment
- Agent-driven editing: AI-powered auto color grading via MCP
- Plugin system: third-party feature extension support
- Native RAW support: direct RAW negative processing

---

## 🤝 Contributing

Contributions are welcome! Follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is open-sourced under the GPL-v3 License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Vue.js / Vite / TypeScript frontend technology stack
- All developers who have contributed code and suggestions to this project
- [Phos](https://link.wtturl.cn/?target=https%3A%2F%2Fgithub.com%2FZacharyHu0%2FPhos&scene=im&aid=497858&lang=zh) project for the computational optics film simulation inspiration

---

## 👤 Author

- **SongZihui-sudo** - *Initial work* - \[SongZihui-sudo\]

⭐ If you find this project helpful, please give it a Star!
