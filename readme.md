# DigitalFilm

> A digital darkroom system that simulates film aesthetics using neural networks.

![app](example/app.png)

DigitalFilm is a project dedicated to film-style simulation, comprising the following components:

- **Model Training Pipeline**
- **MCP Service**
- **Image Editing and Film-Style Conversion Application**
- **Main Backend / Static Assets Backend / Image Processing Service**

The project's objective is to render digital images with a film-like aesthetic using neural networks, while providing a practical and usable editing workflow.

---

## Feature Overview

- Uses neural networks to learn the mapping from digital images to film-style images.
- Supports previewing and saving basic image editing parameters.
- Supports project and image management.
- Supports image uploading, static hosting, and access to generated result images.
- Supports integration with other AI applications via the MCP service.
- Supports image generation and processing via a dedicated, standalone image service.

---

## Project Structure

A typical project structure is as follows:

```text
DigitalFilm/
├── app/                    # DigitalFilm application frontend / desktop client code
├──────── master_backend/         # Main backend: manages data for projects, images, parameters, etc.
├──────── static_backend/         # Static assets backend: hosts uploaded images and generated results
├──────── image_server.py         # Python image processing service
├── pipeline.py             # Training entry point
├── mcp_server.py           # MCP service entry point
├── options/                # Model and training configurations
├── example/                # Example images
└── ...
```

## Environmental Dependencies
To use this project, you must install:

Python
Go
If the frontend component within the `app/` directory utilizes Vue, you will also need:

Node.js
pnpm / npm / yarn

## Model Training
The entry point for training is:

```BASH
python pipeline.py
```
You can adjust training parameters by modifying the configuration files; examples include:

Dataset path
Batch size
Learning rate
LUT dimensions
Whether to enable 3D / 4D LUTs
Number of basis functions
Options such as `residual`, `blend`, etc.

## MCP Service
The project provides an MCP service, allowing other AI applications to integrate with and utilize its capabilities. Startup Method:

```BASH
python mcp_server.py
```
Once started, you can integrate this MCP service into any MCP-compatible AI application or agent system to leverage DigitalFilm's capabilities.

## Application Startup
The `app/` directory contains the DigitalFilm application, which supports the following features:

Basic image editing
Image uploading and project management
Film-style conversion
Image parameter saving and restoration
Pre-startup Requirements
To use the DigitalFilm App, you must first launch the following services:

Python Image Editing Server
Go Main Backend
Go
1. Launch the Image Processing Service
The image processing service is responsible for executing tasks related to image generation, image editing, and model inference.

```BASH
python image_server.py
```

2. Launch the Main Backend
The main backend is responsible for:

Project data management
Image metadata management
Storage of editing parameters
Preset configuration management
Data interaction with the frontend
Navigate to the main backend directory, then compile and run:


``` BASH
go run .
```
Alternatively, compile first:

```BASH
go build -o master_backend
./master_backend
```

3. Launch the Static Assets Backend
The static assets backend is responsible for:

Hosting uploaded original images
Hosting generated result images
Providing HTTP access URLs for use by the frontend and other backend services
Navigate to the static backend directory, then compile and run:

```BASH
go run .
```

Alternatively:

```BASH
go build -o static_backend
./static_backend
```
4. Launch the App
If the `app/` directory contains the frontend project:

```BASH
cd app
npm run dev
```
Or:

```BASH
pnpm dev
```
Once launched, you can access and use the DigitalFilm application in your web browser. Recommended Startup Sequence
It is recommended to launch the components in the following order:

```
python image_server.py
master_backend
static_backend
app
```

## DigitalFilm App Feature Overview
The application currently supports:

- Project creation
- Image uploading
- Viewing image lists
- Basic editing parameter adjustments:
- Exposure
- Contrast
- Highlights
- Film-style parameter adjustments:
- Presets
- Grain
- Highlight Bloom

## Model Description

DigitalFilmv2 is a lightweight model for generating digital-to-film style transformations. Its core concepts include:

Basis 3D LUT mixture
(Optional) Basis 4D LUT mixture
A global feature network to predict LUT mixing weights
(Optional) Residual blending
LUT regularization
Total Variation regularization
Monotonicity regularization

The model supports the following parameters:
- use_3d
- use_4d
- num_basis_3d
- num_basis_4d
- lut3d_dim
- lut4d_dim
- num_context_bins
- learn_blend

Its overall objective is to combine the interpretability and expressive power of LUTs with the predictive capabilities of neural networks—in a lightweight manner—to achieve digital image rendering with a distinct "film look."

The model primarily consists of the following modules:

1. GlobalFeatureNet
A lightweight CNN designed to extract global features from the input image and predict:

- 3D LUT basis weights
- 4D LUT basis weights
- Branch blending weights

2. BasisLUT3D
Learns multiple trainable 3D LUT bases and blends them using the predicted weights:

- Outputs the blended 3D LUT
- Adds an identity LUT as an initial baseline
- Ensures the output remains within the [0, 1] range

3. BasisLUT4D
Learns multiple trainable 4D LUT bases and incorporates a "context" dimension to perform more complex color mapping. 4. TV / Monotonicity Regularization
To ensure the smoothness and plausibility of the LUT, the following are incorporated into the training process:

- TV Regularization
- Monotonicity Regularization

5. Residual Blending
Incorporating a certain proportion of the input image into the final output helps enhance stability and naturalness:

```TEXT
out = 0.7 * lut_output + 0.3 * input
```

## Development Notes
The project currently consists of multiple services; it is recommended to debug them independently during development:

Python Model / Image Service
Go Main Backend
Go Static Assets Backend
Frontend App
