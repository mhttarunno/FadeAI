# FadeAI — Intelligent Neural Inpainting & Photo Editing Studio

A production-ready web application and AI studio for removing unwanted objects, isolating backgrounds, detecting objects, and enhancing photos in real-time. Built on Flask, PyTorch, LaMa, Faster R-CNN, and rembg.

![status](https://img.shields.io/badge/status-production--ready-brightgreen)
![python](https://img.shields.io/badge/python-3.10%2B-blue)
![flask](https://img.shields.io/badge/flask-3.x-black)
![license](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Features

- **Intelligent Object Removal (LaMa / SDXL)**: Fast, seamless deep-learning inpainting that erases unwanted objects and reconstructs natural backgrounds.
- **Smart Object Detection**: Automatically detects people, cars, animals, and items (80 COCO categories) via Faster R-CNN MobileNetV3 and auto-fills mask regions with one click.
- **1-Click Background Remover**: Extracts foreground subjects with high-precision edge cutout (`rembg` / ONNX U2Net) and exports transparent PNGs.
- **AI Image Enhancement**: 2x and 4x super-resolution, intelligent unsharp masking, contrast optimization, and edge-preserving bilateral denoising.
- **Interactive Fabric.js Canvas Studio**:
  - Precision brush tool with adjustable diameter.
  - Polygon lasso tool for complex geometry.
  - Eraser tool to fine-tune selections.
  - Multi-level Undo / Redo (<kbd>Ctrl</kbd>+<kbd>Z</kbd> / <kbd>Ctrl</kbd>+<kbd>Y</kbd>).
  - Pan (<kbd>Space</kbd> + drag) and smooth scroll-wheel zoom.
- **Before / After Comparison**: Interactive split slider and side-by-side view to inspect edits.
- **Responsive Dark & Light Mode**: Seamless theme switching with <kbd>Shift</kbd>+<kbd>D</kbd> shortcut and system theme auto-detection.
- **Real-Time Telemetry Dashboard**: Live metrics tracking processed images, avg. latency, and tool breakdowns.
- **Production-Grade Security**: MIME sniffing verification, file dimension bounds, path traversal guards, and automatic background cleanup.

---

## 🏗️ Architecture & Directory Structure

```
fade-ai/
├── app.py                      # Main application runner and entrypoint
├── fadeai.py                   # Flask application factory & REST endpoints
├── config.py                   # Centralized, environment-driven configuration
├── requirements.txt            # Python dependencies
├── .env.example                # Sample environment configuration
├── run.bat / fade.bat          # 1-click Windows desktop launchers
│
├── models/                     # Deep learning model wrappers (cached singletons)
│   ├── lama_model.py           # LaMa neural inpainting model
│   └── sdxl_model.py           # Stable Diffusion XL inpainting wrapper
│
├── services/                   # Modular domain & processing services
│   ├── inpainting_service.py   # Inpainting pipeline orchestrator
│   ├── detection_service.py    # Faster R-CNN object detection
│   ├── background_service.py   # rembg / U2Net background cutout
│   ├── enhancement_service.py  # Super-resolution & image filtering
│   ├── image_service.py        # Pillow / OpenCV image preprocessing
│   └── stats_service.py        # Thread-safe usage metrics
│
├── utils/                      # Helper utilities
│   ├── validators.py           # Upload, MIME & mask payload validation
│   ├── file_utils.py           # Secure filenames & retention cleanup
│   └── logger.py               # Rotating file & console logging
│
├── templates/                  # Jinja2 HTML templates
│   ├── base.html               # Base layout, navbar, modals & theme engine
│   └── index.html              # Main Studio UI, canvas, toolbar & hero
│
├── static/                     # Frontend static assets
│   ├── css/style.css           # Glassmorphic design system & themes
│   ├── js/app.js               # Canvas editor controller & client logic
│   ├── img/                    # Hero sample photos & assets
│   ├── uploads/                # Staged source uploads (auto-cleaned)
│   └── results/                # Output images (auto-cleaned)
│
└── logs/                       # Rotating runtime logs
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+
- (Optional, recommended for speed) NVIDIA GPU with CUDA

### 2. Clone & Install

```bash
git clone https://github.com/mhttarunno/Fade-AI.git
cd Fade-AI

# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate       # Windows
# source venv/bin/activate     # macOS / Linux

# Install dependencies
pip install -r requirements.txt
```

> **GPU Acceleration:** For CUDA support, install matching PyTorch wheels before requirements:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 3. Configure Environment

```bash
cp .env.example .env
# Edit .env to customize settings (e.g., SECRET_KEY, INPAINT_BACKEND)
```

### 4. Run Application

```bash
# Using 1-click launcher (Windows)
.\run.bat

# Or using Python directly
python app.py
```

Open your browser at **[http://localhost:5000](http://localhost:5000)**.

---

## ⚙️ Configuration Reference (`.env`)

| Variable | Default | Description |
|---|---|---|
| `FLASK_ENV` | `development` | `development` or `production` |
| `SECRET_KEY` | *(dev key)* | Flask session security key |
| `MAX_UPLOAD_MB` | `15` | Maximum upload size in megabytes |
| `MAX_IMAGE_DIMENSION` | `2048` | Maximum image dimension (auto-downscaled) |
| `INPAINT_BACKEND` | `lama` | `lama` (fast CNN) or `sdxl` (high fidelity diffusion) |
| `DEVICE` | `auto` | `auto`, `cuda`, or `cpu` |
| `MASK_DILATE_PX` | `6` | Pixel dilation to cover boundary halos |
| `FILE_RETENTION_MINUTES` | `60` | Auto-purging retention window for temporary files |

---

## 🔌 API Reference

### `POST /api/upload`
Uploads and validates a source image. Field: `image`.
```json
{ "success": true, "filename": "sample_photo.jpg", "url": "/static/uploads/sample_photo.jpg" }
```

### `POST /api/remove-object`
Removes objects bounded by a binary mask data URL.
```json
// Request Body:
{ "filename": "sample_photo.jpg", "mask": "data:image/png;base64,..." }

// Response:
{ "success": true, "result_url": "/static/results/result.png", "processing_time_seconds": 1.25 }
```

### `POST /api/detect-objects`
Runs object detection and returns COCO bounding boxes with category labels.
```json
{ "filename": "sample_photo.jpg" }
```

### `POST /api/remove-background`
Extracts foreground and creates a transparent PNG.
```json
{ "filename": "sample_photo.jpg" }
```

### `POST /api/enhance-image`
Upscales (2x/4x) and enhances image clarity and contrast.
```json
{ "filename": "sample_photo.jpg", "scale": 2, "denoise": true, "sharpen": true }
```

### `GET /api/stats`
Returns system telemetry, latency, and tool usage statistics.

### `GET /api/health`
Readiness check reporting device, acceleration status, and active features.

---

## 📄 License

Distributed under the MIT License. Upstream model weights (LaMa, SDXL, U2Net) remain subject to their respective licenses.
