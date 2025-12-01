Facet Detection Pipeline

Description: End-to-End Deep Learning pipeline for Roof Facet Detection from 2D aerial imagery. Utilizes Multi-Task Segmentation (Roof Area + Raw Edges) followed by a dedicated Shape Refinement Network to repair topological errors (broken edges) and ensure vectorization results in clean, CAD-ready polygons (Douglas-Peucker).

Repository Structure

The project uses a nested structure, separating the two core modeling stages:

roof_estimator: Multi-Task Segmentation (Stage 1)

polygon_estimator: Shape Refinement / Polygonization (Stage 2)

/
├── config.yaml                  # Ana Pipeline ve Model Yollarını İçeren Config
├── inference.py                 # 🚀 ANA INFERENCE SCRIPTI (Çıkarım ve Poligonlaştırmayı Yönetir)
├── dataset/                     # Unified dataset placeholder
├── docs/
│   └── technical_report.md      # Detailed technical document
│
├── roof_estimator/              # Stage 1: Multi-Task Model (Roof Area + Raw Edge)
│   ├── config.yaml              # Hyperparameters for Stage 1 Training
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── roof_edge_detection_model.pth
│
└── polygon_estimator/           # Stage 2: Shape Refinement & Final Polygonization
    ├── config.yaml              # Hyperparameters for Stage 2 Training
    ├── dataset.py
    ├── model.py
    ├── train.py
    └── shape_refinement_model.pth



🛠️ Setup Instructions

Prerequisites

Python 3.8+

CUDA or MPS (for GPU acceleration, optional)

Environment Setup

Create and activate a virtual environment, then install dependencies:

# 1. Create environment (e.g., using conda)
conda create -n solarvis python=3.10
conda activate solarvis

# 2. Install PyTorch (Choose based on your hardware: CUDA, MPS, or CPU)
# Example for CUDA 11.8:
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 3. Install required libraries
pip install -r requirements.txt
# (Assuming requirements.txt contains: segmentation-models-pytorch, albumentations, pyyaml, opencv-python, tqdm)



🚂 Example Training Commands

Before running, ensure paths in the respective config.yaml files point to your local data directories.

1. Stage 1: Train Multi-Task Model (Roof Area + Raw Edge)

This model learns the initial segmentation tasks.

# Run training script from within the roof_estimator directory
python roof_estimator/train.py --config roof_estimator/config.yaml



2. Stage 2: Train Shape Refinement Model

This model learns to fix the topological gaps from the Stage 1 output.

# NOTE: Requires Stage 1 output masks to be generated first and placed in the appropriate input directory for Stage 2.
python polygon_estimator/train.py --config polygon_estimator/config.yaml



🚀 Example Inference Command

The main pipeline script inference.py runs both models sequentially (Multi-Task + Refinement) and performs final polygonization logic.

python inference.py \
    --config config.yaml \
    --input_image path/to/your/input_image.png \
    --output_roof_mask outputs/image_roof.png \
    --output_facets_mask outputs/image_facets.png \
    --output_overlay outputs/image_overlay.png
