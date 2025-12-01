# 🏠 Roof Facet Detection Pipeline  
End-to-end deep learning system for extracting **roof facets** from 2D aerial imagery.  
The pipeline integrates *multi-task segmentation*, *topology refinement*, and *polygon post-processing*  
to generate **clean, vector-ready roof polygons** for solar analysis, GIS, and architectural workflows.

---

## 🚀 Overview

This project implements a **two-stage deep learning pipeline**:

### **Stage 1 — Multi-Task Segmentation (`roof_estimator/`)**
Predicts:
- **Roof area mask**
- **Raw roof edges**

Built using a ResNet-based U-Net++ architecture (Segmentation Models PyTorch).

### **Stage 2 — Shape Refinement (`polygon_estimator/`)**
Refines broken / noisy edge masks to:
- Repair topological errors  
- Produce connected, watertight roof boundaries  
- Enable reliable **polygonization** (Douglas-Peucker, contour tracing, etc.)

## 📦 Project Dependencies

The following libraries are required for running training, inference, and preprocessing:

- **PyTorch**
- **Torchvision / Torchaudio** (optional but common)
- **segmentation-models-pytorch**
- **albumentations**
- **opencv-python**
- **numpy**
- **pyyaml**
- **matplotlib**
- **tqdm**
---

## 🛠️ Setup Instructions

### **Prerequisites**
- Python 3.8+
- CUDA or MPS (optional, for GPU acceleration)
- pip or conda

---

## Example Inference Command

The inference script resolves all paths internally:

- Loads its own `config.yaml`
- Uses predefined input image paths
- Uses internal output directories
- Loads checkpoints automatically

No CLI arguments are required.

```bash
python inference.py
