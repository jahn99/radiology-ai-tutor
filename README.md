# 🩻 Explainable AI for Medical Image Classification

> **Diagnosing Pneumonia in Chest X-Rays with transparency using Grad-CAM and SHAP**

An end-to-end Explainable AI (XAI) system that classifies chest X-ray images as **Normal** or **Pneumonia** and generates visual explanations using two complementary techniques — **Grad-CAM** and **SHAP** — along with a quantitative comparison module.

---

## ✨ Features

- **Binary Classification** — ResNet18 (transfer learning) for Normal vs Pneumonia detection
- **Grad-CAM Heatmaps** — Visual explanations highlighting diagnostic regions via gradient-based localization
- **SHAP Pixel Importance** — Fine-grained pixel-level contribution maps using Shapley values
- **Comparison Module** — IoU-based overlap analysis between the two explanation methods
- **Interactive UI** — Modern dark-themed Streamlit dashboard with sample images and file upload

---

## 📁 Project Structure

```
Xai_project/
├── app/
│   └── main.py              # Streamlit web application
├── data/                     # Dataset (auto-downloaded)
│   ├── images/               # Chest X-ray images
│   ├── metadata.csv          # Full dataset metadata
│   └── subset_metadata.csv   # Filtered subset with labels
├── models/
│   ├── resnet_wrapper.py     # ResNet18 model architecture
│   └── resnet18_weights.pth  # Trained model weights (generated)
├── utils/
│   └── data_subset.py        # Dataset download & preparation script
├── train.py                  # Training pipeline
├── inference.py              # Prediction wrapper
├── gradcam.py                # Grad-CAM implementation (pytorch-grad-cam)
├── shap_explainer.py         # SHAP implementation (GradientExplainer)
├── comparison.py             # XAI method comparison (IoU)
├── requirements.txt          # Python dependencies
└── README.md
```

---

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.9+ (tested on 3.14)
- pip

### 1. Clone the Repository

```bash
git clone https://github.com/Aneeshrao123/XAI_ORANGE.git
cd XAI_ORANGE
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note (macOS with Homebrew Python):** If you encounter an `externally-managed-environment` error, use:
> ```bash
> python3 -m pip install -r requirements.txt --break-system-packages
> ```
> Or use a virtual environment:
> ```bash
> python3 -m venv venv
> source venv/bin/activate
> pip install -r requirements.txt
> ```

### 3. Download Dataset

This script downloads a balanced subset of chest X-ray images from the [ieee8023/covid-chestxray-dataset](https://github.com/ieee8023/covid-chestxray-dataset):

```bash
python3 utils/data_subset.py
```

### 4. Train the Model

Trains a ResNet18 classifier (transfer learning, ~5 epochs) and saves weights:

```bash
python3 train.py
```

Expected output:
```
Starting training on cpu...
Epoch 1/5: 100%|████████████| 8/8 [00:00<00:00]
Loss: 0.6379
...
Model saved to models/resnet18_weights.pth
```

### 5. Launch the Application

```bash
streamlit run app/main.py
```

Open your browser at **http://localhost:8501**.

---

## 🚀 Usage

1. **Select a sample image** from the sidebar dropdown, or **upload your own** chest X-ray (JPG/PNG)
2. Click **"Generate Diagnosis & Explanations"**
3. View the results across three tabs:
   - **Grad-CAM** — Heatmap overlay showing regions of interest
   - **SHAP** — Pixel importance map
   - **Comparison** — IoU score and overlay visualization (Red = Grad-CAM, Blue = SHAP, Magenta = Overlap)
4. Adjust the **XAI ROI Threshold** slider to control sensitivity of the comparison

---

## 🔬 How It Works

### Model
- **Architecture**: ResNet18 pretrained on ImageNet
- **Transfer Learning**: Base layers frozen, only the classification head is trained
- **Output**: Raw logit → Sigmoid applied at inference time

### Grad-CAM
- Hooks into `model.layer4[-1]` (last convolutional layer)
- Computes gradient-weighted activation maps
- Produces a coarse localization heatmap
- Based on: [Grad-CAM: Visual Explanations from Deep Networks](https://arxiv.org/abs/1610.02391)

### SHAP
- Uses `GradientExplainer` with a background set of training images
- Computes Shapley values for each pixel
- Produces fine-grained importance maps
- Based on: [A Unified Approach to Interpreting Model Predictions](https://arxiv.org/abs/1705.07874)

### Comparison
- Thresholds both heatmaps to create binary ROI masks
- Computes **Intersection over Union (IoU)** between the two masks
- Generates a qualitative agreement summary

---

## 📦 Key Dependencies

| Package | Purpose |
|---------|---------|
| `torch` / `torchvision` | Deep learning framework |
| `grad-cam` | Grad-CAM implementation ([pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam)) |
| `shap` | SHAP explainability ([slundberg/shap](https://github.com/slundberg/shap)) |
| `streamlit` | Web application framework |
| `opencv-python` | Image processing |
| `matplotlib` | Visualization |

---

## 📄 References

1. Selvaraju, R.R. et al. — *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization* — [arXiv:1610.02391](https://arxiv.org/abs/1610.02391)
2. Lundberg, S.M. & Lee, S.I. — *A Unified Approach to Interpreting Model Predictions* — [arXiv:1705.07874](https://arxiv.org/abs/1705.07874)
3. Cohen, J.P. et al. — *COVID-19 Image Data Collection* — [GitHub](https://github.com/ieee8023/covid-chestxray-dataset)

---

## 📝 License

This project is for educational and research purposes.
