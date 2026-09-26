# NeuroScribe — Handwritten Character Recognition

NeuroScribe is a production-grade deep learning system designed to recognize handwritten characters from the MNIST (digits) and EMNIST (alphanumeric) datasets. It implements a high-performance PyTorch-to-ONNX pipeline for stable, cross-platform browser inference.

## 🚀 Features
- **Production-Grade Stability**: Powered by ONNX Runtime Web, bypassing fragile conversion tools.
- **High Accuracy**: Optimized CNN architecture achieving >99% on MNIST and >87% on EMNIST.
- **Live Demo**: Interactive drawing canvas with real-time predictions and confidence scores.
- **3D Visualizations**: Integrated Three.js views of the CNN architecture and training analytics.
- **Strict Normalization**: Bit-perfect Z-score standardization alignment between training and inference.

## 🛠️ Project Structure
```text
.
├── data/                   # Model weights (.pth, .onnx) and training metrics
├── src/
│   ├── backend/            # PyTorch training & export logic
│   │   ├── core/           # Model definitions (HandwrittenCNN)
│   │   ├── training/       # Unified training scripts
│   │   └── utils/          # ONNX export & evaluation helpers
│   └── frontend/           # Web application (HTML, CSS, JS)
├── LICENSE                # MIT License
└── README.md               # Technical Specification
```

## ⚙️ Installation & Setup

### Backend (Training)
1. Clone the repository:
   ```bash
   git clone https://github.com/Bunny-1432/handwritten-recognition.git
   cd handwritten-recognition
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision onnx
   ```
3. Train and Export:
   ```bash
   # Train and export MNIST
   python src/backend/training/fast_train.py --dataset mnist
   python src/backend/utils/export_onnx.py --dataset mnist

   # Train and export EMNIST
   python src/backend/training/fast_train.py --dataset emnist
   python src/backend/utils/export_onnx.py --dataset emnist
   ```

### Frontend (Deployment)
- **Live Demo (GitHub Pages)**: [https://bunny-1432.github.io/handwritten-recognition/](https://bunny-1432.github.io/handwritten-recognition/)
- **Local Server**:
  ```bash
  npm run dev
  # → http://localhost:3000
  ```


## 🧠 Technical Specification

### 1. Architecture
The system uses a Convolutional Neural Network (CNN) with:
- **Feature Extraction**: 3 Convolutional blocks with Batch Normalization and ReLU.
- **Global Average Pooling**: Fixed spatial resolution ($1 \times 1$) to ensure model stability.
- **Classifier**: Fully connected head with Dropout (0.5) for regularization.

### 2. Deployment Pipeline (The "Production" Path)
To ensure stability, the project uses the **ONNX (Open Neural Network Exchange)** standard:
`PyTorch (.pth)` $\to$ `ONNX (.onnx)` $\to$ `ONNX Runtime Web`

This eliminates dependency conflicts between NumPy and TensorFlow.js converters.

### 3. Preprocessing & Normalization
Accuracy is ensured by strict Z-score normalization:
$$\text{Input} = \frac{\text{Pixel} - \mu}{\sigma}$$
- **MNIST**: $\mu = 0.1307, \sigma = 0.3081$
- **EMNIST**: $\mu = 0.1751, \sigma = 0.3332$

## 📊 Performance
| Dataset | Classes | Test Accuracy | Params |
|---------|---------|----------------|--------|
| MNIST   | 10      | 99.2%          | ~335K   |
| EMNIST  | 62      | 87.3%          | ~545K   |
