# Plan: Investigate Model Prediction Issues
## Analysis of Current Findings

### 1. Frontend Preprocessing (`src/frontend/js/canvas.js`)
- **Sizing**: Bounding box is found, padded by 15px, scaled to 20x20 (preserving aspect ratio), and then centered into a 28x28 canvas.
- **Normalization**:
    - Grayscale conversion: `(R+G+B)/3`.
    - Inversion: `255 - avg` (since background is white).
    - Scaling: `inverted / 255.0`.
    - **Result**: Input tensor is in range `[0, 1]`.
- **Format**: Returns `tf.tensor4d(grayscale, [1, 28, 28, 1])` (NHWC).

### 2. Backend Training (`src/backend/training/train.py`)
- **Normalization**:
    - `transforms.ToTensor()`: Scales pixels to `[0, 1]`.
    - `transforms.Normalize(mean=cfg["mean"], std=cfg["std"])`:
        - MNIST: `mean=(0.1307,), std=(0.3081,)`
        - EMNIST: `mean=(0.1751,), std=(0.3332,)`
    - **Result**: Input tensor is standardized (Z-score normalized), not just `[0, 1]`.
- **Format**: PyTorch expects NCHW (`[batch, 1, 28, 28]`).

### 3. Inference Logic (`src/frontend/js/inference.js`)
- **Shape Handling**: Reshapes the tensor to match `model.inputs[0].shape`.
- **Preprocessing**: Relies on `canvas.js` which only provides `[0, 1]` scaling.

## Identified Mismatches

| Feature | Frontend (Inference) | Backend (Training) | Mismatch? |
| :--- | :--- | :--- | :--- |
| **Normalization** | `[0, 1]` scaling | Z-score `(x - mean) / std` | **YES** |
| **Data Format** | NHWC (`[1, 28, 28, 1]`) | NCHW (`[1, 1, 28, 28]`) | **YES** (Potential) |
| **Sizing/Centering** | Bounding box $\rightarrow$ 20x20 $\rightarrow$ 28x28 | Standard MNIST/EMNIST transforms | **Possible** |

## Detailed Breakdown of Mismatches

1.  **Normalization Mismatch (Critical)**: 
    The model was trained on data standardized with specific means and standard deviations. The frontend is providing raw `[0, 1]` values. This is the most likely cause for the model always returning the same output (the input is shifted far from the trained distribution).
    
2.  **Data Format (NHWC vs NCHW)**:
    PyTorch uses NCHW. TensorFlow.js (Layers API) typically uses NHWC. When converting a PyTorch model to TFJS, the conversion tool usually handles the transpose of weights, but the input tensor must match what the resulting TFJS model expects.
    - `inference.js` reads `model.inputs[0].shape`. If the converted model expects NHWC, the current `[1, 28, 28, 1]` is correct.
    - However, if the model was converted without layout transformation, it might be expecting NCHW or a flattened vector.

## Proposed Fixes

1.  **Update Frontend Normalization**:
    - Implement the Z-score normalization in `canvas.js` using the constants found in `train.py`.
    - For MNIST: `(val - 0.1307) / 0.3081`
    - For EMNIST: `(val - 0.1751) / 0.3332`

2.  **Verify Tensor Layout**:
    - Confirm the expected input shape of the loaded TFJS model.
    - If there is a mismatch between PyTorch (NCHW) and TFJS (NHWC) that wasn't handled during conversion, the tensor needs to be transposed.

## Execution Steps

1.  [ ] Modify `src/frontend/js/canvas.js` to include Z-score normalization based on the dataset.
2.  [ ] Modify `src/frontend/js/inference.js` (or `canvas.js`) to ensure the tensor layout matches the model's expectation.
3.  [ ] Verify if any other preprocessing (like the 20x20 scaling) differs significantly from the training set distribution.
