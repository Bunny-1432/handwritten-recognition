"""
export_tfjs.py
Export the trained PyTorch CNN model to TensorFlow.js format.
Requires:
  pip install onnx onnx2tf tensorflowjs
"""

import os
import torch
import torch.onnx
from model import HandwrittenCNN

def export_model(weights_path='models/mnist_cnn.pth', output_dir='../web/data/model'):
    if not os.path.exists(weights_path):
        print(f"Weights not found at {weights_path}. Train the model first.")
        return

    # Load Model
    model = HandwrittenCNN(num_classes=10)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()

    # 1. Export to ONNX
    onnx_path = weights_path.replace('.pth', '.onnx')
    dummy_input = torch.randn(1, 1, 28, 28)
    
    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path, 
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Exported to ONNX: {onnx_path}")

    # 2. Convert ONNX to TFJS (requires onnx2tf and tensorflowjs_converter in environment)
    # This is a shell command execution wrapper
    os.makedirs(output_dir, exist_ok=True)
    
    print("\nTo complete the export to TFJS, run the following commands in your terminal:")
    print(f"1. pip install onnx2tf tensorflowjs")
    print(f"2. onnx2tf -i {onnx_path} -o tf_saved_model")
    print(f"3. tensorflowjs_converter --input_format=tf_saved_model tf_saved_model {output_dir}")
    print("\nNote: The web demo currently uses a pre-hosted TFJS MNIST model, so this is optional.")

if __name__ == "__main__":
    export_model()
