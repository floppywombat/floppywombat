import torch
import argparse
import onnx
from fastai.vision.all import load_learner, DynamicUnet
from pathlib import Path
from torchvision.models import resnet34

# Add DynamicUnet to safe globals
torch.serialization.add_safe_globals([DynamicUnet])

def convert_to_onnx(model_path, output_path):
    """
    Convert FastAI model (dynamic UNet with ResNet34 backbone) to ONNX format.
    """
    model_path = Path(model_path)
    
    # Load model based on file extension
    if model_path.suffix == '.pkl':
        # Load FastAI pickle model
        learn = load_learner(model_path)
        model = learn.model
    else:
        # Assuming it's a PyTorch state dict
        try:
            # Load the complete model
            model = torch.load(model_path, map_location=torch.device('cpu'))  # Removed weights_only=True
            if not isinstance(model, DynamicUnet):
                # If we need to create the architecture
                backbone = resnet34(weights=None)
                encoder = torch.nn.Sequential(*list(backbone.children())[:-2])
                new_model = DynamicUnet(encoder, n_out=1, img_size=(512, 512))
                new_model.load_state_dict(model)
                model = new_model
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")
    
    model.eval()
    
    # Create dummy input matching the expected input size
    # Model expects images in the format (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 512, 512)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=True
    )
    
    # Verify the exported model
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    
    print(f"\nModel exported to {output_path}")
    print("\nModel inputs:")
    for input in onnx_model.graph.input:
        print(f"{input.name}: {[d.dim_value for d in input.type.tensor_type.shape.dim]}")
    print("\nModel outputs:")
    for output in onnx_model.graph.output:
        print(f"{output.name}: {[d.dim_value for d in output.type.tensor_type.shape.dim]}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a FastAI/PyTorch Dynamic UNet model (with ResNet34 backbone) to ONNX format."
    )
    parser.add_argument("model_path", type=str, help="Path to the model file (.pkl for FastAI or .pt for PyTorch)")
    parser.add_argument("output_path", type=str, help="Path for the output ONNX file.")

    args = parser.parse_args()
    convert_to_onnx(args.model_path, args.output_path)
