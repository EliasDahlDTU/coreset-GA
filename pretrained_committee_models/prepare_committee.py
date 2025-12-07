"""
Download and prepare pretrained committee models for MNIST.

Downloads ImageNet-pretrained models from torchvision and adapts them for MNIST
by modifying the final classification layer to output 10 classes.
"""

import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
import config


def adapt_resnet18(num_classes=10):
    """Load ResNet18 and adapt for MNIST."""
    model = models.resnet18(weights='IMAGENET1K_V1')
    
    # ResNet expects 3 channels, MNIST has 1 - add a conv layer to convert
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    
    # Replace final layer for 10 classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model


def adapt_vgg11(num_classes=10):
    """Load VGG11 and adapt for MNIST."""
    model = models.vgg11(weights='IMAGENET1K_V1')
    
    # VGG expects 3 channels, MNIST has 1 - modify first conv layer
    model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
    
    # Replace classifier for 10 classes
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    
    return model


def adapt_mobilenet_v2(num_classes=10):
    """Load MobileNetV2 and adapt for MNIST."""
    model = models.mobilenet_v2(weights='IMAGENET1K_V1')
    
    # MobileNet expects 3 channels, MNIST has 1 - modify first conv layer
    model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    
    # Replace classifier for 10 classes
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model


def prepare_committee_models(
    model_names=None,
    num_classes=10,
    output_dir=None,
    device=None
):
    """
    Download and prepare all committee models.
    
    Args:
        model_names: List of model names to prepare. If None, uses config.COMMITTEE_MODEL_NAMES
        num_classes: Number of output classes (default: 10 for MNIST)
        output_dir: Directory to save models. If None, uses config.PRETRAINED_COMMITTEE_MODELS_DIR
        device: Device to load models on. If None, uses config.COMMITTEE_DEVICE
    """
    if model_names is None:
        model_names = config.COMMITTEE_MODEL_NAMES
    
    if output_dir is None:
        output_dir = config.PRETRAINED_COMMITTEE_MODELS_DIR
    
    if device is None:
        device = config.COMMITTEE_DEVICE
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Model adapters
    adapters = {
        "resnet18": adapt_resnet18,
        "vgg11": adapt_vgg11,
        "mobilenet_v2": adapt_mobilenet_v2
    }
    
    print(f"Preparing {len(model_names)} committee models for {num_classes} classes...")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}\n")
    
    prepared_models = {}
    
    for model_name in model_names:
        if model_name not in adapters:
            print(f"⚠️  Warning: Unknown model '{model_name}', skipping...")
            continue
        
        print(f"📥 Downloading and adapting {model_name}...")
        
        try:
            # Adapt model
            model = adapters[model_name](num_classes=num_classes)
            
            # Move to device
            model = model.to(device)
            model.eval()
            
            # Save model
            model_path = output_dir / f"{model_name}.pth"
            torch.save(model.state_dict(), model_path)
            
            # Also save full model for easier loading
            full_model_path = output_dir / f"{model_name}_full.pth"
            torch.save(model, full_model_path)
            
            prepared_models[model_name] = model_path
            print(f"  ✓ Saved to {model_path}")
            print(f"  ✓ Full model saved to {full_model_path}\n")
            
        except Exception as e:
            print(f"  ✗ Error preparing {model_name}: {e}\n")
            continue
    
    print(f"✓ Successfully prepared {len(prepared_models)}/{len(model_names)} models")
    
    return prepared_models


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare pretrained committee models for MNIST")
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help="Model names to prepare (default: from config)"
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=10,
        help="Number of output classes (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: from config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (default: from config)"
    )
    
    args = parser.parse_args()
    
    prepare_committee_models(
        model_names=args.models,
        num_classes=args.num_classes,
        output_dir=args.output_dir,
        device=args.device
    )

