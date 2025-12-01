import torch
from cnn_model import LeafDiseaseResNet18, get_model

# Create model
model = LeafDiseaseResNet18(num_classes=16, pretrained=True)
print(f"Model created successfully!")

# Test with random input
dummy_batch = torch.randn(4, 3, 224, 224)
output = model(dummy_batch)
print(f"Input shape: {dummy_batch.shape}")
print(f"Output shape: {output.shape}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")
