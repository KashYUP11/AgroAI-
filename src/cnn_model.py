import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F


class ChannelAttention(nn.Module):
    """
    Channel Attention Module - focuses on "what" information.
    Uses squeeze-and-excitation approach.
    """

    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module - focuses on "where" information.
    Uses channel compression approach.
    """

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_concat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(x_concat)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM)
    Combines Channel and Spatial attention mechanisms.
    """

    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.channel_attention(x)
        out = out * self.spatial_attention(out)
        return out


class LeafDiseaseResNet18(nn.Module):
    """
    ResNet18-based CNN with CBAM attention for Leaf Disease Classification.

    Architecture:
    - ResNet18 backbone (pre-trained on ImageNet)
    - CBAM attention modules after each residual block
    - Custom classifier head with dropout for regularization
    - Supports variable number of classes
    """

    def __init__(self, num_classes=16, pretrained=True, dropout_rate=0.5):
        """
        Args:
            num_classes (int): Number of disease classes (default: 16)
            pretrained (bool): Use ImageNet pre-trained weights (default: True)
            dropout_rate (float): Dropout rate for regularization (default: 0.5)
        """
        super(LeafDiseaseResNet18, self).__init__()

        self.num_classes = num_classes
        self.dropout_rate = dropout_rate

        # Load pre-trained ResNet18
        resnet18 = models.resnet18(pretrained=pretrained)

        # Extract backbone layers
        self.conv1 = resnet18.conv1
        self.bn1 = resnet18.bn1
        self.relu = resnet18.relu
        self.maxpool = resnet18.maxpool

        # Residual blocks with attention
        self.layer1 = resnet18.layer1
        self.cbam1 = CBAM(64)

        self.layer2 = resnet18.layer2
        self.cbam2 = CBAM(128)

        self.layer3 = resnet18.layer3
        self.cbam3 = CBAM(256)

        self.layer4 = resnet18.layer4
        self.cbam4 = CBAM(512)

        # Global Average Pooling
        self.avgpool = resnet18.avgpool

        # Custom classifier head
        self.fc_layers = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input image tensor of shape (batch, 3, 224, 224)

        Returns:
            torch.Tensor: Logits of shape (batch, num_classes)
        """
        # Initial convolution and pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual blocks with attention
        x = self.layer1(x)
        x = self.cbam1(x) * x  # Apply attention

        x = self.layer2(x)
        x = self.cbam2(x) * x

        x = self.layer3(x)
        x = self.cbam3(x) * x

        x = self.layer4(x)
        x = self.cbam4(x) * x

        # Global pooling and classification
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc_layers(x)

        return x

    def get_feature_maps(self, x, layer='layer3'):
        """
        Extract feature maps from intermediate layers for visualization.

        Args:
            x (torch.Tensor): Input image tensor
            layer (str): Layer name ('layer1', 'layer2', 'layer3', 'layer4')

        Returns:
            torch.Tensor: Feature maps from specified layer
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        if layer == 'layer1':
            return x
        x = self.cbam1(x) * x

        x = self.layer2(x)
        if layer == 'layer2':
            return x
        x = self.cbam2(x) * x

        x = self.layer3(x)
        if layer == 'layer3':
            return x
        x = self.cbam3(x) * x

        x = self.layer4(x)
        return x


class LeafDiseaseModelEnsemble(nn.Module):
    """
    Ensemble of multiple LeafDiseaseResNet18 models for improved predictions.
    Can average predictions from multiple models trained with different seeds.
    """

    def __init__(self, num_models=3, num_classes=16):
        """
        Args:
            num_models (int): Number of models in ensemble
            num_classes (int): Number of disease classes
        """
        super(LeafDiseaseModelEnsemble, self).__init__()
        self.models = nn.ModuleList([
            LeafDiseaseResNet18(num_classes=num_classes)
            for _ in range(num_models)
        ])

    def forward(self, x):
        """Average predictions from all models."""
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        return torch.stack(outputs).mean(dim=0)

    def load_checkpoint(self, model_idx, checkpoint_path):
        """Load pre-trained weights for a specific model."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.models[model_idx].load_state_dict(checkpoint['model_state_dict'])


def get_model(num_classes=16, model_type='single', pretrained=True, dropout_rate=0.5):
    """
    Factory function to create models.

    Args:
        num_classes (int): Number of classes
        model_type (str): 'single' for single model, 'ensemble' for ensemble
        pretrained (bool): Use pre-trained ImageNet weights
        dropout_rate (float): Dropout rate for regularization

    Returns:
        nn.Module: Model instance
    """
    if model_type == 'single':
        return LeafDiseaseResNet18(num_classes, pretrained, dropout_rate)
    elif model_type == 'ensemble':
        return LeafDiseaseModelEnsemble(num_models=3, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


# Example usage and model testing
if __name__ == "__main__":
    # Create model
    model = LeafDiseaseResNet18(num_classes=16, pretrained=True, dropout_rate=0.5)
    print(model)

    # Test with dummy input
    dummy_input = torch.randn(4, 3, 224, 224)  # Batch of 4 images
    output = model(dummy_input)
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Test feature map extraction
    features = model.get_feature_maps(dummy_input, layer='layer3')
    print(f"\nFeature maps from layer3: {features.shape}")
