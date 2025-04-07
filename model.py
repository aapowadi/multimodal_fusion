import torch
import torch.nn as nn
from timm import create_model

class MultiModalViT(nn.Module):
    def __init__(self, num_classes, num_channels_list, pretrained=True):
        """
        MultiModal Vision Transformer for variable channel inputs.

        Args:
            num_classes (int): Number of output classes.
            num_channels_list (list): List of channel counts for each modality (e.g., [2, 13, 7, N_crop, 7, N_weather]).
            pretrained (bool): Whether to use pretrained ViT weights.
        """
        super(MultiModalViT, self).__init__()
        self.num_modalities = len(num_channels_list)
        
        # Create a ViT backbone for each modality with adjusted input channels
        self.backbones = nn.ModuleList()
        for num_channels in num_channels_list:
            # Load pretrained ViT-B and modify the patch embedding for custom channels
            backbone = create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=0)
            # Replace the patch embedding layer (originally 3 channels)
            backbone.patch_embed.proj = nn.Conv2d(
                num_channels, backbone.embed_dim, kernel_size=16, stride=16
            )
            self.backbones.append(backbone)

        # Calculate total feature size (number of modalities * 768 [CLS token size])
        self.feature_size = self.num_modalities * 768
        self.head = nn.Linear(self.feature_size, num_classes)

    def forward(self, x):
        """
        Forward pass for multiple modalities.

        Args:
            x (list): List of tensors, each of shape (batch_size, C_i, 224, 224) where C_i varies.

        Returns:
            torch.Tensor: Logits of shape (batch_size, num_classes).
        """
        batch_size = x[0].shape[0]
        features = []
        
        # Process each modality through its backbone
        for i, modality in enumerate(x):
            feat = self.backbones[i](modality)  # (batch_size, 768) from [CLS] token
            features.append(feat)
        
        # Concatenate features along the feature dimension
        concat_features = torch.cat(features, dim=1)  # (batch_size, num_modalities * 768)
        logits = self.head(concat_features)  # (batch_size, num_classes)
        return logits

# Example usage with the dataset
num_channels_list = [2, 13, 7, 1, 7, 3]  # Example: [S1, S2, MODIS, Crop (1 var), Soil, Weather (3 vars)]
num_classes = 10  # Adjust based on your task
model = MultiModalViT(num_classes=num_classes, num_channels_list=num_channels_list, pretrained=True)