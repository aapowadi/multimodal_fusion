import torch.nn as nn
from timm import create_model
# Define the multimodal ViT model
class MultiModalViT(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(MultiModalViT, self).__init__()
        # Load pretrained ViT-B model (patch size 16, input 224x224)
        self.backbone = create_model('vit_base_patch16_224', pretrained=pretrained)
        # Remove the original classification head
        self.backbone.head = nn.Identity()
        # New head for 5 modalities (5 * 768 from [CLS] tokens)
        self.head = nn.Linear(5 * 768, num_classes)

    def forward(self, x):
        # Input x: (batch_size, 5, 3, 224, 224)
        batch_size = x.shape[0]
        # Reshape to process all modalities through the backbone
        x = x.view(batch_size * 5, 3, 224, 224)
        # Get features, including [CLS] tokens
        features = self.backbone.forward_features(x)  # (batch_size * 5, 197, 768)
        # Extract [CLS] tokens (first token)
        cls_tokens = features[:, 0]  # (batch_size * 5, 768)
        # Reshape to (batch_size, 5 * 768)
        cls_concat = cls_tokens.view(batch_size, -1)
        # Pass through classification head
        logits = self.head(cls_concat)  # (batch_size, num_classes)
        return logits