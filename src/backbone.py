# src/backbone.py
import torch
import torch.nn as nn
import torchvision.models as models
from src import config

class ResNetBackbone(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True):
        super().__init__()
        if backbone_name == 'resnet50':
            backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
            # C3 from layer2, C4 from layer3, C5 from layer4
            self.c3_out_channels = 512
            self.c4_out_channels = 1024
            self.c5_out_channels = 2048
        elif backbone_name == 'resnet101': # Example for another backbone
            backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
            self.c3_out_channels = 512
            self.c4_out_channels = 1024
            self.c5_out_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        # Extract layers up to C5 (layer4)
        # These are the children of the ResNet model
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1 # C2, stride 4 (not used directly by FPN in this example)
        self.layer2 = backbone.layer2 # C3, stride 8
        self.layer3 = backbone.layer3 # C4, stride 16
        self.layer4 = backbone.layer4 # C5, stride 32

        # Freeze backbone layers if desired (common for transfer learning)
        # For fine-tuning, you might unfreeze later or train end-to-end
        # for param in self.parameters():
        #     param.requires_grad = False


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W)
        Returns:
            Dict[str, torch.Tensor]: Feature maps from C3, C4, C5
        """
        x = self.stem(x)
        c1 = self.layer1(x) # Not used for FPN input, but computed
        c3 = self.layer2(c1) # Corresponds to ResNet layer2 output
        c4 = self.layer3(c3) # Corresponds to ResNet layer3 output
        c5 = self.layer4(c4) # Corresponds to ResNet layer4 output

        return {'c3': c3, 'c4': c4, 'c5': c5}

if __name__ == '__main__':
    # Test the backbone
    backbone_model = ResNetBackbone(backbone_name=config.BACKBONE, pretrained=True)
    backbone_model.to(config.DEVICE)
    backbone_model.eval() # Set to eval mode for testing

    dummy_input = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)
    features = backbone_model(dummy_input)

    print("Backbone output feature shapes:")
    for name, tensor in features.items():
        print(f"{name}: {tensor.shape}")

    # Expected output shapes for IMAGE_SIZE=640 (may vary slightly based on exact ResNet impl.)
    # c3: torch.Size([2, 512, 80, 80])   (640 / 8 = 80)
    # c4: torch.Size([2, 1024, 40, 40])  (640 / 16 = 40)
    # c5: torch.Size([2, 2048, 20, 20])  (640 / 32 = 20)