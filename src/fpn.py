# src/fpn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src import config

class FPN(nn.Module):
    def __init__(self, c3_in_channels, c4_in_channels, c5_in_channels, out_channels):
        """
        Args:
            c3_in_channels (int): Number of channels for C3 feature map from backbone.
            c4_in_channels (int): Number of channels for C4 feature map from backbone.
            c5_in_channels (int): Number of channels for C5 feature map from backbone.
            out_channels (int): Number of output channels for all FPN pyramid levels (P3-P6).
        """
        super().__init__()
        self.out_channels = out_channels

        # Lateral layers (1x1 convolutions to match channels)
        self.lat_c5 = nn.Conv2d(c5_in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.lat_c4 = nn.Conv2d(c4_in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.lat_c3 = nn.Conv2d(c3_in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        # Top-down pathway smoothing layers (3x3 convolutions)
        # These are applied after merging lateral and upsampled top-down features
        self.smooth_p4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.smooth_p3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # P6 layer: often created by a 3x3 conv with stride 2 on P5 (or C5)
        # This further downsamples to detect larger objects
        self.p6_conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
        # Alternative for P6 using C5 directly:
        # self.p6_conv_from_c5 = nn.Conv2d(c5_in_channels, out_channels, kernel_size=3, stride=2, padding=1)


        # Initialize weights (optional, but good practice)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1) # ReLU works well with Kaiming init
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _upsample_add(self, x, y):
        """
        Upsamples x and adds it to y.
        x: feature map from top-down pathway (needs upsampling)
        y: feature map from lateral connection (same resolution as upsampled x)
        """
        _, _, H, W = y.shape
        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def forward(self, features):
        """
        Args:
            features (Dict[str, torch.Tensor]): Dictionary from backbone {'c3': tensor, 'c4': tensor, 'c5': tensor}
        Returns:
            Dict[str, torch.Tensor]: Feature pyramid maps {'p3': tensor, 'p4': tensor, 'p5': tensor, 'p6': tensor}
        """
        c3, c4, c5 = features['c3'], features['c4'], features['c5']

        # Bottom-up to Top-down pathway + Lateral connections
        # P5: Apply 1x1 conv to C5
        p5 = self.lat_c5(c5)  # Shape: [B, out_channels, H/32, W/32]

        # P4: Upsample P5, add to 1x1 convolved C4, then apply 3x3 smooth conv
        # Lateral connection from C4
        lat_c4 = self.lat_c4(c4) # Shape: [B, out_channels, H/16, W/16]
        # Top-down from P5 (upsampled) + lateral C4
        p4_merged = self._upsample_add(p5, lat_c4)
        p4 = self.smooth_p4(p4_merged) # Shape: [B, out_channels, H/16, W/16]

        # P3: Upsample P4, add to 1x1 convolved C3, then apply 3x3 smooth conv
        # Lateral connection from C3
        lat_c3 = self.lat_c3(c3) # Shape: [B, out_channels, H/8, W/8]
        # Top-down from P4 (upsampled) + lateral C3
        p3_merged = self._upsample_add(p4, lat_c3)
        p3 = self.smooth_p3(p3_merged) # Shape: [B, out_channels, H/8, W/8]

        # P6: Apply a 3x3 stride-2 conv on P5
        # P6 provides features for detecting larger objects
        p6 = self.p6_conv(p5) # Shape: [B, out_channels, H/64, W/64]
        # If using p6_conv_from_c5:
        # p6 = self.p6_conv_from_c5(c5)

        return {'p3': p3, 'p4': p4, 'p5': p5, 'p6': p6}

if __name__ == '__main__':
    # Test the FPN module
    # Assuming backbone.py is in the same src directory
    from src.backbone import ResNetBackbone

    # Initialize backbone
    backbone_model = ResNetBackbone(backbone_name=config.BACKBONE, pretrained=False) # No need for pretrained weights for shape testing
    backbone_model.to(config.DEVICE)
    backbone_model.eval()

    # Initialize FPN
    # Get channel dimensions from the backbone instance
    fpn_module = FPN(
        c3_in_channels=backbone_model.c3_out_channels,
        c4_in_channels=backbone_model.c4_out_channels,
        c5_in_channels=backbone_model.c5_out_channels,
        out_channels=config.FPN_OUT_CHANNELS
    ).to(config.DEVICE)
    fpn_module.eval()

    dummy_input = torch.randn(2, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(config.DEVICE)
    backbone_features = backbone_model(dummy_input)
    fpn_features = fpn_module(backbone_features)

    print("FPN output feature shapes:")
    for name, tensor in fpn_features.items():
        print(f"{name}: {tensor.shape} (Stride: {config.IMAGE_SIZE // tensor.shape[-1]})")

    # Expected output shapes for IMAGE_SIZE=640, FPN_OUT_CHANNELS=256:
    # p3: torch.Size([2, 256, 80, 80]) (Stride: 8)
    # p4: torch.Size([2, 256, 40, 40]) (Stride: 16)
    # p5: torch.Size([2, 256, 20, 20]) (Stride: 32)
    # p6: torch.Size([2, 256, 10, 10]) (Stride: 64)