

# src/heads.py
import torch
import torch.nn as nn
from src import config

class DetectionHeads(nn.Module):
    def __init__(self, in_channels, num_anchors_per_level, num_classes, num_convs=4):
        """
        Args:
            in_channels (int): Number of input channels from FPN levels (e.g., config.FPN_OUT_CHANNELS).
            num_anchors_per_level (int): Number of anchors generated per spatial location on an FPN map.
                                       (e.g., config.NUM_ANCHORS_PER_LEVEL)
            num_classes (int): Number of object classes to detect (e.g., config.NUM_CLASSES).
            num_convs (int): Number of intermediate convolutional layers in each head.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors_per_level

        # Classification Head
        cls_layers = []
        for _ in range(num_convs):
            cls_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            cls_layers.append(nn.ReLU(inplace=True))
        # Final conv layer for classification scores
        # Output channels: num_anchors * num_classes
        # For each anchor, predict scores for each class
        self.cls_subnet_output = nn.Conv2d(in_channels, num_anchors_per_level * num_classes,
                                           kernel_size=3, stride=1, padding=1)
        self.classification_subnet = nn.Sequential(*cls_layers)

        # Bounding Box Regression Head
        reg_layers = []
        for _ in range(num_convs):
            reg_layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            reg_layers.append(nn.ReLU(inplace=True))
        # Final conv layer for bounding box offsets
        # Output channels: num_anchors * 4 (for dx, dy, dw, dh)
        self.reg_subnet_output = nn.Conv2d(in_channels, num_anchors_per_level * 4,
                                           kernel_size=3, stride=1, padding=1)
        self.regression_subnet = nn.Sequential(*reg_layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # Special initialization for the final layer of the classification subnet (RetinaNet practice)
        # Prioritizes background class at the beginning of training
        # Helps with stability when most anchors are background
        pi = 0.01  # Prior probability for foreground
        bias_value = -torch.log(torch.tensor((1.0 - pi) / pi))
        nn.init.constant_(self.cls_subnet_output.bias, bias_value)
        nn.init.normal_(self.cls_subnet_output.weight, std=0.01) # Small std for weights

        nn.init.normal_(self.reg_subnet_output.weight, std=0.01) # Small std for weights
        nn.init.constant_(self.reg_subnet_output.bias, 0)


    def forward(self, fpn_features_list):
        """
        Args:
            fpn_features_list (List[torch.Tensor]): A list of feature maps from FPN levels [P3, P4, P5, P6].
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                - cls_outputs_list: List of classification logits for each FPN level.
                                    Each tensor shape: [B, H_level * W_level * A, C]
                - reg_outputs_list: List of regression deltas for each FPN level.
                                    Each tensor shape: [B, H_level * W_level * A, 4]
        """
        cls_outputs_list = []
        reg_outputs_list = []

        for feature_map in fpn_features_list:
            # Classification pass
            cls_intermediate = self.classification_subnet(feature_map)
            cls_logits = self.cls_subnet_output(cls_intermediate) # Shape: [B, A*C, H_level, W_level]

            # Regression pass
            reg_intermediate = self.regression_subnet(feature_map)
            reg_deltas = self.reg_subnet_output(reg_intermediate) # Shape: [B, A*4, H_level, W_level]

            # Reshape outputs for easier processing later:
            # cls_logits: [B, A*C, H, W] -> [B, H, W, A*C] -> [B, H*W*A, C]
            B, _, H, W = cls_logits.shape
            cls_logits = cls_logits.permute(0, 2, 3, 1).contiguous()
            cls_logits = cls_logits.view(B, -1, self.num_classes) # Each row is an anchor's class scores
            cls_outputs_list.append(cls_logits)

            # reg_deltas: [B, A*4, H, W] -> [B, H, W, A*4] -> [B, H*W*A, 4]
            reg_deltas = reg_deltas.permute(0, 2, 3, 1).contiguous()
            reg_deltas = reg_deltas.view(B, -1, 4) # Each row is an anchor's box deltas
            reg_outputs_list.append(reg_deltas)

        return cls_outputs_list, reg_outputs_list


if __name__ == '__main__':
    from src.fpn import FPN # Assuming FPN is in src
    from src.backbone import ResNetBackbone # Assuming Backbone is in src

    # Initialize backbone
    backbone_model = ResNetBackbone(backbone_name=config.BACKBONE, pretrained=False)
    backbone_model.eval()

    # Initialize FPN
    fpn_module = FPN(
        c3_in_channels=backbone_model.c3_out_channels,
        c4_in_channels=backbone_model.c4_out_channels,
        c5_in_channels=backbone_model.c5_out_channels,
        out_channels=config.FPN_OUT_CHANNELS
    )
    fpn_module.eval()

    # Initialize Detection Heads
    # Note: config.NUM_CLASSES should be number of actual object classes (e.g., 2 for helmet, vest)
    # The classification head will output logits for these classes.
    # Background is often handled by having no class score high enough, or by adding a background class.
    # For Focal Loss, it's common to predict K object classes, and the "background"
    # is implicitly handled by the loss function if no foreground class gets a high score.
    detection_heads = DetectionHeads(
        in_channels=config.FPN_OUT_CHANNELS,
        num_anchors_per_level=config.NUM_ANCHORS_PER_LEVEL,
        num_classes=config.NUM_CLASSES # e.g., 2 for helmet, vest
    )
    detection_heads.eval()

    # Move to device
    device = config.DEVICE
    backbone_model.to(device)
    fpn_module.to(device)
    detection_heads.to(device)

    # Create dummy input and pass through the pipeline
    batch_size = 2
    dummy_input = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)

    backbone_features = backbone_model(dummy_input)
    fpn_output_dict = fpn_module(backbone_features)
    
    # The heads expect a list of FPN feature maps [P3, P4, P5, P6]
    # Make sure the order matches how anchors will be generated
    fpn_features_list = [fpn_output_dict['p3'], fpn_output_dict['p4'], fpn_output_dict['p5'], fpn_output_dict['p6']]

    cls_preds_list, reg_preds_list = detection_heads(fpn_features_list)

    print("Detection Heads Output Shapes (per FPN level):")
    level_names = ['P3', 'P4', 'P5', 'P6']
    for i, (cls_pred, reg_pred) in enumerate(zip(cls_preds_list, reg_preds_list)):
        level_name = level_names[i]
        fpn_level_shape = fpn_features_list[i].shape # B, C, H_level, W_level
        H_level, W_level = fpn_level_shape[2], fpn_level_shape[3]
        num_anchors_on_level = H_level * W_level * config.NUM_ANCHORS_PER_LEVEL
        
        print(f"--- {level_name} (Input HxW: {H_level}x{W_level}) ---")
        print(f"  Class Preds Shape: {cls_pred.shape}") # Expected: [B, H_level*W_level*A, NUM_CLASSES]
        assert cls_pred.shape == (batch_size, num_anchors_on_level, config.NUM_CLASSES)
        print(f"  Reg Preds Shape:   {reg_pred.shape}")   # Expected: [B, H_level*W_level*A, 4]
        assert reg_pred.shape == (batch_size, num_anchors_on_level, 4)

    # Example expected output for P3 (80x80 feature map) with B=2, A=9, NUM_CLASSES=2:
    # Class Preds Shape: torch.Size([2, 80*80*9, 2]) -> [2, 57600, 2]
    # Reg Preds Shape:   torch.Size([2, 80*80*9, 4]) -> [2, 57600, 4]