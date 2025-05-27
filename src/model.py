

# src/model.py
import torch
import torch.nn as nn
from src import config
from src.backbone import ResNetBackbone
from src.fpn import FPN
from src.heads import DetectionHeads

class PPEObjectDetector(nn.Module):
    def __init__(self, backbone_name, fpn_out_channels, num_classes,
                 num_anchors_per_level, fpn_levels_to_use=None, pretrained_backbone=True):
        """
        Args:
            backbone_name (str): Name of the backbone (e.g., 'resnet50').
            fpn_out_channels (int): Output channels for FPN layers.
            num_classes (int): Number of object classes.
            num_anchors_per_level (int): Number of anchors per spatial location on an FPN map.
            fpn_levels_to_use (List[str], optional): FPN levels to use for head predictions.
                                                     Defaults to ['p3', 'p4', 'p5', 'p6'].
            pretrained_backbone (bool): Whether to use a pretrained backbone.
        """
        super().__init__()

        self.backbone = ResNetBackbone(backbone_name=backbone_name, pretrained=pretrained_backbone)

        # Determine FPN levels to use from the FPN output
        if fpn_levels_to_use is None:
            self.fpn_levels_to_use = ['p3', 'p4', 'p5', 'p6'] # Default order
        else:
            self.fpn_levels_to_use = fpn_levels_to_use


        self.fpn = FPN(
            c3_in_channels=self.backbone.c3_out_channels,
            c4_in_channels=self.backbone.c4_out_channels,
            c5_in_channels=self.backbone.c5_out_channels,
            out_channels=fpn_out_channels
        )

        self.detection_heads = DetectionHeads(
            in_channels=fpn_out_channels,
            num_anchors_per_level=num_anchors_per_level,
            num_classes=num_classes
        )

    def forward(self, images):
        """
        Args:
            images (torch.Tensor): Batch of input images, shape (B, C, H, W).
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                - cls_preds_list: List of classification logits for each FPN level.
                - reg_preds_list: List of regression deltas for each FPN level.
        """
        # 1. Get backbone features (c3, c4, c5)
        backbone_features = self.backbone(images) # Returns a dict {'c3': ..., 'c4': ..., 'c5': ...}

        # 2. Get FPN features (p3, p4, p5, p6)
        fpn_output_dict = self.fpn(backbone_features) # Returns a dict {'p3': ..., 'p4': ..., ...}

        # 3. Select the FPN features to pass to the heads in the correct order
        fpn_features_for_heads = [fpn_output_dict[level_name] for level_name in self.fpn_levels_to_use]

        # 4. Get predictions from detection heads
        # cls_preds_list: list of [B, H_level*W_level*A, NUM_CLASSES]
        # reg_preds_list: list of [B, H_level*W_level*A, 4]
        cls_preds_list, reg_preds_list = self.detection_heads(fpn_features_for_heads)

        return cls_preds_list, reg_preds_list

    def freeze_backbone(self):
        print("Freezing backbone weights.")
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        print("Unfreezing backbone weights.")
        for param in self.backbone.parameters():
            param.requires_grad = True

    def freeze_fpn(self):
        print("Freezing FPN weights.")
        for param in self.fpn.parameters():
            param.requires_grad = False
    
    def unfreeze_fpn(self):
        print("Unfreezing FPN weights.")
        for param in self.fpn.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    # Load configurations
    device = config.DEVICE
    batch_size = 2

    # Instantiate the model
    model = PPEObjectDetector(
        backbone_name=config.BACKBONE,
        fpn_out_channels=config.FPN_OUT_CHANNELS,
        num_classes=config.NUM_CLASSES,
        num_anchors_per_level=config.NUM_ANCHORS_PER_LEVEL,
        pretrained_backbone=True # Start with pretrained backbone
    ).to(device)

    # model.freeze_backbone() # Optionally freeze backbone for initial training phase

    model.eval() # Set to evaluation mode for testing inference

    # Create a dummy input tensor
    dummy_images = torch.randn(batch_size, 3, config.IMAGE_SIZE, config.IMAGE_SIZE).to(device)

    # Perform a forward pass
    print(f"Input image shape: {dummy_images.shape}")
    cls_preds, reg_preds = model(dummy_images)

    print("\nModel Output (predictions from DetectionHeads):")
    level_names = model.fpn_levels_to_use #['P3', 'P4', 'P5', 'P6']
    for i, (cls_p, reg_p) in enumerate(zip(cls_preds, reg_preds)):
        level_name = level_names[i]
        # Calculate expected total anchors for this level to verify
        # This requires knowing the H, W of the FPN feature map for this level
        # We can infer H,W based on stride (or get it from a test fpn pass)
        # Example for P3 (stride 8): H_level = IMAGE_SIZE/8, W_level = IMAGE_SIZE/8
        # num_anchors_on_level = (config.IMAGE_SIZE // config.ANCHOR_STRIDES[i])**2 * config.NUM_ANCHORS_PER_LEVEL
        
        print(f"--- Level {level_name} ---")
        print(f"  Class Preds Shape: {cls_p.shape}") # Expected: [B, num_total_anchors_at_level, NUM_CLASSES]
        print(f"  Reg Preds Shape:   {reg_p.shape}")   # Expected: [B, num_total_anchors_at_level, 4]
        
        # Assertions for shape consistency
        # Example calculation for total anchors at P3 for image_size=640, stride=8, A=9
        # H_p3 = 640/8 = 80, W_p3 = 80. Total_anchors_p3 = 80*80*9 = 57600
        # cls_p.shape should be [B, 57600, NUM_CLASSES]
        # reg_p.shape should be [B, 57600, 4]
        
        # A more robust way to check total anchors based on actual output shape
        num_total_anchors_at_level_from_output = cls_p.shape[1]
        assert cls_p.shape == (batch_size, num_total_anchors_at_level_from_output, config.NUM_CLASSES)
        assert reg_p.shape == (batch_size, num_total_anchors_at_level_from_output, 4)

    print(f"\nModel created successfully on {device}!")
    
    # Test freezing/unfreezing (optional)
    # model.freeze_backbone()
    # for name, param in model.backbone.named_parameters():
    #     if param.requires_grad == False:
    #         print(f"Frozen: {name}")
    #         break # just check one
    
    # model.unfreeze_backbone()
    # for name, param in model.backbone.named_parameters():
    #     if param.requires_grad == True:
    #         print(f"Unfrozen: {name}")
    #         break