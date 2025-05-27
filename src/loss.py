

# src/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from src import config # For IOU_THRESHOLD_POSITIVE, IOU_THRESHOLD_NEGATIVE
from src.utils import box_iou # We'll need this for matching targets to anchors

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        """
        Focal Loss for dense object detection.
        Args:
            alpha (float): Weighting factor for positive samples.
            gamma (float): Focusing parameter.
            reduction (str): 'mean', 'sum', or 'none'.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Logits from the model (before sigmoid),
                                   shape [N, num_classes] or [batch_size, total_anchors, num_classes].
            targets (torch.Tensor): Ground truth labels (0 or 1 for binary, or one-hot encoded for multi-class),
                                    same shape as inputs (or broadcastable).
                                    For object detection, targets are often [N, num_classes] where N is num anchors,
                                    and each row is one-hot encoding of the target class for that anchor.
        Returns:
            torch.Tensor: The calculated Focal Loss.
        """
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # pt = p if y=1, 1-p if y=0
        
        # For multi-class, targets are one-hot. We want to apply alpha to positive class only.
        # If targets are [0,0,1,0], alpha_t for that sample would be alpha.
        # If targets are [0,0,0,0] (background), alpha_t would be 1-alpha (implicitly handled if alpha applies to foreground)
        # Simpler: use alpha for positive class, 1-alpha for negative classes
        # This often means for a positive anchor, target class gets alpha, other classes get 1-alpha
        # A common way:
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha) # This needs careful thought for multi-class
                                                                        # For one-hot, alpha_t applies to the one-hot '1' pos
                                                                        # and 1-alpha to the '0' positions.

        # For RetinaNet-style Focal Loss, it's typically applied per class.
        # The target for a positive anchor for class `c` is 1, and 0 for other classes.
        # The target for a negative anchor is 0 for all classes.
        
        # Let's consider inputs [N_anchors, C_classes] and targets [N_anchors, C_classes] (one-hot)
        # pt = p for positive class, 1-p for negative classes
        # The `targets` tensor will be 1 for the true class of positive anchors, and 0 otherwise.
        # For negative anchors, `targets` will be all zeros.
        
        # alpha_factor is self.alpha for positive examples (target=1) and 1-self.alpha for negative examples (target=0)
        alpha_factor = torch.where(targets == 1, self.alpha, torch.tensor(1.0 - self.alpha, device=inputs.device))
        focal_weight = alpha_factor * (1.0 - pt)**self.gamma
        loss = focal_weight * BCE_loss

        if self.reduction == 'mean':
            # Normalize by the number of positive anchors (as in RetinaNet)
            # This requires knowing which anchors were assigned as positive.
            # If `targets` only contains 0s and 1s based on assignment:
            num_positive_anchors = (targets.sum(dim=1) > 0).sum() # Number of anchors assigned to any GT object
            if num_positive_anchors > 0:
                return loss.sum() / num_positive_anchors
            else:
                return loss.sum() * 0.0 # Or just loss.mean() if no positives, or handle as 0 loss.
                                        # Returning 0 prevents NaN if no positives.
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss

class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0/9.0, reduction='mean'): # RetinaNet uses beta = 1.0/9.0
        super().__init__()
        self.beta = beta
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): Predicted regression deltas [N_positive_anchors, 4].
            targets (torch.Tensor): Ground truth regression deltas [N_positive_anchors, 4].
        Returns:
            torch.Tensor: The calculated Smooth L1 Loss.
        """
        diff = torch.abs(inputs - targets)
        loss = torch.where(
            diff < self.beta,
            0.5 * diff**2 / self.beta, # Loss for |x| < beta
            diff - 0.5 * self.beta     # Loss for |x| >= beta
        )
        
        if self.reduction == 'mean':
            # Normalize by the number of positive anchors (same as Focal Loss)
            # The input `inputs` and `targets` should already be only for positive anchors.
            return loss.sum() / inputs.size(0) if inputs.size(0) > 0 else torch.tensor(0.0, device=inputs.device)
        elif self.reduction == 'sum':
            return loss.sum()
        else: # 'none'
            return loss


class DetectionLoss(nn.Module):
    def __init__(self, num_classes, all_anchors_list, 
                 focal_loss_alpha=0.25, focal_loss_gamma=2.0, 
                 smooth_l1_beta=1.0/9.0, 
                 pos_iou_thresh=config.IOU_THRESHOLD_POSITIVE, 
                 neg_iou_thresh=config.IOU_THRESHOLD_NEGATIVE):
        super().__init__()
        self.num_classes = num_classes
        self.all_anchors_list_device = [anchors.to(config.DEVICE) for anchors in all_anchors_list]
        # Concatenate all anchors for easier processing during target assignment
        self.all_anchors_cat = torch.cat(self.all_anchors_list_device, dim=0) # [total_anchors, 4]

        self.focal_loss = FocalLoss(alpha=focal_loss_alpha, gamma=focal_loss_gamma, reduction='mean')
        self.smooth_l1_loss = SmoothL1Loss(beta=smooth_l1_beta, reduction='mean')
        
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh

    def _encode_targets(self, gt_boxes, gt_labels, anchors):
        """
        Encodes ground truth boxes and labels into targets for anchors.
        Matches anchors to ground truth boxes based on IoU.

        Args:
            gt_boxes (torch.Tensor): Ground truth boxes for a single image [num_gt, 4] (xmin, ymin, xmax, ymax).
            gt_labels (torch.Tensor): Ground truth labels for a single image [num_gt] (class indices, 0 to num_classes-1).
            anchors (torch.Tensor): All anchors for the image [total_anchors, 4].

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - cls_targets (torch.Tensor): [total_anchors, num_classes] - one-hot encoded class targets for anchors.
                                              All zeros for negative/ignored anchors.
                - reg_targets (torch.Tensor): [total_anchors, 4] - encoded regression targets (dx,dy,dw,dh) for positive anchors.
                                              Zeros for negative/ignored anchors.
                - assigned_anchor_indices (torch.Tensor): Indices of anchors that were assigned as positive.
        """
        total_anchors = anchors.size(0)
        cls_targets = torch.zeros((total_anchors, self.num_classes), device=anchors.device)
        reg_targets = torch.zeros((total_anchors, 4), device=anchors.device)
        
        if gt_boxes.numel() == 0: # No ground truth objects in this image
            # All anchors are background. Focal loss handles this.
            # For RetinaNet, these would get a cls_target of all zeros.
            return cls_targets, reg_targets, torch.tensor([], dtype=torch.long, device=anchors.device)

        # Calculate IoU between all anchors and all GT boxes
        iou_matrix = box_iou(anchors, gt_boxes) # [total_anchors, num_gt]

        # For each anchor, find the GT box with the highest IoU
        max_iou_per_anchor, best_gt_idx_per_anchor = iou_matrix.max(dim=1) # [total_anchors]

        # Assign anchors to GTs:
        # 1. Anchors with IoU > pos_iou_thresh with *any* GT are positive for that GT.
        positive_mask = max_iou_per_anchor >= self.pos_iou_thresh
        assigned_gt_labels = gt_labels[best_gt_idx_per_anchor[positive_mask]] # Labels for positive anchors
        
        # Set classification targets: one-hot encode the assigned class label
        if positive_mask.sum() > 0:
            cls_targets[positive_mask, assigned_gt_labels] = 1.0

        # 2. Anchors with IoU < neg_iou_thresh with *all* GTs are negative.
        #    These already have cls_targets as all zeros.
        # 3. Anchors between neg_iou_thresh and pos_iou_thresh are ignored (cls_targets remain zero).
        
        # For regression targets: only for positive anchors
        if positive_mask.sum() > 0:
            positive_anchors = anchors[positive_mask]
            assigned_gt_boxes = gt_boxes[best_gt_idx_per_anchor[positive_mask]] # GT boxes for positive anchors

            # Encode regression targets (dx, dy, dw, dh)
            # (xa, ya, wa, ha) for anchors
            xa = (positive_anchors[:, 0] + positive_anchors[:, 2]) / 2.0
            ya = (positive_anchors[:, 1] + positive_anchors[:, 3]) / 2.0
            wa = positive_anchors[:, 2] - positive_anchors[:, 0]
            ha = positive_anchors[:, 3] - positive_anchors[:, 1]

            # (xg, yg, wg, hg) for ground truth boxes
            xg = (assigned_gt_boxes[:, 0] + assigned_gt_boxes[:, 2]) / 2.0
            yg = (assigned_gt_boxes[:, 1] + assigned_gt_boxes[:, 3]) / 2.0
            wg = assigned_gt_boxes[:, 2] - assigned_gt_boxes[:, 0]
            hg = assigned_gt_boxes[:, 3] - assigned_gt_boxes[:, 1]

            # RetinaNet regression targets (can vary based on implementation)
            # Typically: tx = (xg - xa) / wa, ty = (yg - ya) / ha
            #            tw = log(wg / wa), th = log(hg / ha)
            # Variances can also be applied here if the model predicts deltas divided by variance
            eps = 1e-7 # To prevent log(0) or division by zero
            tx = (xg - xa) / (wa + eps)
            ty = (yg - ya) / (ha + eps)
            tw = torch.log(wg / (wa + eps) + eps) # Add eps inside log for stability if wg/wa is very small
            th = torch.log(hg / (ha + eps) + eps)
            
            reg_targets[positive_mask] = torch.stack([tx, ty, tw, th], dim=1)
            
        assigned_anchor_indices = torch.where(positive_mask)[0]
        return cls_targets, reg_targets, assigned_anchor_indices


    def forward(self, cls_preds_list, reg_preds_list, batched_gt_boxes, batched_gt_labels):
        """
        Args:
            cls_preds_list (List[torch.Tensor]): List of classification logits from each FPN level.
                                                 Each tensor [B, H_level*W_level*A, num_classes].
            reg_preds_list (List[torch.Tensor]): List of regression deltas from each FPN level.
                                                 Each tensor [B, H_level*W_level*A, 4].
            batched_gt_boxes (List[torch.Tensor]): List of GT boxes for each image in the batch.
                                                  Each element is [num_gt_for_image, 4].
            batched_gt_labels (List[torch.Tensor]): List of GT labels for each image in the batch.
                                                   Each element is [num_gt_for_image].
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: classification_loss, regression_loss
        """
        # Concatenate predictions from all FPN levels
        # This makes target assignment across all anchors simpler
        # cls_preds: [B, total_anchors, num_classes]
        # reg_preds: [B, total_anchors, 4]
        cls_preds_cat = torch.cat([preds.view(preds.size(0), -1, self.num_classes) for preds in cls_preds_list], dim=1)
        reg_preds_cat = torch.cat([preds.view(preds.size(0), -1, 4) for preds in reg_preds_list], dim=1)

        batch_size = cls_preds_cat.size(0)
        
        batch_cls_targets = []
        batch_reg_targets = []
        batch_positive_anchor_indices = [] # List of tensors, one per image

        for i in range(batch_size):
            gt_boxes_img = batched_gt_boxes[i].to(config.DEVICE)
            gt_labels_img = batched_gt_labels[i].to(config.DEVICE)
            
            # self.all_anchors_cat is already on device
            cls_target_img, reg_target_img, pos_indices_img = self._encode_targets(
                gt_boxes_img, gt_labels_img, self.all_anchors_cat
            )
            batch_cls_targets.append(cls_target_img)
            batch_reg_targets.append(reg_target_img)
            batch_positive_anchor_indices.append(pos_indices_img)

        # Stack targets for the batch
        cls_targets_tensor = torch.stack(batch_cls_targets, dim=0) # [B, total_anchors, num_classes]
        reg_targets_tensor = torch.stack(batch_reg_targets, dim=0) # [B, total_anchors, 4]

        # --- Classification Loss ---
        # Reshape for Focal Loss: inputs [N, C], targets [N, C] where N is B * total_anchors
        # We only care about anchors that are either positive or negative (not ignored based on IoU)
        # Focal loss `targets` should be 0 for negative, 1 for the correct class of positive.
        # Our `cls_targets_tensor` is already in this format.
        
        # Create a mask for anchors that are either positive or "true negatives"
        # True negatives are those with max_iou < neg_iou_thresh
        # Positives are those with max_iou >= pos_iou_thresh
        # We want to compute loss over all these.
        # The current cls_targets_tensor is 1 for positive class, 0 otherwise.
        # This is suitable for binary_cross_entropy_with_logits used in FocalLoss.
        
        # Flatten predictions and targets for loss calculation
        cls_preds_flat = cls_preds_cat.reshape(-1, self.num_classes)
        cls_targets_flat = cls_targets_tensor.reshape(-1, self.num_classes)
        
        # For Focal Loss normalization by num positive anchors, we need the sum of positives across the batch.
        # Let FocalLoss handle its own normalization based on the targets passed to it.
        classification_loss = self.focal_loss(cls_preds_flat, cls_targets_flat)

        # --- Regression Loss ---
        # Only for positive anchors
        # Gather predictions and targets for positive anchors across the batch
        positive_reg_preds = []
        positive_reg_targets = []
        total_positive_anchors_in_batch = 0

        for i in range(batch_size):
            pos_indices_img = batch_positive_anchor_indices[i] # Indices for THIS image's anchors
            if pos_indices_img.numel() > 0:
                positive_reg_preds.append(reg_preds_cat[i, pos_indices_img, :])
                positive_reg_targets.append(reg_targets_tensor[i, pos_indices_img, :])
                total_positive_anchors_in_batch += pos_indices_img.numel()
        
        if total_positive_anchors_in_batch > 0:
            reg_preds_for_loss = torch.cat(positive_reg_preds, dim=0)   # [total_pos_in_batch, 4]
            reg_targets_for_loss = torch.cat(positive_reg_targets, dim=0) # [total_pos_in_batch, 4]
            regression_loss = self.smooth_l1_loss(reg_preds_for_loss, reg_targets_for_loss)
        else:
            regression_loss = torch.tensor(0.0, device=cls_preds_cat.device) # No positive anchors, so no reg loss

        return classification_loss, regression_loss


if __name__ == '__main__':
    from src.utils import generate_all_anchors
    print("--- Testing Loss Functions ---")
    
    # --- Test Focal Loss ---
    print("\nTesting Focal Loss...")
    focal_loss_fn = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean')
    # Example: 2 anchors, 3 classes
    # Anchor 0: positive for class 1
    # Anchor 1: negative
    cls_logits_example = torch.tensor([[-1.0, 2.0, -0.5],  # Anchor 0, high score for class 1
                                    [-2.0, -1.5, -1.0]], dtype=torch.float32) # Anchor 1, all low scores
    cls_targets_example = torch.tensor([[0.0, 1.0, 0.0],   # Anchor 0, target class 1
                                     [0.0, 0.0, 0.0]], dtype=torch.float32) # Anchor 1, background
    
    loss_focal = focal_loss_fn(cls_logits_example, cls_targets_example)
    print(f"Focal Loss example: {loss_focal.item()}")
    # Expected: Focal loss should be computed, normalized by 1 (num positive anchors)

    # --- Test Smooth L1 Loss ---
    print("\nTesting Smooth L1 Loss...")
    smooth_l1_fn = SmoothL1Loss(beta=1.0/9.0, reduction='mean')
    # Example: 2 positive anchors
    reg_preds_example = torch.tensor([[0.1, 0.2, 0.05, -0.1],
                                   [1.0, 1.5, 2.0, 1.2]], dtype=torch.float32)
    reg_targets_example = torch.tensor([[0.12, 0.18, 0.0, -0.08],
                                     [0.8, 1.6, 2.2, 1.0]], dtype=torch.float32)
    loss_smooth_l1 = smooth_l1_fn(reg_preds_example, reg_targets_example)
    print(f"Smooth L1 Loss example: {loss_smooth_l1.item()}")


    # --- Test Full DetectionLoss ---
    print("\nTesting DetectionLoss (target assignment and combined loss)...")
    num_test_classes = 3
    config.NUM_CLASSES = num_test_classes # Temporarily override for test
    
    # Generate dummy anchors (use config, but can be simplified for test)
    _image_size = 64 # smaller image for faster anchor gen in test
    _fpn_strides = [8, 16]
    _anchor_sizes = [[16, 32], [32, 64]]
    _aspect_ratios = [1.0]
    _num_anchors_per_loc = len(_anchor_sizes[0]) * len(_aspect_ratios)

    test_all_anchors_list = generate_all_anchors(_image_size, _fpn_strides, _anchor_sizes, _aspect_ratios)
    test_all_anchors_cat = torch.cat(test_all_anchors_list, dim=0).to(config.DEVICE)
    total_test_anchors = test_all_anchors_cat.shape[0]
    print(f"Total test anchors: {total_test_anchors}")


    detection_loss_fn = DetectionLoss(
        num_classes=num_test_classes,
        all_anchors_list=test_all_anchors_list # Pass the list, it will be moved to device and cat inside
    )

    # Dummy model predictions (batch size 1 for simplicity)
    # cls_preds_list: [B, H*W*A, C] for each FPN level
    # reg_preds_list: [B, H*W*A, 4] for each FPN level
    batch_s = 1
    cls_preds_list_test = []
    reg_preds_list_test = []
    for anchors_level in test_all_anchors_list:
        num_anchors_level = anchors_level.shape[0]
        cls_preds_list_test.append(torch.randn(batch_s, num_anchors_level, num_test_classes).to(config.DEVICE))
        reg_preds_list_test.append(torch.randn(batch_s, num_anchors_level, 4).to(config.DEVICE))

    # Dummy ground truth for one image
    # gt_boxes_img1 should overlap with some anchors to make them positive
    # Pick an anchor and make a GT box that overlaps it well.
    # e.g., first anchor from the first level: test_all_anchors_list[0][0,:]
    anchor_to_match = test_all_anchors_list[0][50, :].unsqueeze(0) # Pick an anchor e.g. 50th
    print(f"Anchor to match: {anchor_to_match}")
    
    gt_box1_img1 = anchor_to_match.clone() # Perfect match initially
    # Slightly modify for realism
    gt_box1_img1[0,0] += 2 
    gt_box1_img1[0,1] += 2
    gt_box1_img1[0,2] -= 1
    gt_box1_img1[0,3] -= 1

    gt_box2_img1 = torch.tensor([[10.0, 10.0, 30.0, 30.0]], device=config.DEVICE) # Another box
    
    # Check IoU of gt_box1_img1 with the anchor it was derived from
    iou_check = box_iou(gt_box1_img1, anchor_to_match)
    print(f"IoU of gt_box1 with its original anchor: {iou_check.item()}") # Should be high


    batched_gt_boxes_test = [torch.cat([gt_box1_img1, gt_box2_img1], dim=0)]
    batched_gt_labels_test = [torch.tensor([0, 1], dtype=torch.long, device=config.DEVICE)] # Class 0 and Class 1


    # Test _encode_targets (internal method, but useful for debugging)
    print("\nTesting _encode_targets...")
    cls_targets_enc, reg_targets_enc, pos_indices_enc = detection_loss_fn._encode_targets(
        batched_gt_boxes_test[0], batched_gt_labels_test[0], test_all_anchors_cat
    )
    print(f"Encoded cls_targets shape: {cls_targets_enc.shape}") # [total_test_anchors, num_test_classes]
    print(f"Encoded reg_targets shape: {reg_targets_enc.shape}") # [total_test_anchors, 4]
    print(f"Number of positive anchors found: {pos_indices_enc.numel()}")
    assert pos_indices_enc.numel() > 0, "No positive anchors found in _encode_targets test. Check IoU thresholds or GT box placement."
    
    # Verify one-hot encoding and regression targets for a positive anchor
    if pos_indices_enc.numel() > 0:
        example_pos_idx = pos_indices_enc[0]
        print(f"  Example positive anchor index: {example_pos_idx.item()}")
        print(f"  Cls target for this anchor: {cls_targets_enc[example_pos_idx, :]}")
        print(f"  Reg target for this anchor: {reg_targets_enc[example_pos_idx, :]}")


    # Test forward pass of DetectionLoss
    print("\nTesting DetectionLoss forward pass...")
    cls_loss, reg_loss = detection_loss_fn(cls_preds_list_test, reg_preds_list_test, 
                                           batched_gt_boxes_test, batched_gt_labels_test)

    print(f"Calculated Classification Loss: {cls_loss.item()}")
    print(f"Calculated Regression Loss: {reg_loss.item()}")
    assert cls_loss.item() >= 0
    assert reg_loss.item() >= 0

    # Restore original NUM_CLASSES from config if it was changed
    config.NUM_CLASSES = len(config.get_class_names())