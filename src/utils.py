# src/utils.py
import torch
import numpy as np
from src import config # To access ANCHOR_SIZES, ASPECT_RATIOS, ANCHOR_STRIDES, IMAGE_SIZE
import torchvision # For NMS

# ------------- Existing Anchor Generation and IoU Functions -------------

def generate_anchors_for_level(feature_map_size, base_anchor_sizes, aspect_ratios, stride):
    """
    Generates anchor boxes for a single FPN level.
    (Code as you provided - no changes here)
    """
    H_level, W_level = feature_map_size
    num_anchors_per_loc = len(base_anchor_sizes) * len(aspect_ratios)
    grid_y, grid_x = torch.meshgrid(torch.arange(H_level), torch.arange(W_level), indexing='ij')
    center_x = (grid_x.float() + 0.5) * stride
    center_y = (grid_y.float() + 0.5) * stride
    centers = torch.stack((center_x, center_y, center_x, center_y), dim=-1)
    centers = centers.view(-1, 1, 4).repeat(1, num_anchors_per_loc, 1)
    anchor_dims = []
    for size in base_anchor_sizes:
        for ratio in aspect_ratios:
            anchor_h = size / np.sqrt(ratio)
            anchor_w = size * np.sqrt(ratio)
            anchor_dims.append([anchor_w, anchor_h])
    anchor_dims = torch.tensor(anchor_dims, dtype=torch.float32)
    wh_for_offsets = anchor_dims.unsqueeze(0)
    half_w = wh_for_offsets[:, :, 0] / 2.0
    half_h = wh_for_offsets[:, :, 1] / 2.0
    offsets = torch.stack([-half_w, -half_h, half_w, half_h], dim=-1).squeeze(0)
    offsets = offsets.unsqueeze(0).repeat(H_level * W_level, 1, 1)
    anchors_level = centers + offsets
    anchors_level = anchors_level.view(-1, 4)
    return anchors_level

def generate_all_anchors(image_size, fpn_strides, fpn_anchor_sizes, aspect_ratios):
    """
    Generates anchor boxes for all FPN levels.
    (Code as you provided - no changes here)
    """
    all_anchors = []
    for i, stride in enumerate(fpn_strides):
        feature_map_h = image_size // stride
        feature_map_w = image_size // stride
        level_anchor_sizes = fpn_anchor_sizes[i]
        anchors_level = generate_anchors_for_level(
            feature_map_size=(feature_map_h, feature_map_w),
            base_anchor_sizes=level_anchor_sizes,
            aspect_ratios=aspect_ratios,
            stride=stride
        )
        all_anchors.append(anchors_level)
    return all_anchors

def box_iou(boxes1, boxes2):
    """
    Calculate Intersection over Union (IoU) between two sets of boxes.
    (Code as you provided - no changes here)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    iou = inter / (union + 1e-6)
    return iou

# ------------- New Post-processing Functions -------------

def decode_boxes(anchor_boxes_all_levels, reg_preds_all_levels):
    """
    Decodes regression predictions to actual bounding box coordinates.
    Assumes reg_preds are (tx, ty, tw, th) relative to anchors.

    Args:
        anchor_boxes_all_levels (torch.Tensor): All anchor boxes concatenated, 
                                                shape [total_anchors, 4] (xmin, ymin, xmax, ymax).
        reg_preds_all_levels (torch.Tensor): Regression predictions from the model, concatenated,
                                             shape [batch_size, total_anchors, 4] or [total_anchors, 4].
                                             Order: tx, ty, tw, th.

    Returns:
        torch.Tensor: Decoded bounding boxes, shape same as reg_preds_all_levels (xmin, ymin, xmax, ymax).
    """
    if reg_preds_all_levels.ndim == 3: # Batched predictions
        anchor_boxes_all_levels = anchor_boxes_all_levels.unsqueeze(0).expand_as(reg_preds_all_levels)

    xa = (anchor_boxes_all_levels[..., 0] + anchor_boxes_all_levels[..., 2]) / 2.0
    ya = (anchor_boxes_all_levels[..., 1] + anchor_boxes_all_levels[..., 3]) / 2.0
    wa = anchor_boxes_all_levels[..., 2] - anchor_boxes_all_levels[..., 0]
    ha = anchor_boxes_all_levels[..., 3] - anchor_boxes_all_levels[..., 1]

    tx = reg_preds_all_levels[..., 0]
    ty = reg_preds_all_levels[..., 1]
    tw = reg_preds_all_levels[..., 2]
    th = reg_preds_all_levels[..., 3]

    eps = 1e-7 
    pred_center_x = tx * wa + xa
    pred_center_y = ty * ha + ya
    # Ensure wa and ha are positive before taking exp to avoid issues if anchors are invalid
    # This should ideally be handled by ensuring anchors are valid, but as a safeguard:
    pred_w = torch.exp(tw) * torch.clamp(wa, min=eps) 
    pred_h = torch.exp(th) * torch.clamp(ha, min=eps)
    
    pred_xmin = pred_center_x - pred_w / 2.0
    pred_ymin = pred_center_y - pred_h / 2.0
    pred_xmax = pred_center_x + pred_w / 2.0
    pred_ymax = pred_center_y + pred_h / 2.0

    decoded_boxes = torch.stack([pred_xmin, pred_ymin, pred_xmax, pred_ymax], dim=-1)
    return decoded_boxes

def postprocess_detections_per_image(
    cls_scores_img, 
    reg_preds_img,  
    anchors_img,    
    image_shape,    
    conf_threshold=config.CONFIDENCE_THRESHOLD, 
    nms_iou_thresh=config.NMS_IOU_THRESHOLD
    ):
    """
    Post-processes raw model outputs for a single image to get final detections.
    1. Decodes boxes.
    2. Filters by confidence score.
    3. Applies NMS per class.
    
    Args:
        cls_scores_img (torch.Tensor): Sigmoid scores for all anchors [N_anchors, N_classes].
        reg_preds_img (torch.Tensor): Regression deltas for all anchors [N_anchors, 4].
        anchors_img (torch.Tensor): Anchor boxes for this image [N_anchors, 4].
        image_shape (tuple): (height, width) of the image the boxes should be clipped to.
        conf_threshold (float): Confidence threshold for filtering.
        nms_iou_thresh (float): IoU threshold for NMS.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            boxes (Tensor[N_dets, 4]), scores (Tensor[N_dets]), labels (Tensor[N_dets])
    """
    device = cls_scores_img.device
    num_classes = cls_scores_img.shape[1]

    decoded_boxes = decode_boxes(anchors_img, reg_preds_img) 

    all_boxes_list = [] # Use lists to append, then cat
    all_scores_list = []
    all_labels_list = []

    for class_idx in range(num_classes):
        class_specific_scores = cls_scores_img[:, class_idx] 

        keep_idxs_conf = class_specific_scores >= conf_threshold
        
        if not torch.any(keep_idxs_conf): # More efficient check
            continue

        filtered_scores = class_specific_scores[keep_idxs_conf]
        filtered_boxes = decoded_boxes[keep_idxs_conf] 

        # Clip boxes to image dimensions
        # image_shape is (H, W)
        # boxes are (xmin, ymin, xmax, ymax)
        # Ensure filtered_boxes is not empty before clamping
        if filtered_boxes.numel() > 0:
            filtered_boxes[:, 0].clamp_(min=0, max=image_shape[1] -1) # xmin < W (use W-1 for 0-indexed max)
            filtered_boxes[:, 1].clamp_(min=0, max=image_shape[0] -1) # ymin < H
            filtered_boxes[:, 2].clamp_(min=0, max=image_shape[1] -1) # xmax < W
            filtered_boxes[:, 3].clamp_(min=0, max=image_shape[0] -1) # ymax < H

            # Further ensure xmin < xmax and ymin < ymax after clamping, remove if not
            valid_boxes_mask = (filtered_boxes[:, 0] < filtered_boxes[:, 2]) & (filtered_boxes[:, 1] < filtered_boxes[:, 3])
            if not torch.any(valid_boxes_mask):
                continue
            filtered_boxes = filtered_boxes[valid_boxes_mask]
            filtered_scores = filtered_scores[valid_boxes_mask]


        if filtered_boxes.numel() == 0: # Check again after potential filtering by valid_boxes_mask
            continue
            
        keep_idxs_nms = torchvision.ops.nms(filtered_boxes, filtered_scores, nms_iou_thresh)

        final_boxes_class = filtered_boxes[keep_idxs_nms]
        final_scores_class = filtered_scores[keep_idxs_nms]
        final_labels_class = torch.full_like(final_scores_class, fill_value=class_idx, dtype=torch.long)

        all_boxes_list.append(final_boxes_class)
        all_scores_list.append(final_scores_class)
        all_labels_list.append(final_labels_class)

    if not all_boxes_list: 
        return (
            torch.empty((0, 4), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.float32, device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    final_boxes_all_classes = torch.cat(all_boxes_list, dim=0)
    final_scores_all_classes = torch.cat(all_scores_list, dim=0)
    final_labels_all_classes = torch.cat(all_labels_list, dim=0)
    
    return final_boxes_all_classes, final_scores_all_classes, final_labels_all_classes


# ------------- Main test block -------------
if __name__ == '__main__':
    # --- Test anchor generation (original test) ---
    print("--- Testing Anchor Generation ---")
    image_size_cfg = config.IMAGE_SIZE # Renamed to avoid conflict with function arg
    fpn_strides_cfg = config.ANCHOR_STRIDES
    fpn_anchor_sizes_cfg = config.ANCHOR_SIZES
    aspect_ratios_cfg = config.ASPECT_RATIOS
    num_anchors_per_loc_cfg = config.NUM_ANCHORS_PER_LEVEL

    all_anchors_list_test = generate_all_anchors(image_size_cfg, fpn_strides_cfg, fpn_anchor_sizes_cfg, aspect_ratios_cfg)
    level_names = ['P3', 'P4', 'P5', 'P6']
    total_anchors_generated = 0
    for i, anchors_level in enumerate(all_anchors_list_test):
        level_name = level_names[i]
        stride = fpn_strides_cfg[i]
        H_level = image_size_cfg // stride
        W_level = image_size_cfg // stride
        expected_num_anchors = H_level * W_level * num_anchors_per_loc_cfg
        print(f"Level {level_name} (Stride {stride}, Feature Map {H_level}x{W_level}): Generated anchors shape: {anchors_level.shape}")
        assert anchors_level.shape[0] == expected_num_anchors
        assert anchors_level.shape[1] == 4
        total_anchors_generated += anchors_level.shape[0]
        assert torch.all(anchors_level[:, 0] < anchors_level[:, 2]) 
        assert torch.all(anchors_level[:, 1] < anchors_level[:, 3]) 
    print(f"\nTotal anchors generated across all levels: {total_anchors_generated}")
    concatenated_anchors_test = torch.cat(all_anchors_list_test, dim=0)
    print(f"Shape of all concatenated anchors: {concatenated_anchors_test.shape}")

    # --- Test IoU calculation (original test) ---
    print("\n--- Testing IoU Calculation ---")
    boxes1_np = np.array([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=np.float32)
    boxes2_np = np.array([[0, 0, 10, 10], [8, 8, 12, 12], [20, 20, 30, 30]], dtype=np.float32)
    boxes1 = torch.from_numpy(boxes1_np)
    boxes2 = torch.from_numpy(boxes2_np)
    iou_matrix = box_iou(boxes1, boxes2)
    print("Boxes1:\n", boxes1); print("Boxes2:\n", boxes2); print("IoU Matrix:\n", iou_matrix)
    expected_iou_00 = 1.0
    expected_iou_01 = 4.0 / 112.0 
    expected_iou_11 = 16.0 / 100.0
    assert torch.isclose(iou_matrix[0, 0], torch.tensor(expected_iou_00))
    assert torch.isclose(iou_matrix[0, 1], torch.tensor(expected_iou_01)) 
    assert torch.isclose(iou_matrix[1, 1], torch.tensor(expected_iou_11))
    assert torch.isclose(iou_matrix[0,2], torch.tensor(0.0))
    print("IoU test passed.")


    # --- Test decode_boxes ---
    print("\n--- Testing decode_boxes ---")
    test_anchors = torch.tensor([[50., 50., 150., 150.], [100., 100., 200., 200.]], dtype=torch.float32) 
    test_reg_preds = torch.tensor([
        [0.0, 0.0, 0.0, 0.0], 
        [0.1, -0.1, torch.log(torch.tensor(1.2)), torch.log(torch.tensor(0.8))]
    ], dtype=torch.float32)
    decoded = decode_boxes(test_anchors, test_reg_preds)
    print("Test Anchors:\n", test_anchors)
    print("Test Reg Preds (tx,ty,tw,th):\n", test_reg_preds)
    print("Decoded Boxes (xmin,ymin,xmax,ymax):\n", decoded)
    expected_decoded_0 = test_anchors[0]
    expected_decoded_1 = torch.tensor([100., 100., 220., 180.])
    assert torch.allclose(decoded[0], expected_decoded_0, atol=1e-5)
    assert torch.allclose(decoded[1], expected_decoded_1, atol=1e-5)
    print("decode_boxes test passed.")

    # --- Test postprocess_detections_per_image ---
    print("\n--- Testing postprocess_detections_per_image ---")
    N_anchors_test = 100
    # Use NUM_CLASSES from config for consistency if available, else default
    N_classes_test = config.NUM_CLASSES if hasattr(config, 'NUM_CLASSES') and config.NUM_CLASSES is not None else 3 
    
    img_h_test, img_w_test = 640, 640
    
    dummy_cls_scores = torch.sigmoid(torch.randn(N_anchors_test, N_classes_test)) 
    dummy_reg_preds = torch.randn(N_anchors_test, 4) * 0.1 
    dummy_anchors = torch.rand(N_anchors_test, 4) 
    # Scale dummy anchors to image size and ensure xmin < xmax, ymin < ymax
    dummy_anchors[:, 0] *= img_w_test * 0.8 # xmin
    dummy_anchors[:, 1] *= img_h_test * 0.8 # ymin
    dummy_anchors[:, 2] = dummy_anchors[:, 0] + torch.rand(N_anchors_test) * (img_w_test * 0.2) + 10 # xmax > xmin
    dummy_anchors[:, 3] = dummy_anchors[:, 1] + torch.rand(N_anchors_test) * (img_h_test * 0.2) + 10 # ymax > ymin
    dummy_anchors.clamp_(min=0)

    dummy_cls_scores[0, 0] = 0.9 
    dummy_cls_scores[1, 0] = 0.8 
    dummy_cls_scores[2, 1] = 0.95 
    
    dummy_anchors[1, :] = dummy_anchors[0, :] + torch.tensor([5.0, 5.0, 5.0, 5.0]) # Ensure some overlap for NMS

    final_boxes, final_scores, final_labels = postprocess_detections_per_image(
        dummy_cls_scores, dummy_reg_preds, dummy_anchors, (img_h_test, img_w_test),
        conf_threshold=0.7, nms_iou_thresh=0.5
    )
    print(f"Found {len(final_scores)} detections after postprocessing.")
    if len(final_scores) > 0 :
        print("Final Boxes:\n", final_boxes)
        print("Final Scores:\n", final_scores)
        print("Final Labels:\n", final_labels)
    
    # Expected outcome is tricky due to randomness, but check for basic validity
    # We expect at most 2 unique detections (anchor 0/1 for class 0 might become 1 after NMS, anchor 2 for class 1)
    assert len(final_scores) <= 2 
    if len(final_scores) > 0:
        assert final_boxes.shape[1] == 4
        assert final_scores.ndim == 1
        assert final_labels.ndim == 1
        assert final_boxes.shape[0] == final_scores.shape[0] == final_labels.shape[0]
    print("postprocess_detections_per_image test ran.")