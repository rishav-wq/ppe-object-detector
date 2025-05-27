

# src/evaluate.py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from src import config
from src.dataset import PPEDataset, collate_fn
from src.model import PPEObjectDetector
from src.utils import generate_all_anchors, postprocess_detections_per_image
from torchmetrics.detection.mean_ap import MeanAveragePrecision

@torch.no_grad()
def evaluate_model(model, data_loader, all_anchors_cat, device, image_size_from_config):
    model.eval()
    
    # Initialize torchmetrics mAP calculator
    # You might need to adjust 'iou_thresholds' if your evaluation standard is different
    # For COCO, it's usually a range [0.5:0.05:0.95]
    # For Pascal VOC, it's often a single IoU threshold like 0.5
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True) # Get per-class AP too
                                # box_format='xyxy' is default
    
    progress_bar = tqdm(data_loader, desc="Evaluating", unit="batch")

    for batch_idx, batch_data in enumerate(progress_bar):
        if batch_data is None or batch_data[0] is None:
            print(f"Skipping problematic evaluation batch {batch_idx}")
            continue
            
        images, batched_gt_boxes, batched_gt_labels = batch_data
        images = images.to(device)

        # Raw model predictions
        # cls_preds_list: list of [B, H_level*W_level*A, NUM_CLASSES]
        # reg_preds_list: list of [B, H_level*W_level*A, 4]
        cls_preds_list_raw, reg_preds_list_raw = model(images)

        # Concatenate predictions from all FPN levels for easier processing
        # cls_preds_batch: [B, total_anchors, num_classes]
        # reg_preds_batch: [B, total_anchors, 4]
        cls_preds_batch = torch.cat([preds.view(preds.size(0), -1, config.NUM_CLASSES) for preds in cls_preds_list_raw], dim=1)
        reg_preds_batch = torch.cat([preds.view(preds.size(0), -1, 4) for preds in reg_preds_list_raw], dim=1)

        # Apply sigmoid to classification scores
        cls_scores_batch = torch.sigmoid(cls_preds_batch) # [B, total_anchors, num_classes]

        # --- Process predictions and targets for each image in the batch ---
        preds_for_metric = []
        targets_for_metric = []

        batch_size_current = images.shape[0]
        for i in range(batch_size_current):
            img_cls_scores = cls_scores_batch[i] # [total_anchors, num_classes]
            img_reg_preds = reg_preds_batch[i]   # [total_anchors, 4]
            
            # Post-process to get final detections for this image
            # image_shape needs to be (H, W) of the input to the model (after resize)
            # This is config.IMAGE_SIZE
            final_boxes, final_scores, final_labels = postprocess_detections_per_image(
                img_cls_scores,
                img_reg_preds,
                all_anchors_cat.to(device), # Ensure anchors are on the same device
                image_shape=(image_size_from_config, image_size_from_config),
                conf_threshold=config.CONFIDENCE_THRESHOLD,
                nms_iou_thresh=config.NMS_IOU_THRESHOLD
            )
            
            preds_for_metric.append({
                "boxes": final_boxes.cpu(),   # torchmetrics expects CPU tensors
                "scores": final_scores.cpu(),
                "labels": final_labels.cpu(),
            })

            # Format ground truth for torchmetrics
            gt_boxes_img = batched_gt_boxes[i].cpu() # Ensure CPU
            gt_labels_img = batched_gt_labels[i].cpu() # Ensure CPU
            targets_for_metric.append({
                "boxes": gt_boxes_img,
                "labels": gt_labels_img,
            })
        
        # Update metric with predictions and targets for the current batch
        if preds_for_metric and targets_for_metric: # Only if there are valid items
             metric.update(preds_for_metric, targets_for_metric)

    # Compute final mAP results
    try:
        map_results = metric.compute()
    except Exception as e:
        print(f"Error computing mAP: {e}")
        print("This might happen if no predictions or no ground truths were processed.")
        map_results = None

    return map_results


def main_evaluate(checkpoint_path):
    device = config.DEVICE
    print(f"Using device: {device}")

    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Infer class names and image size from checkpoint if available, else use config
    # This makes evaluation more robust if config changes later
    loaded_class_names = checkpoint.get('class_names', config.CLASS_NAMES)
    loaded_image_size = checkpoint.get('image_size', config.IMAGE_SIZE)
    num_classes = len(loaded_class_names)

    print(f"Model trained with {num_classes} classes: {loaded_class_names}")
    print(f"Model trained with image size: {loaded_image_size}")


    # Dataset and DataLoader (use validation set for evaluation)
    print("Loading validation dataset...")
    val_dataset = PPEDataset(
        data_dir=config.VAL_DATA_DIR, # Ensure this path is correct in config.py
        class_names=loaded_class_names, # Use class names from checkpoint
        image_size=loaded_image_size,   # Use image size from checkpoint
        is_train=False 
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE, # Can adjust; usually doesn't need to be too small for eval
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False
    )
    print(f"Validation samples: {len(val_dataset)}")

    # Model
    print("Initializing model...")
    model = PPEObjectDetector(
        backbone_name=config.BACKBONE, # Assuming backbone config matches saved model
        fpn_out_channels=config.FPN_OUT_CHANNELS,
        num_classes=num_classes, # Use num_classes from checkpoint/loaded_class_names
        num_anchors_per_level=config.NUM_ANCHORS_PER_LEVEL,
        pretrained_backbone=False # Don't load pretrained weights, we are loading our trained ones
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded successfully.")

    # Generate anchors (must match the configuration used during training)
    # Anchors are based on image_size, strides, etc.
    all_anchors_list = generate_all_anchors(
        image_size=loaded_image_size, # Use image size from checkpoint
        fpn_strides=config.ANCHOR_STRIDES,
        fpn_anchor_sizes=config.ANCHOR_SIZES,
        aspect_ratios=config.ASPECT_RATIOS
    )
    all_anchors_cat = torch.cat(all_anchors_list, dim=0) # [total_anchors, 4]

    # Perform evaluation
    map_results = evaluate_model(model, val_loader, all_anchors_cat, device, loaded_image_size)

    if map_results:
        print("\n--- Evaluation Results ---")
        for k, v in map_results.items():
            if isinstance(v, torch.Tensor) and v.numel() == 1:
                print(f"  {k}: {v.item():.4f}")
            elif isinstance(v, torch.Tensor): # e.g. map_per_class
                print(f"  {k}:")
                for i, class_ap in enumerate(v):
                    if i < len(loaded_class_names):
                        class_name = loaded_class_names[i]
                        print(f"    {class_name}: {class_ap.item():.4f}")
                    else:
                        print(f"    Class Index {i}: {class_ap.item():.4f} (Name not found)")
            else:
                print(f"  {k}: {v}") # For other types like lists of numbers for 'map_small', etc.
    else:
        print("Evaluation did not produce mAP results.")


if __name__ == '__main__':
    # IMPORTANT: Specify the path to your best trained model checkpoint
    # This should be a .pth file from your ../weights/ directory
    # For example: best_checkpoint_path = "../weights/ppe_detector_best_epochX_vallossY.pth"
    
    # Find the "best" checkpoint automatically, or specify manually
    weights_dir = "../weights"
    best_checkpoint_path = None
    best_val_loss = float('inf')

    if os.path.exists(weights_dir):
        for f_name in os.listdir(weights_dir):
            if f_name.startswith("ppe_detector_best_epoch") and f_name.endswith(".pth"):
                try:
                    # Extract validation loss from filename, e.g., ppe_detector_best_epoch10_valloss0.1234.pth
                    loss_str = f_name.split("valloss")[1].replace(".pth", "")
                    val_loss = float(loss_str)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint_path = os.path.join(weights_dir, f_name)
                except Exception:
                    pass # Ignore files that don't match the naming pattern
    
    if best_checkpoint_path:
        print(f"Found best checkpoint: {best_checkpoint_path} with val_loss: {best_val_loss:.4f}")
        main_evaluate(best_checkpoint_path)
    elif os.path.exists(os.path.join(weights_dir, "ppe_detector_last.pth")):
        print("Best checkpoint not found by naming convention, using last checkpoint.")
        main_evaluate(os.path.join(weights_dir, "ppe_detector_last.pth"))
    else:
        print("No suitable checkpoint found in ../weights/. Please train the model first or specify path manually.")