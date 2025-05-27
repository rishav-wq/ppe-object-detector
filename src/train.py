import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))



# src/train.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
import time

from src import config
from src.dataset import PPEDataset, collate_fn # Assuming dataset.py is in src
from src.model import PPEObjectDetector       # Assuming model.py is in src
from src.loss import DetectionLoss            # Assuming loss.py is in src
from src.utils import generate_all_anchors    # Assuming utils.py is in src

# --- For mAP calculation (placeholder, actual mAP is more complex) ---
# from torchmetrics.detection.mean_ap import MeanAveragePrecision # Example if using torchmetrics
# We'll just do loss for validation for now to keep it simpler in this script.
# True mAP would require converting model outputs to detection boxes, NMS, etc.

def train_one_epoch(model, data_loader, loss_fn, optimizer, device, epoch, total_epochs):
    model.train() # Set model to training mode
    
    total_loss = 0.0
    total_cls_loss = 0.0
    total_reg_loss = 0.0
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Training]", unit="batch")

    for batch_idx, batch_data in enumerate(progress_bar):
        if batch_data is None or batch_data[0] is None: # From collate_fn if all items were None
            print(f"Skipping problematic batch {batch_idx}")
            continue
            
        images, batched_gt_boxes, batched_gt_labels = batch_data
        
        images = images.to(device)
        # GT boxes and labels are lists of tensors; they will be moved to device inside loss_fn or _encode_targets

        optimizer.zero_grad()
        
        # Forward pass
        cls_preds_list, reg_preds_list = model(images)
        
        # Calculate loss
        # loss_fn expects gt_boxes and gt_labels to be lists of tensors (one per image in batch)
        # cls_preds_list and reg_preds_list are also lists (one per FPN level)
        classification_loss, regression_loss = loss_fn(
            cls_preds_list, reg_preds_list, 
            batched_gt_boxes, batched_gt_labels
        )
        
        current_loss = classification_loss + regression_loss
        
        # Backward pass and optimize
        if torch.isfinite(current_loss): # Check for NaN/Inf loss
            current_loss.backward()
            optimizer.step()
            
            total_loss += current_loss.item()
            total_cls_loss += classification_loss.item()
            total_reg_loss += regression_loss.item()
        else:
            print(f"Warning: Encountered non-finite loss (NaN or Inf) in batch {batch_idx}. Skipping batch.")
            # Optionally, you might want to log more details or even stop training if this happens frequently.

        # Update progress bar
        progress_bar.set_postfix(
            loss=f"{current_loss.item():.4f}",
            cls_loss=f"{classification_loss.item():.4f}",
            reg_loss=f"{regression_loss.item():.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.1e}"
        )
        
    avg_epoch_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0
    avg_cls_loss = total_cls_loss / len(data_loader) if len(data_loader) > 0 else 0
    avg_reg_loss = total_reg_loss / len(data_loader) if len(data_loader) > 0 else 0
    
    return avg_epoch_loss, avg_cls_loss, avg_reg_loss


@torch.no_grad() # Decorator to disable gradient calculations
def validate_one_epoch(model, data_loader, loss_fn, device, epoch, total_epochs):
    model.eval() # Set model to evaluation mode
    
    total_val_loss = 0.0
    total_val_cls_loss = 0.0
    total_val_reg_loss = 0.0
    
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{total_epochs} [Validation]", unit="batch")

    for batch_idx, batch_data in enumerate(progress_bar):
        if batch_data is None or batch_data[0] is None:
            print(f"Skipping problematic validation batch {batch_idx}")
            continue
            
        images, batched_gt_boxes, batched_gt_labels = batch_data
        
        images = images.to(device)
        
        cls_preds_list, reg_preds_list = model(images)
        
        classification_loss, regression_loss = loss_fn(
            cls_preds_list, reg_preds_list, 
            batched_gt_boxes, batched_gt_labels
        )
        
        current_val_loss = classification_loss + regression_loss

        if torch.isfinite(current_val_loss):
            total_val_loss += current_val_loss.item()
            total_val_cls_loss += classification_loss.item()
            total_val_reg_loss += regression_loss.item()
        else:
            print(f"Warning: Encountered non-finite val_loss in batch {batch_idx}.")


        progress_bar.set_postfix(
            val_loss=f"{current_val_loss.item():.4f}"
        )

    avg_val_epoch_loss = total_val_loss / len(data_loader) if len(data_loader) > 0 else 0
    avg_val_cls_loss = total_val_cls_loss / len(data_loader) if len(data_loader) > 0 else 0
    avg_val_reg_loss = total_val_reg_loss / len(data_loader) if len(data_loader) > 0 else 0

    # Placeholder for mAP calculation
    # mAP_calculator = MeanAveragePrecision(...)
    # for preds, targets in validation_data:
    #    detections = postprocess(preds) # Convert model output to [boxes, scores, labels]
    #    mAP_calculator.update(detections, targets)
    # val_map = mAP_calculator.compute()
    # print(f"Validation mAP: {val_map['map']:.4f}")
    
    return avg_val_epoch_loss, avg_val_cls_loss, avg_val_reg_loss


def main():
    # --- Setup ---
    device = config.DEVICE
    print(f"Using device: {device}")

    # Create output directory for weights
    weights_dir = "../weights" # Relative to src/
    os.makedirs(weights_dir, exist_ok=True)

    # DataLoaders
    print("Loading datasets...")
    train_dataset = PPEDataset(
        data_dir=config.TRAIN_DATA_DIR,
        class_names=config.CLASS_NAMES,
        image_size=config.IMAGE_SIZE,
        is_train=True
    )
    val_dataset = PPEDataset(
        data_dir=config.VAL_DATA_DIR,
        class_names=config.CLASS_NAMES,
        image_size=config.IMAGE_SIZE,
        is_train=False # No augmentations for validation
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2, # Adjust based on your system
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False # Can speed up CPU to GPU data transfer
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE, # Can often be larger for validation if not doing backprop
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True if device == "cuda" else False
    )
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    print(f"Train Dataloader length: {len(train_loader)}, Val Dataloader length: {len(val_loader)}")

    # Model
    print("Initializing model...")
    model = PPEObjectDetector(
        backbone_name=config.BACKBONE,
        fpn_out_channels=config.FPN_OUT_CHANNELS,
        num_classes=config.NUM_CLASSES,
        num_anchors_per_level=config.NUM_ANCHORS_PER_LEVEL,
        pretrained_backbone=True 
    ).to(device)

    # Generate anchors (these are fixed based on config and don't need to be on device yet,
    # but DetectionLoss will move its copy to device)
    all_anchors_list = generate_all_anchors(
        image_size=config.IMAGE_SIZE,
        fpn_strides=config.ANCHOR_STRIDES,
        fpn_anchor_sizes=config.ANCHOR_SIZES,
        aspect_ratios=config.ASPECT_RATIOS
    )

    # Loss Function
    loss_fn = DetectionLoss(
        num_classes=config.NUM_CLASSES,
        all_anchors_list=all_anchors_list # Loss function will handle moving anchors to device
    )

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    # optimizer = optim.SGD(
    #     model.parameters(),
    #     lr=config.LEARNING_RATE,
    #     momentum=config.MOMENTUM,
    #     weight_decay=config.WEIGHT_DECAY
    # )
    
    # Learning Rate Scheduler (optional, but often helpful)
    # Example: ReduceLROnPlateau or StepLR
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)


    # --- Training Loop ---
    print("Starting training...")
    best_val_loss = float('inf')

    for epoch in range(config.NUM_EPOCHS):
        start_time = time.time()
        
        avg_train_loss, avg_train_cls_loss, avg_train_reg_loss = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, epoch, config.NUM_EPOCHS
        )
        
        avg_val_loss, avg_val_cls_loss, avg_val_reg_loss = validate_one_epoch(
            model, val_loader, loss_fn, device, epoch, config.NUM_EPOCHS
        )
        
        end_time = time.time()
        epoch_duration = end_time - start_time

        print(f"\n--- Epoch {epoch+1}/{config.NUM_EPOCHS} Summary ---")
        print(f"Time: {epoch_duration:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f} (Cls: {avg_train_cls_loss:.4f}, Reg: {avg_train_reg_loss:.4f})")
        print(f"Val Loss  : {avg_val_loss:.4f} (Cls: {avg_val_cls_loss:.4f}, Reg: {avg_val_reg_loss:.4f})")

        # Update LR scheduler
        if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(avg_val_loss)
        elif lr_scheduler is not None: # For other schedulers like StepLR
            lr_scheduler.step()

        # Save best model checkpoint (based on validation loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = os.path.join(weights_dir, f"ppe_detector_best_epoch{epoch+1}_valloss{avg_val_loss:.4f}.pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'class_names': config.CLASS_NAMES, # Save class names for later inference
                'image_size': config.IMAGE_SIZE
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")

        # Save last model checkpoint
        last_checkpoint_path = os.path.join(weights_dir, "ppe_detector_last.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_val_loss,
            'class_names': config.CLASS_NAMES,
            'image_size': config.IMAGE_SIZE
        }, last_checkpoint_path)
        print(f"Saved last model checkpoint to {last_checkpoint_path}")
        print("-" * 50)

    print("Training finished.")

if __name__ == '__main__':
    # Add sys.path modification here if running train.py directly from src/ and imports fail
    # import sys
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # project_root = os.path.dirname(script_dir)
    # if project_root not in sys.path:
    #    sys.path.append(project_root)
    
    main()