import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# src/predict.py
import torch
import torchvision.transforms.v2 as T
from torchvision import tv_tensors
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import random
import glob

from src import config
from src.model import PPEObjectDetector
from src.utils import generate_all_anchors, postprocess_detections_per_image

# For Colab/Jupyter display (optional)
# from IPython.display import display

def load_model_for_inference(checkpoint_path, device):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    loaded_class_names = checkpoint.get('class_names', config.CLASS_NAMES)
    loaded_image_size = checkpoint.get('image_size', config.IMAGE_SIZE)
    num_classes = len(loaded_class_names)

    print(f"Model trained with {num_classes} classes: {loaded_class_names}")
    print(f"Model trained with image size: {loaded_image_size}")

    model = PPEObjectDetector(
        backbone_name=config.BACKBONE,
        fpn_out_channels=config.FPN_OUT_CHANNELS,
        num_classes=num_classes,
        num_anchors_per_level=config.NUM_ANCHORS_PER_LEVEL,
        pretrained_backbone=False 
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set to evaluation mode
    print("Model loaded successfully and set to evaluation mode.")
    
    return model, loaded_class_names, loaded_image_size

def preprocess_image(image_path, target_size):
    """Loads and preprocesses a single image for inference."""
    try:
        image_pil = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Error: Image not found at {image_path}")
        return None, None

    original_w, original_h = image_pil.width, image_pil.height
    
    # Basic transforms: Resize and ToTensor
    # No data augmentation for inference
    transform = T.Compose([
        T.Resize((target_size, target_size), antialias=True),
        T.ToDtype(torch.float32, scale=True), # Converts PIL to Tensor and scales to [0,1]
    ])
    
    # Wrap PIL image in tv_tensors.Image for v2 transforms
    image_tv = tv_tensors.Image(image_pil)
    transformed_image_tv = transform(image_tv) # This will be a tv_tensors.Image containing a tensor
    
    # Extract the tensor data
    img_tensor = transformed_image_tv.data if isinstance(transformed_image_tv, tv_tensors.Image) else transformed_image_tv
    
    return img_tensor, (original_h, original_w)


def draw_detections_on_image(image_pil, boxes, labels, scores, class_names, colors):
    """Draws bounding boxes and labels on a PIL image."""
    draw = ImageDraw.Draw(image_pil)
    try:
        # Try to load a nicer font, fallback to default
        font = ImageFont.truetype("arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    for box, label_idx, score in zip(boxes, labels, scores):
        xmin, ymin, xmax, ymax = box
        class_name = class_names[label_idx]
        color = colors[label_idx % len(colors)] # Cycle through colors

        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline=color, width=3)
        
        text = f"{class_name}: {score:.2f}"
        # Get text size to position it
        # For Pillow < 9.0.0 textsize = draw.textsize(text, font=font)
        # For Pillow >= 9.0.0
        if hasattr(draw, "textbbox"): # Pillow >= 9.0.0
             text_bbox = draw.textbbox((xmin, ymin), text, font=font)
             text_width = text_bbox[2] - text_bbox[0]
             text_height = text_bbox[3] - text_bbox[1]
        else: # Older Pillow
            text_width, text_height = draw.textsize(text, font=font)


        text_x = xmin
        text_y = ymin - text_height - 2  # Position text above the box
        if text_y < 0: # If text goes off screen, put it inside
            text_y = ymin + 2

        # Draw a filled rectangle behind the text for better visibility
        draw.rectangle(
            [(text_x, text_y), (text_x + text_width, text_y + text_height)],
            fill=color
        )
        draw.text((text_x, text_y), text, fill="black", font=font) # Black text

    return image_pil


def predict_on_image(model, image_path, all_anchors_cat, class_names, target_image_size, device,
                     conf_threshold=config.CONFIDENCE_THRESHOLD, 
                     nms_iou_thresh=config.NMS_IOU_THRESHOLD):
    
    img_tensor, (original_h, original_w) = preprocess_image(image_path, target_image_size)
    if img_tensor is None:
        return None

    img_tensor = img_tensor.unsqueeze(0).to(device) # Add batch dimension and move to device

    with torch.no_grad():
        cls_preds_list_raw, reg_preds_list_raw = model(img_tensor)

    # Concatenate and apply sigmoid (similar to evaluate.py)
    cls_preds_batch = torch.cat([preds.view(preds.size(0), -1, len(class_names)) for preds in cls_preds_list_raw], dim=1)
    reg_preds_batch = torch.cat([preds.view(preds.size(0), -1, 4) for preds in reg_preds_list_raw], dim=1)
    cls_scores_batch = torch.sigmoid(cls_preds_batch)

    # Get predictions for the single image in the batch
    img_cls_scores = cls_scores_batch[0] # [total_anchors, num_classes]
    img_reg_preds = reg_preds_batch[0]   # [total_anchors, 4]

    # Post-process
    # The image_shape for postprocessing should be the size the model sees (target_image_size)
    # as the predicted boxes are relative to this scaled image.
    pred_boxes_scaled, pred_scores, pred_labels = postprocess_detections_per_image(
        img_cls_scores,
        img_reg_preds,
        all_anchors_cat.to(device),
        image_shape=(target_image_size, target_image_size),
        conf_threshold=conf_threshold,
        nms_iou_thresh=nms_iou_thresh
    )

    # Scale boxes back to original image size
    if pred_boxes_scaled.numel() > 0:
        scale_x = original_w / target_image_size
        scale_y = original_h / target_image_size
        
        final_boxes_original_scale = pred_boxes_scaled.clone().cpu()
        final_boxes_original_scale[:, 0] *= scale_x # xmin
        final_boxes_original_scale[:, 1] *= scale_y # ymin
        final_boxes_original_scale[:, 2] *= scale_x # xmax
        final_boxes_original_scale[:, 3] *= scale_y # ymax

        # Clip to original image boundaries just in case scaling pushes them out slightly
        final_boxes_original_scale[:, 0].clamp_(min=0, max=original_w)
        final_boxes_original_scale[:, 1].clamp_(min=0, max=original_h)
        final_boxes_original_scale[:, 2].clamp_(min=0, max=original_w)
        final_boxes_original_scale[:, 3].clamp_(min=0, max=original_h)

        return final_boxes_original_scale, pred_labels.cpu(), pred_scores.cpu()
    else:
        return torch.empty((0,4)), torch.empty((0), dtype=torch.long), torch.empty((0))


if __name__ == '__main__':
    device = config.DEVICE

    # --- Configuration for Demo ---
    # 1. Find Best Checkpoint or use Last
    weights_dir = "../weights"
    best_checkpoint_path = None
    best_val_loss = float('inf')
    if os.path.exists(weights_dir):
        for f_name in os.listdir(weights_dir):
            if f_name.startswith("ppe_detector_best_epoch") and f_name.endswith(".pth"):
                try:
                    loss_str = f_name.split("valloss")[1].replace(".pth", "")
                    val_loss = float(loss_str)
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_checkpoint_path = os.path.join(weights_dir, f_name)
                except Exception: pass 
    
    if not best_checkpoint_path and os.path.exists(os.path.join(weights_dir, "ppe_detector_last.pth")):
        best_checkpoint_path = os.path.join(weights_dir, "ppe_detector_last.pth")
        print("Using last checkpoint as best was not found by naming convention.")
    
    if not best_checkpoint_path:
        print(f"No checkpoint found in {weights_dir}. Please train the model first.")
        exit()

    # 2. Load Model
    model, class_names, trained_image_size = load_model_for_inference(best_checkpoint_path, device)

    # 3. Generate Anchors (must match model's training config)
    all_anchors_list = generate_all_anchors(
        image_size=trained_image_size, 
        fpn_strides=config.ANCHOR_STRIDES,
        fpn_anchor_sizes=config.ANCHOR_SIZES,
        aspect_ratios=config.ASPECT_RATIOS
    )
    all_anchors_cat = torch.cat(all_anchors_list, dim=0)

    # 4. Define Image Path(s) for Demo
    #    Replace this with path(s) to your own test images
    #    You can get some images from your validation set: config.VAL_DATA_DIR
    demo_image_paths = []
    if os.path.exists(config.VAL_DATA_DIR):
        val_images = glob.glob(os.path.join(config.VAL_DATA_DIR, "*.jpg")) # or .png, .jpeg
        val_images.extend(glob.glob(os.path.join(config.VAL_DATA_DIR, "*.png")))
        val_images.extend(glob.glob(os.path.join(config.VAL_DATA_DIR, "*.jpeg")))
        if val_images:
            # Select a few random images from validation set for demo
            num_demo_images = min(5, len(val_images))
            demo_image_paths = random.sample(val_images, num_demo_images)
        else:
            print(f"No images found in validation directory: {config.VAL_DATA_DIR}")
    
    if not demo_image_paths:
        # Fallback if no validation images found, provide a placeholder
        print("Please provide path(s) to image(s) for the demo in `demo_image_paths` list.")
        # demo_image_paths = ["path/to/your/image1.jpg", "path/to/your/image2.png"] 
        exit()
    
    print(f"\nRunning demo on {len(demo_image_paths)} images: {demo_image_paths}")

    # 5. Create output directory for demo images
    demo_output_dir = "../demo_outputs"
    os.makedirs(demo_output_dir, exist_ok=True)

    # 6. Define colors for classes (optional, for nicer visualization)
    #    Generate random colors if you have many classes
    colors = []
    for _ in range(len(class_names)):
        colors.append(tuple(np.random.randint(50, 255, size=3).tolist()))


    # 7. Loop through demo images
    for img_path in demo_image_paths:
        print(f"\nProcessing: {img_path}")
        
        # Perform prediction
        pred_boxes, pred_labels, pred_scores = predict_on_image(
            model, img_path, all_anchors_cat, class_names, trained_image_size, device
        )

        if pred_boxes is None : # Error during preprocessing
            continue

        print(f"  Found {len(pred_scores)} objects.")
        # for s, l, b in zip(pred_scores, pred_labels, pred_boxes):
        #    print(f"    Class: {class_names[l]}, Score: {s:.2f}, Box: {b.tolist()}")

        # Load original image again for drawing (to draw on full resolution)
        image_to_draw_on = Image.open(img_path).convert("RGB")
        
        # Draw detections
        result_image = draw_detections_on_image(
            image_to_draw_on, 
            pred_boxes.numpy(), # Convert to numpy for Pillow
            pred_labels.numpy(), 
            pred_scores.numpy(),
            class_names,
            colors
        )

        # Save or display
        base_img_name = os.path.basename(img_path)
        output_path = os.path.join(demo_output_dir, f"detected_{base_img_name}")
        result_image.save(output_path)
        print(f"  Saved detection result to: {output_path}")

        # Optional: Display image (e.g., in Jupyter/Colab or using matplotlib)
        # For local script, matplotlib can be used:
        # import matplotlib.pyplot as plt
        # plt.imshow(result_image)
        # plt.axis('off')
        # plt.title(f"Detections for {base_img_name}")
        # plt.show()
        
        # In Colab/Jupyter:
        # display(result_image) 

    print(f"\nDemo finished. Output images are in: {demo_output_dir}")