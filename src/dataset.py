# src/dataset.py


import torch
from torch.utils.data import Dataset, DataLoader
import glob # os is already imported by sys.path block if used
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision.transforms.v2 as T
from torchvision import tv_tensors # Import tv_tensors
from src import config

class PPEDataset(Dataset):
    def __init__(self, data_dir, class_names, image_size, transforms=None, is_train=True):
        self.data_dir = data_dir
        self.class_names = class_names
        self.class_to_idx = {name: i for i, name in enumerate(class_names)}
        self.image_size = image_size # This is the target size *after* resize
        self.is_train = is_train

        image_extensions = ["*.jpg", "*.png", "*.jpeg"]
        self.image_files = []
        for ext in image_extensions:
            self.image_files.extend(sorted(glob.glob(os.path.join(self.data_dir, ext))))
        
        if not self.image_files:
            raise FileNotFoundError(f"No images found in {self.data_dir} with .jpg, .png, or .jpeg extensions.")

        valid_image_files = []
        for img_path in self.image_files:
            base_filename = os.path.splitext(os.path.basename(img_path))[0]
            ann_path_exact = os.path.join(self.data_dir, base_filename + ".xml")
            if os.path.exists(ann_path_exact):
                valid_image_files.append(img_path)
        
        self.image_files = valid_image_files
        if not self.image_files:
            raise FileNotFoundError(f"No images found with corresponding XML annotations in {self.data_dir}.")

        if transforms is None:
            self.transforms = self._get_default_transforms()
        else:
            self.transforms = transforms

    def _get_default_transforms(self):
        transform_list = []
        if self.is_train:
            transform_list.extend([
                T.RandomPhotometricDistort(p=0.5),
                T.RandomZoomOut(fill=(123, 117, 104), side_range=(1.0, 1.5), p=0.3),
                T.RandomApply([
                    T.RandomIoUCrop(min_scale=0.3, max_scale=1.0, min_aspect_ratio=0.5, max_aspect_ratio=2.0)
                ], p=0.5),
                T.RandomHorizontalFlip(p=0.5),
            ])
        
        transform_list.extend([
            # Resize will operate on image and tv_tensors.BoundingBoxes correctly
            T.Resize((self.image_size, self.image_size), antialias=True), 
            T.ToDtype(torch.float32, scale=True), 
            T.SanitizeBoundingBoxes(), 
        ])
        return T.Compose(transform_list)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        base_filename = os.path.splitext(os.path.basename(img_path))[0]
        ann_path = os.path.join(self.data_dir, base_filename + ".xml")

        try:
            image_pil = Image.open(img_path).convert("RGB")
            original_h, original_w = image_pil.height, image_pil.width # PIL (height, width)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}. Skipping.")
            return None 

        boxes_list = []
        labels_list = []
        if os.path.exists(ann_path):
            try:
                tree = ET.parse(ann_path)
                root = tree.getroot()
                for member in root.findall("object"):
                    class_name = member.find("name").text
                    if class_name not in self.class_to_idx:
                        continue 
                    xmin = float(member.find("bndbox/xmin").text)
                    ymin = float(member.find("bndbox/ymin").text)
                    xmax = float(member.find("bndbox/xmax").text)
                    ymax = float(member.find("bndbox/ymax").text)
                    if xmax <= xmin or ymax <= ymin:
                        continue
                    boxes_list.append([xmin, ymin, xmax, ymax])
                    labels_list.append(self.class_to_idx[class_name])
            except Exception as e:
                print(f"Error parsing annotation {ann_path}: {e}. Assuming no objects.")
        
        boxes_tensor = torch.tensor(boxes_list, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_list, dtype=torch.long)

        # Wrap boxes_tensor in tv_tensors.BoundingBoxes
        # canvas_size is (height, width)
        bounding_boxes_tv = tv_tensors.BoundingBoxes(
            boxes_tensor, 
            format=tv_tensors.BoundingBoxFormat.XYXY, 
            canvas_size=(original_h, original_w)
        )
        # Note: if boxes_tensor is empty (0 objects), bounding_boxes_tv will be an empty BoundingBoxes object.
        # e.g., tv_tensors.BoundingBoxes(tensor([], size=(0, 4)), format=XYXY, canvas_size=(h, w))

        target = {}
        target['boxes'] = bounding_boxes_tv
        target['labels'] = labels_tensor # Labels remain a simple tensor
        target['image_id'] = torch.tensor([idx]) # For potential COCO eval

        # The image passed to transforms should ideally also be a tv_tensor.Image
        # or transforms.v2 will convert it. Let's be explicit.
        image_tv = tv_tensors.Image(image_pil)


        # Apply transforms
        if self.transforms:
            # Pass tv_tensors.Image and the target dict containing tv_tensors.BoundingBoxes
            transformed_image_tv, transformed_target = self.transforms(image_tv, target)
        else:
            # This case should ideally not happen if default transforms are set
            print("Warning: No transforms applied. Raw image and target returned.")
            transformed_image_tv = image_tv 
            transformed_target = target

        # Extract plain tensors for DataLoader collation and loss function
        # The image from transforms will be a tv_tensors.Image containing a torch.Tensor
        final_image_tensor = transformed_image_tv.data if isinstance(transformed_image_tv, tv_tensors.Image) else transformed_image_tv

        final_boxes_obj = transformed_target['boxes']
        if isinstance(final_boxes_obj, tv_tensors.BoundingBoxes):
            final_boxes_tensor = final_boxes_obj.data
            # Ensure even empty boxes have the correct device if possible (though unlikely to matter for empty)
            if final_boxes_tensor.numel() == 0 and hasattr(final_boxes_obj, 'device'):
                 final_boxes_tensor = final_boxes_tensor.to(final_boxes_obj.device)
        else: # Should not happen if transforms pipeline is correct
            print(f"Warning: transformed_target['boxes'] is not a BoundingBoxes object, but {type(final_boxes_obj)}")
            final_boxes_tensor = final_boxes_obj 
        
        # Ensure empty boxes tensor still has 2 dimensions (N, 4) -> (0, 4)
        if final_boxes_tensor.numel() == 0 and final_boxes_tensor.ndim == 1:
            final_boxes_tensor = final_boxes_tensor.reshape(0, 4)


        final_labels_tensor = transformed_target['labels']
        
        return final_image_tensor, final_boxes_tensor, final_labels_tensor

def collate_fn(batch):
    batch = [item for item in batch if item is not None] 
    if not batch: 
        return None, None, None
    images = []
    batched_gt_boxes = []
    batched_gt_labels = []
    for item in batch:
        images.append(item[0])
        batched_gt_boxes.append(item[1])
        batched_gt_labels.append(item[2])
    
    try:
        images_tensor = torch.stack(images, 0)
    except RuntimeError as e:
        print(f"Error during torch.stack in collate_fn: {e}")
        print("This might be due to inconsistent image tensor shapes after transformation.")
        for i, img in enumerate(images):
            print(f"Image {i} shape: {img.shape}")
        # Propagate the error or return None to skip batch
        return None, None, None


    return images_tensor, batched_gt_boxes, batched_gt_labels

if __name__ == '__main__':
    # Ensure config is loaded correctly if running this file directly
    # The sys.path.append at the top should help 'from src import config' work
    # when 'python src/dataset.py' is run from the project root 'ppe_detector/'.
    # If running from 'ppe_detector/src/', then 'from . import config' might be an alternative
    # but 'from src import config' with proper sys.path is generally more robust for testing.

    print(f"Using classes from config: {config.CLASS_NAMES}")
    print(f"Number of classes from config: {config.NUM_CLASSES}")

    actual_data_path_to_test = config.TRAIN_DATA_DIR 
    
    if not os.path.exists(actual_data_path_to_test):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        resolved_path_check = os.path.normpath(os.path.join(script_dir, actual_data_path_to_test))
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"TEST DATA PATH NOT FOUND using config.TRAIN_DATA_DIR: '{actual_data_path_to_test}'")
        print(f"Attempted to resolve to: '{resolved_path_check}' (does this exist?)")
        print(f"Please ensure 'TRAIN_DATA_DIR' in config.py is correct and the directory exists.")
        print(f"Current working directory: {os.getcwd()}")
        print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    else:
        abs_data_path = os.path.abspath(actual_data_path_to_test)
        print(f"--- Testing PPEDataset with actual data from (abs path): {abs_data_path} ---")
        try:
            actual_dataset = PPEDataset(
                data_dir=abs_data_path, # Pass absolute path for clarity in test
                class_names=config.CLASS_NAMES,
                image_size=config.IMAGE_SIZE,
                is_train=True
            )
            print(f"Found {len(actual_dataset)} samples in the dataset.")

            if len(actual_dataset) > 0:
                for i in range(min(3, len(actual_dataset))): 
                    print(f"\n--- Testing Sample {i} ---")
                    sample = actual_dataset[i]
                    if sample is None:
                        print(f"Sample {i} could not be loaded or was skipped.")
                        continue
                    img, gt_boxes, gt_labels = sample
                    print(f"  Image shape: {img.shape}, Image dtype: {img.dtype}")
                    print(f"  GT Boxes shape: {gt_boxes.shape}, GT Boxes dtype: {gt_boxes.dtype}")
                    print(f"  GT Labels shape: {gt_labels.shape}, GT Labels dtype: {gt_labels.dtype}")
                    print(f"  GT Boxes (first 5):\n{gt_boxes[:5]}")
                    print(f"  GT Labels (first 5):\n{gt_labels[:5]}")
                    
                    assert isinstance(img, torch.Tensor), f"Image for sample {i} is not a tensor"
                    assert isinstance(gt_boxes, torch.Tensor), f"GT Boxes for sample {i} is not a tensor"
                    assert isinstance(gt_labels, torch.Tensor), f"GT Labels for sample {i} is not a tensor"

                    assert img.shape[0] == 3 and img.shape[1] == config.IMAGE_SIZE and img.shape[2] == config.IMAGE_SIZE, \
                        f"Unexpected image shape for sample {i}: {img.shape}"
                    
                    if gt_boxes.numel() > 0:
                        assert gt_boxes.ndim == 2 and gt_boxes.shape[1] == 4, \
                            f"Unexpected gt_boxes shape for sample {i}: {gt_boxes.shape}"
                        assert gt_labels.ndim == 1, f"Unexpected gt_labels shape for sample {i}: {gt_labels.shape}"
                        assert gt_boxes.shape[0] == gt_labels.shape[0], \
                            f"Mismatch in num boxes and labels for sample {i}"
                    else: 
                        assert gt_labels.numel() == 0, \
                            f"Labels present for sample {i} while boxes are empty"
                        assert gt_boxes.shape == (0,4), \
                            f"Empty boxes tensor has wrong shape for sample {i}: {gt_boxes.shape}"
                
                print("\n--- Testing DataLoader with actual data ---")
                actual_loader = DataLoader(
                    actual_dataset,
                    batch_size=2, # Small batch size for test
                    shuffle=False, # False for reproducible test
                    collate_fn=collate_fn,
                    num_workers=0 
                )

                for batch_idx, batch_data in enumerate(actual_loader):
                    if batch_data is None or batch_data[0] is None:
                        print(f"Batch {batch_idx + 1} was skipped or had an error during collation.")
                        continue
                    
                    images_batch, gt_boxes_batch, gt_labels_batch = batch_data
                    print(f"\nBatch {batch_idx + 1}:")
                    print(f"  Images tensor shape: {images_batch.shape}, dtype: {images_batch.dtype}")
                    print(f"  Length of GT Boxes list: {len(gt_boxes_batch)}")
                    print(f"  Length of GT Labels list: {len(gt_labels_batch)}")
                    
                    if len(gt_boxes_batch) > 0:
                        print(f"    GT Boxes for first image in batch shape: {gt_boxes_batch[0].shape}")
                        print(f"    GT Labels for first image in batch shape: {gt_labels_batch[0].shape}")
                    if batch_idx >= 0: # Test only one or two batches
                        break 
            else:
                print("No samples found. Check dataset path and file structure.")

        except Exception as e:
            import traceback
            print(f"Error during actual dataset test: {e}")
            traceback.print_exc()