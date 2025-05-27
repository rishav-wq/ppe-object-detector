# src/config.py
import torch
import os # Import os for path joining
import yaml # Import yaml if you might use data.yaml

# --- Dataset Configuration ---
# MAIN_DATA_PARENT_DIR = "../data/" # Parent of your specific dataset folder
# YOUR_DATASET_FOLDER_NAME = "PPE_DATASET_ROBOFLOW" # Or whatever your dataset is called
# MAIN_DATA_DIR = os.path.join(MAIN_DATA_PARENT_DIR, YOUR_DATASET_FOLDER_NAME)

# OR, if your structure is simply ../data/train, ../data/valid:
TRAIN_DATA_DIR = "../data/train/"  # UPDATE: Path to your training folder (mixed images/XMLs)
VAL_DATA_DIR = "../data/valid/"    # UPDATE: Path to your validation folder (mixed images/XMLs)
# TEST_DATA_DIR = "../data/test/"  # Optional: Path to your test folder

# CLASSES_FILE: This needs to point to where your class definition file is.
# Option A: If it's ppe_classes.txt inside ../data/
CLASSES_FILE = "../data/ppe_classes.txt" # Keep if this is correct

# Option B: If it's a data.yaml from Roboflow inside one of your dataset folders
# (e.g., ../data/PPE_DATASET_ROBOFLOW/data.yaml)
# CLASSES_FILE = os.path.join(MAIN_DATA_DIR, "data.yaml") # Adjust MAIN_DATA_DIR if using this

IMAGE_SIZE = 640

# --- Model Configuration --- (Keep as is)
BACKBONE = 'resnet50'
FPN_OUT_CHANNELS = 256
# NUM_CLASSES will be set by get_class_names

# Anchor configuration (Keep as is)
ANCHOR_STRIDES = [8, 16, 32, 64]
ANCHOR_SIZES = [
    [32, 64, 128],
    [64, 128, 256],
    [128, 256, 512],
    [256, 512, 768]
]
ASPECT_RATIOS = [0.5, 1.0, 2.0]
NUM_ANCHORS_PER_LEVEL = len(ANCHOR_SIZES[0]) * len(ASPECT_RATIOS)

# --- Training Configuration --- (Keep as is, adjust BATCH_SIZE if memory issues)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 # Reduced batch size as a precaution, can increase if memory allows
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MOMENTUM = 0.9
IOU_THRESHOLD_POSITIVE = 0.5
IOU_THRESHOLD_NEGATIVE = 0.4

# --- Post-processing --- (Keep as is)
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.45

# --- Utility to load class names ---
def get_class_names():
    class_names_list = []
    if not os.path.exists(CLASSES_FILE):
        print(f"CRITICAL: CLASSES_FILE not found at {CLASSES_FILE}")
        return ['default_class'] # Fallback to prevent crash, but indicates error

    if CLASSES_FILE.endswith(".yaml"):
        with open(CLASSES_FILE, 'r') as f:
            try:
                data_yaml = yaml.safe_load(f)
                if 'names' in data_yaml and isinstance(data_yaml['names'], list):
                    class_names_list = data_yaml['names']
                    print(f"Loaded {len(class_names_list)} classes from YAML: {class_names_list}")
                else:
                    print(f"Warning: 'names' key not found or not a list in {CLASSES_FILE}")
            except yaml.YAMLError as e:
                print(f"Error parsing YAML file {CLASSES_FILE}: {e}")
    elif CLASSES_FILE.endswith(".txt"):
         with open(CLASSES_FILE, 'r') as f:
            class_names_list = [line.strip() for line in f.readlines() if line.strip()]
            print(f"Loaded {len(class_names_list)} classes from TXT: {class_names_list}")
    else:
        print(f"Warning: Unknown format for CLASSES_FILE: {CLASSES_FILE}. Expecting .txt or .yaml")

    if not class_names_list:
         print(f"CRITICAL: No class names loaded from {CLASSES_FILE}. Please check the path and format.")
         return ['error_loading_classes'] # Fallback
    return class_names_list

CLASS_NAMES = get_class_names()
NUM_CLASSES = len(CLASS_NAMES)
if 'error_loading_classes' in CLASS_NAMES or 'default_class' in CLASS_NAMES :
    print("WARNING: Problem loading class names. NUM_CLASSES might be incorrect.")


print(f"Device: {DEVICE}")
print(f"Number of classes: {NUM_CLASSES} ({CLASS_NAMES})")
print(f"Training data directory: {TRAIN_DATA_DIR}")
print(f"Validation data directory: {VAL_DATA_DIR}")