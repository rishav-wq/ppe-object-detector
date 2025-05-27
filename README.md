# PPE Object Detection Model

## Overview
This project implements an object detection model designed to identify Personal Protective Equipment (PPE) such as helmets, vests, boots, and gloves in images. The model utilizes a ResNet50 pre-trained backbone with a Feature Pyramid Network (FPN) and custom detection heads (classification and regression subnets) built on top.


## Features
-   Detects multiple PPE classes (e.g., 'boots', 'gloves', 'helmet', 'Person', 'Vest', etc.).
-   Built using PyTorch.
-   Backbone: ResNet50 (pre-trained on ImageNet).
-   Detection Architecture: Feature Pyramid Network (FPN).
-   Includes scripts for training, evaluation (mAP), and prediction/demo.

## Directory Structure

ppe_detector/
├── data/ # Contains class definition file (e.g., ppe_classes.txt or data.yaml)
│ └── README_DATA.md # Information about the dataset source and structure
├── demo_outputs/ # Sample output images from predict.py (not for submission, generated locally)
├── demo_showcase/ # Selected demo images for submission
├── final_model_weights/ # Contains the best trained .pth model file
├── src/ # Source code
│ ├── init.py
│ ├── backbone.py # ResNet50 backbone feature extractor
│ ├── config.py # Configuration parameters
│ ├── dataset.py # PyTorch Dataset and DataLoader for PPE data
│ ├── evaluate.py # Evaluation script (calculates mAP)
│ ├── fpn.py # FPN implementation
│ ├── heads.py # Detection heads
│ ├── loss.py # Loss functions (Focal Loss, Smooth L1)
│ ├── model.py # Full detector model combining backbone, FPN, heads
│ ├── predict.py # Script for running inference and generating demo images
│ ├── train.py # Main training script
│ └── utils.py # Utility functions (anchors, NMS, IoU, etc.)
├── weights/ # Saved model weights during training (not for submission, generated locally)
├── README.md # This file
├── requirements.txt # Python dependencies
└── Experience_Report.md # Report on the project experience (or .pdf/.docx)
└── evaluation_metrics.txt # Text file with mAP, Precision, Recall scores


## Setup Instructions

1.  **Clone the Repository (if applicable)**
    ```bash
    # git clone <repository-url>
    # cd ppe_detector
    ```

2.  **Create and Activate a Python Virtual Environment**
    It is highly recommended to use a virtual environment.
    ```bash
    python -m venv venv
    ```
    Activate the environment:
    *   Windows: `venv\Scripts\activate`
    *   macOS/Linux: `source venv/bin/activate`

3.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## Dataset

The model was trained and evaluated on a dataset of [**"construction site images," "industrial workplace images"**].
Annotations are in Pascal VOC XML format, with images and their corresponding XML files located in the same directory within `train/` and `valid/` splits.

*   **Class Definitions:** The classes detected by the model are defined in `data/ppe_classes.txt`. The current model is trained for the following classes: `['boots', 'gloves', 'helmet', 'Person', 'Vest', 'Glass', 'Mask', 'face', 'forearm', 'head', 'Ear-protection', 'wirst', 'protective clothing']`.
*   **Data Source:** [** "Downloaded from Kaggle/Roboflow Universe "**].
*   **Note:** The full raw image dataset is **not included** in this submission due to its size. To use your own data, structure it as described in `data/README_DATA.md`, ensuring images and XML annotations are placed within `data/train/` and `data/valid/` subdirectories (or update paths in `src/config.py`).

## How to Run

Ensure paths in `src/config.py` (especially `TRAIN_DATA_DIR`, `VAL_DATA_DIR`, `CLASSES_FILE`) are correctly set up if you are using your own dataset or a different directory structure.

1.  **Training the Model:**
    To train the model from scratch or fine-tune:
    ```bash
    python src/train.py
    ```
    Checkpoints (best and last) will be saved in the `weights/` directory (which will be created if it doesn't exist).

2.  **Evaluating the Model:**
    To evaluate a trained model (e.g., the best checkpoint) on the validation set and calculate mAP:
    ```bash
    python src/evaluate.py
    ```
    This script will automatically try to find the best checkpoint in `weights/` (or `final_model_weights/` if you've moved it there and adjusted the script) or use the last saved checkpoint. Evaluation results will be printed to the console.

3.  **Running Predictions (Demo):**
    To run inference on sample images and generate outputs with detections drawn:
    ```bash
    python src/predict.py
    ```
    This script will use the best (or last) checkpoint, process images (by default, a few random images from the validation set, or you can specify paths in the script), and save the output images to the `demo_outputs/` directory.

## Model Architecture
*   **Backbone:** ResNet50 (pre-trained on ImageNet).
*   **Neck:** Feature Pyramid Network (FPN) to generate multi-scale feature maps.
*   **Detection Heads:** Shared convolutional sub-networks for object classification (predicting class probabilities per anchor) and bounding box regression (predicting offsets from anchors).

## Evaluation Summary
(You can optionally place your `evaluation_metrics.txt` content here or refer to the separate file)

Key metrics on the validation set:
*   For detailed per-class AP and other metrics, please see `evaluation_metrics.txt` or the Experience Report.

Sample output images demonstrating the model's detection capabilities on test images can be found in the `DEMO_SHOWCASE/` folder. These images were generated using `src/predict.py` with the best trained model checkpoint.


[**Briefly mention any standout observations, e.g., "The model performs well on classes like 'helmet' and 'boots' but shows challenges with 'gloves'. The -1 AP for 'Person' and 'Vest' is attributed to their absence in the validation set ground truth."**]

## Further Details
For a detailed account of the project implementation, challenges, learnings, and AI tool usage, please refer to the `Experience_Report.md` (or `.pdf`/`.docx`).
