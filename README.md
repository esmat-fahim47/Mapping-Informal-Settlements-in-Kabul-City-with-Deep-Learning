# Mapping Informal Settlements in Kabul with Deep Learning

This repository implements a semantic segmentation pipeline to classify Kabul city into three classes:

- **Formal settlements**
- **Background / other land cover**
- **Informal settlements**

This project focuses on mapping informal and background areas. The workflow includes dataset preparation, model training with a custom DeepLabV3-ResNet101 backbone, testing, visualization, evaluation metrics, and citywide inference using sliding-window batch predictions.

---

## Pipeline Overview

The project follows a structured workflow:

### 1. Dataset Preparation
- Custom PyTorch Dataset class loads images and masks
- Supports train / validation / test splits
- Optional augmentation (horizontal and vertical flips)
- ImageNet normalization for compatibility with pretrained ResNet backbones

### 2. Model Architecture
- DeepLabV3 segmentation model
- Custom ResNet-101 backbone wrapper
- Pretrained ImageNet weights
- Output classes configurable (`num_classes`)

### 3. Training
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam
- **Learning rate scheduler:** StepLR
- Early stopping with validation monitoring
- Best checkpoint automatically saved

### 4. Evaluation
Model performance is evaluated using:
- Pixel Accuracy
- Intersection over Union (IoU)
- Mean IoU
- Precision
- Recall
- F1 Score
- Confusion Matrix

### 5. Prediction Output
Predicted segmentation masks are saved as:
- Single-band GeoTIFF
- Pixel values correspond to class indices

### 6. Visualization of Predicted Results
- Original satellite image
- Ground Truth Mask
- Predicted segmentation mask

### 7. Citywide Inference
Large images are processed using a sliding-window approach:
- Overlapping patches
- Batch inference for efficiency
- Patch stitching with probability averaging
- Final full-resolution segmentation map

---

## Class Encoding

| Class ID | Description |
|----------|-------------|
| 0        | Formal      |
| 1        | Background  |
| 2        | Informal    |

---

## Requirements / Dependencies

This pipeline runs in **Jupyter Notebook** locally. The dataset should be downloaded and saved locally.

- Python >= 3.12.3
- Torch >= 2.5.1+cu121
- Torchvision >= 0.20.1+cu121
- Numpy >= 1.26.4
- Pillow >= 10.4.0
- Rasterio >= 1.4.3
- Scikit-learn >= 1.5.1
- Matplotlib >= 3.9.2
- Natsort >= 8.4.0

---

## Dataset Structure

The dataset should follow this folder structure:


 <img width="191" height="268" alt="kela" src="https://github.com/user-attachments/assets/9bc7c6a4-d5f7-40c1-970a-4944cf5bccf8" />



- Masks must be single-band TIFFs with class indices
- The image and corresponding mask must have the same name.

---

## Output Products

The pipeline produces:

- Trained segmentation model (`.pth`)
- Evaluation Metrics
- Best validation checkpoint
- Predicted masks for test images (Example Below):
- ![Picture1](https://github.com/user-attachments/assets/511d2aee-e09c-45ad-b835-a880db44a87a)
- Citywide segmentation GeoTIFF (Example Below):
  - <img width="488" height="429" alt="image" src="https://github.com/user-attachments/assets/cabe884f-7fe4-424e-b4a4-4eb02a978c52" />
  - <img width="488" height="418" alt="image" src="https://github.com/user-attachments/assets/a5725014-3da8-465d-80a7-0ce01546c490" />

---

## Notes

- GPU acceleration is automatically used if available.
- Windows users may need to set `num_workers=0` for PyTorch DataLoader.
- Input masks must contain class indices instead of RGB colors.

---

## TorchGeo Training Pipeline (Alternative / Experimental Implementation)

This repository also includes an alternative training pipeline using **TorchGeo** and **PyTorch Lightning** for mapping informal settlements in Kabul city.  

This pipeline replaces the manual PyTorch training loop with the TorchGeo data framework and Lightning configuration system, making the workflow more modular, reproducible, and scalable.

---

### Overview

The TorchGeo pipeline consists of three main components:

1. Dataset Class  
2. DataModule  
3. Lightning Training Configuration  

These components handle:

- Dataset loading  
- Augmentation  
- Normalization  
- Batching  
- Training configuration  

---

### Dataset

The same dataset used in the main pipeline can be used here.

#### Classes

| Class ID | Description         |
|----------|-------------------|
| 0        | Formal settlements |
| 1        | Background         |
| 2        | Informal settlements |

#### Expected Folder Structure

- Images must be RGB `.tif` files
- Masks must contain class indices (0,1,2)

Example:





The dataset loader automatically:

- Loads images
- Converts them to tensors
- Normalizes pixel values to [0,1]
- Loads segmentation masks

---

### DataModule

The `KabulInformalSettlementDatamodule` manages dataset loading and preprocessing.

**Data Augmentation (Training)**

- Random horizontal flip
- Random vertical flip

These augmentations are applied jointly to images and masks using **Kornia**.

**Normalization**

- Images are normalized using ImageNet statistics:
- mean = [0.485, 0.456, 0.406]
- std = [0.229, 0.224, 0.225]

**Batch Configuration**

- Default settings:
- batch_size = 16
- num_workers = 0


---

### Model Configuration

The model is defined using a Lightning configuration file.

- **Architecture:** DeepLabV3+
- **Backbone:** ResNet-101
- **Pretrained weights:** ImageNet
- **Segmentation Classes:** `num_classes = 3`
- **Loss Function:** Cross Entropy

---

### Training Configuration

Example Lightning configuration:

```yaml
model:
  class_path: datasets.custom_task.CustomSemanticSegmentationTask
  init_args:
    model: "deeplabv3"
    backbone: "resnet101"
    weights: true
    in_channels: 3
    num_classes: 3
    loss: "ce"
    lr: 1e-3
    tmax: 50

data:
  class_path: datasets.kabul_informal_settlement_datamodule.KabulInformalSettlementDatamodule
  init_args:
    batch_size: 16
    root: "path_to_dataset"

trainer:
  max_epochs: 50
  accelerator: "gpu"
  devices: 1
  precision: 32
  log_every_n_steps: 10

```
### Logging

Training logs are recorded using TensorBoard.

Logs are saved in:

lightning_logs/torchgeo/

You can monitor training with:

tensorboard --logdir lightning_logs

### Advantages of the TorchGeo Pipeline

Modular dataset management

Clean separation of data and model logic

Lightning training configuration

Reproducible experiments

### Disadvantages of the TorchGeo Pipeline

Currently only supports training and obtaining overall F1, Precision, Recall, and Pixel Accuracy.

Inference on larger images depends on newer versions of TorchGeo. Check TorchGeo GitHub
 for updates.

### Pipeline Components
- kabul_informal_settlement_dataset.py
- kabul_informal_settlement_datamodule.py
- config.yaml

### Running the Pipeline
Training can be launched with Lightning CLI:
python train.py fit --config configs/torchgeo_training.yaml
