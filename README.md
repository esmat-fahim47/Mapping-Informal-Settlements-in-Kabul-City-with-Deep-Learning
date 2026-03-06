# Mapping-Informal-Settlements-in-Kabul-City-with-Deep-Learning

DeepLabV3 Informal Settlement Segmentation

This repository implements a semantic segmentation pipeline using DeepLabV3 with a ResNet-101 backbone to classify urban areas into three classes:

Formal settlements

Background / other land cover

Informal settlements

The model is trained on satellite imagery patches and predicts pixel-level land-use classes.
The pipeline includes dataset loading, training, validation, testing, evaluation metrics, patch-based inference, and city-scale sliding window prediction.

Model Architecture

The model uses:

DeepLabV3 segmentation architecture

ResNet-101 backbone pretrained on ImageNet

Custom dataset loader for image-mask semantic segmentation

Key characteristics:

Input: RGB satellite imagery

Output: 3-class semantic segmentation mask

Loss function: CrossEntropyLoss

Optimizer: Adam

Learning rate scheduler: StepLR

Project Pipeline

The workflow implemented in this repository:

Dataset preparation and loading

Data augmentation

Model initialization (DeepLabV3 + ResNet101)

Training

Validation and checkpointing

Testing and metric evaluation

Prediction mask export

Visualization

Large-scale sliding-window inference for full satellite scenes



Dataset Structure

The dataset must follow this directory structure:

dataset_root/
│
├── train/
│   ├── image/
│   │   ├── img_001.tif
│   │   ├── img_002.tif
│   │   └── ...
│   │
│   └── mask/
│       ├── img_001.tif
│       ├── img_002.tif
│       └── ...
│
├── val/
│   ├── image/
│   └── mask/
│
└── test/
    ├── image/
    └── mask/

Important requirements

Images must be RGB

Masks must be single-band TIFF

Mask values must represent class indices

0 = Formal
1 = Background
2 = Informal
Installation

Clone the repository:

git clone https://github.com/yourusername/informal-settlement-segmentation.git

cd informal-settlement-segmentation

Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt
Dependencies

Main libraries required for this project:

torch
torchvision
numpy
pillow
matplotlib
rasterio
scikit-learn
natsort

Optional (recommended):

tqdm
requirements.txt

Example requirements.txt:

torch
torchvision
numpy
pillow
matplotlib
rasterio
scikit-learn
natsort
Training

To train the model:

trained_model = train_model(
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    scheduler=scheduler,
    num_epochs=50,
    patience=20
)

Training includes:

pixel accuracy

mean IoU

confusion matrix

F1 score

precision

recall

The best model checkpoint is automatically saved as:

best_checkpoint.pth
Evaluation

Testing is performed with:

test_metrics = test_model(
    test_loader,
    trained_model,
    criterion
)

Metrics computed:

Global Pixel Accuracy

Mean IoU

IoU per class

F1 Score

Precision

Recall

Confusion Matrix

Producer Accuracy

User Accuracy

Predicted masks are exported as:

predictions/*.tif
Visualization

Predictions are visualized with a color map:

Class	Color	Meaning
0	Dark Blue	Formal
1	Gray	Background
2	Orange	Informal

Visualization includes:

Original image

Ground truth mask

Predicted mask

Large-Scale Inference

City-scale predictions are generated using sliding window inference.

Parameters:

Patch size: 224 × 224
Stride: 112 (50% overlap)
Batch size: 8

Steps:

Divide large satellite images into patches

Predict segmentation for each patch

Merge overlapping predictions

Average probabilities

Generate final classification map

Output:

predicted_mask (H × W)
Output Example

The final predicted segmentation map is converted to RGB using the class colormap and visualized using matplotlib.

Model Weights

After training, the final model weights are saved as:

model_resnet101.pth

To load for inference:

model.load_state_dict(torch.load("model_resnet101.pth"))
model.eval()
Applications

This repository is designed for:

Informal settlement mapping

Urban land-use classification

Satellite image semantic segmentation

Large-scale urban monitoring

License

This project is released under the MIT License.

Author

Esmat Tariq Fahim

Research focus:

Geospatial Machine Learning

Remote Sensing

Informal Settlement Detection

Semantic Segmentation of Satellite Imagery
