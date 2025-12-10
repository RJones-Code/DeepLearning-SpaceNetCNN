# SpaceNet7 Building Segmentation with U-Net

This project implements a U-Net model for building footprint segmentation on the SpaceNet7 dataset. It includes preprocessing, training, evaluation, and visualization tools.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Environment Setup](#environment-setup)
- [Dependencies](#dependencies)
- [Dataset](#dataset)
- [Configuration](#configuration)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Notes](#notes)
- [Contributions](#contributions)

---

## Project Overview

The goal of this project is to segment buildings in satellite imagery using a U-Net. The pipeline includes:

- Loading SpaceNet7 dataset tiles
- Creating smaller chips with image-mask pairs
- Training a U-Net with weighted BCE + Dice loss
- Evaluating model performance with metrics such as IoU, Dice, and accuracy
- Visualizing predictions with overlays

---

## Getting Started

### Clone the repository

```bash
git clone <https://github.com/RJones-Code/DeepLearning-SpaceNetCNN.git>
cd DeepLearning-SpaceNetCNN
```

## Creating Enviroment
We recommend using Python 3.10+ and creating a virtual environment.
```bash
python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate
```

## Dependencies
Install the required packages:
```bash
pip install -r requirements.txt
```

## Dataset
You can either:

1. Download manually from the website and place it in a local folder.

2. Use Kaggle API to automatically pull the dataset:
```bash
# test_load.py
# This script can pull the SpaceNet7 dataset from Kaggle
# Make sure you have created a Kaggle API token: https://www.kaggle.com/docs/api

from loadData import download_spacenet7

download_spacenet7(cache_dir="path/to/cache")  # optional cache directory
```

## Configuration
Create a .env file in the root of the repo:
```bash
SPACENET_ROOT=/path/to/SN7_buildings_train/train/
```
This should point to the folder containing the dataset images.

### Running the Main Driver
The main training and evaluation script is main.py. Run it as:
```bash
python main.py
```
Main will:
- Load the dataset
- Generate chips from full images
- nTrain the U-Net model
- Save the best model to unet_best.pth
- Save verification plots in verification_plots/
- Evaluate the model on a validation set

## Training the Model
You can customize training parameters such as:
- epochs
- batch_size
- learning_rate
- pos_weight / neg_weight for weighted loss
- patience for early stopping

Example call in main.py:
```bash
trained_model = train_model(
    dataset=train_ds,
    model=UNet_Model,
    epochs=50,
    batch_size=8,
    lr=1e-4,
    device=device,
    pos_weight=6,
    neg_weight=2,
    patience=10
)
```

## Evaluation
Use evaluate_model to compute metrics on a validation/test set:
```bash
from UNet.evaluate import evaluate_model

evaluate_model(trained_model, eval_ds, device=device)
```
Metrics computed:
- IoU (Intersection over Union)
- Dice coefficient
- Accuracy 

### Notes

---

- Generated chips are currently saved in memory. You can use preprocess_chips to save chips to a folder and load later to speed up training.
- Best model weights are saved as unet_best.pth automatically during training.
- Visualization outputs are saved under verification_plots/.
- Make sure your .env file points to the correct dataset root folder.

### Contributions

---

Russell Jones (RJones-Code): 
- UNet Model
- Metric Functions
- Learning Rate Scheduling
- Training

Theodore Boswell (PlatinumFrog):
- Data Loader
- Chips
- Evaluation
- Visualize Graphs
