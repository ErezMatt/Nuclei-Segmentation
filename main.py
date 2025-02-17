import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchstain
import albumentations as A
from sklearn.model_selection import train_test_split
from PIL import Image

from config import *
from utils import *
from data import MoNuSegDataset
from models import UNet, ResUNet, LinkNet34
from loss import MultiLoss 
import metrics

# Mapping model names to their corresponding classes
MODELS = {
    "unet" : UNet,
    "resunet" : ResUNet,
    "linknet" : LinkNet34
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", choices=[*MODELS], default="unet")
    parser.add_argument("--train", action="store_true", help="Enable training mode")
    parser.add_argument("--resume", action="store_true", help="Load model from existing checkpoint")
    args = parser.parse_args()
    training_mode = args.train
    print(DEVICE)

    # Define evaluation metrics
    model_metrics = {
        "accuracy": metrics.accuracy, 
        "dice": metrics.dice_coef, 
        "iou": metrics.binary_iou
    }

    # Load training image names
    images_names = os.listdir(os.path.join(TRAIN_DIR_PATH, "img"))
    if not images_names:
        raise ValueError("No training images found")
    
    # Load test image names
    test_img_names = os.listdir(os.path.join(TEST_DIR_PATH, "img"))
    if not test_img_names:
        raise ValueError("No test images found")

    train_names, val_names = train_test_split(images_names, test_size=0.1)
    
    # Define data augmentation transformations for training images
    train_transform = A.Compose([
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ])
    
    # Create stain normalizer and fit it to a training sample image
    normalizer_target_path = os.path.join(TRAIN_DIR_PATH, "img", images_names[0])
    normalizer_target = Image.open(normalizer_target_path).convert("RGB")
    normalizer_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x*255)
    ])
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='torch')
    normalizer.fit(normalizer_transform(normalizer_target))

    # Create datasets for training, validation, and testing
    train_dataset = MoNuSegDataset(TRAIN_DIR_PATH, train_names, patch_size=PATCH_SIZE, size=IMG_SIZE, normalizer=normalizer, transform=train_transform)
    val_dataset = MoNuSegDataset(TRAIN_DIR_PATH, val_names, patch_size=PATCH_SIZE, size=IMG_SIZE, normalizer=normalizer)
    test_dataset = MoNuSegDataset(TEST_DIR_PATH, test_img_names, patch_size=PATCH_SIZE, size=IMG_SIZE, normalizer=normalizer)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    # Ensure that the selected model exists in MODELS
    if args.model not in MODELS:
        raise ValueError(f"Invalid model choice '{args.model}'. Choose from {list(*MODELS)}")
    # Initialize the selected model
    model = MODELS[args.model](3, 1).to(device=DEVICE)

    print(f"Model: {model.name}")
    loss = MultiLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

     # Load checkpoint if resuming or not in training mode
    if not training_mode or args.resume:
        model_path = os.path.join(CHECKPOINT_DIR, f"{model.name}.pth")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        else:
            print(f"Checkpoint '{model_path}' not found. Training from scratch.")
            training_mode = True

    if training_mode:
        history = train_model(model, train_loader, val_loader, optimizer, loss, EPOCHS, model_metrics=model_metrics, device=DEVICE)
        plot_training_history(history, model_metrics)
    
    # Evaluate the model on test data
    run_epoch(model, test_loader, loss, model_metrics=model_metrics, device=DEVICE)
    
    # Save test dataset predictions
    save_model_predictions(model, test_loader, "test_results", device=DEVICE)
