import os
import torch
import torchvision
import matplotlib.pyplot as plt

from metrics import *
from config import CHECKPOINT_DIR

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def patchify(image, size=200, stride=200):
    """Split an image into patches of given size with a specified stride"""
    patches =  image.unfold(1, size, stride).unfold(2, size, stride)
    p = patches.permute(1,2,0,3,4)
    p = p.reshape(-1, p.shape[2], size, size)
    return p

def compute_metrics(outputs, masks, model_metrics):
    """Compute metrics (e.g., accuracy, dice, IoU) for model predictions"""
    with torch.no_grad():
        preds = torch.sigmoid(outputs)
        preds = (preds > 0.5).float()

    return {key: func(preds, masks).item() for key, func in model_metrics.items()}

def run_epoch(model, loader, loss_fn, model_metrics, optimizer=None, device='cpu'):
    """
    Run one epoch of training or validation.

    Returns:
        dict: Averaged loss and metrics for the epoch.
    """
    training = optimizer is not None

    if training:
        mode = "Train"
        model.train()
    else:
        mode = "Validation"
        model.eval()

    metrics = {key : 0.0 for key in ["loss",  *model_metrics]}
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.float().to(device)

        with torch.set_grad_enabled(training):
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            metrics['loss'] += loss.item()

            batch_metrics = compute_metrics(outputs, masks, model_metrics)
            for key in batch_metrics:
                metrics[key] += batch_metrics[key]
            
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
    # Compute average metrics over all batches
    for key in metrics:
        metrics[key] /= len(loader)

    print(f"{mode}: " + ", ".join(f"{key.capitalize()}: {metrics[key]}" for key in metrics))

    return metrics


def train_model(model, train_loader, val_loader, optimizer, loss_fn, epochs, model_metrics, device='cpu'):
    """
    Runs the full training loop with tracking metrics and checkpoint saving
    
    Returns:
        dict: Training history with losses and metric values.
    """

    history = {f"train_{key}": [] for key in ["loss", *model_metrics]}
    history.update({f"val_{key}": [] for key in ["loss", *model_metrics]})

    for epoch in range(epochs):
        print(f"Epoch: {epoch+1}/{epochs}")

        train_metrics = run_epoch(model, train_loader, loss_fn, model_metrics, optimizer, device=device)
        val_metrics = run_epoch(model, val_loader, loss_fn, model_metrics, device=device)

        # Save metrics for analysis
        for key in history:
            if "train" in key:
                history[key].append(train_metrics[key.replace("train_", "")])
            else:
                history[key].append(val_metrics[key.replace("val_", "")])

        # Save model checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        ensure_dir(CHECKPOINT_DIR)
        torch.save(checkpoint, os.path.join(CHECKPOINT_DIR, f"{model.name}.pth"))

        # Save input, masks and predicted masks images
        if (epoch + 1) % 10 == 0:
            save_model_predictions(model, val_loader, "validation_results", device)

    return history

def save_model_predictions(model, loader, dir_path="validation_results", device="cpu"):
    """Save model predictions along with input images and ground truth masks"""
    ensure_dir(dir_path)
    model.eval()

    for idx, (imgs, masks) in enumerate(loader):
        imgs = imgs.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(imgs))
            preds = (preds > 0.5).float()
        
        for i in range(loader.batch_size):
            torchvision.utils.save_image(imgs[i], os.path.join(dir_path, f"input_image{idx}_{i}.png"))
            torchvision.utils.save_image(preds[i], os.path.join(dir_path, f"{idx}_{i}_pred.png"))
            torchvision.utils.save_image(masks[i], os.path.join(dir_path, f"{idx}_{i}.png"))

    model.train()

def plot_training_history(history, model_metrics):
    """Plot the training and validation history of the model"""
    metrics = ["loss", *model_metrics]
    for i, metric in enumerate(metrics):
        plt.figure(i)
        plt.plot(history[f"train_{metric}"])
        plt.plot(history[f"val_{metric}"])
        plt.title(f"Training {metric}")
        plt.ylabel(metric.capitalize())
        plt.xlabel('Epoch')
        plt.legend(['Training', 'Validation'], loc='upper left')
        plt.grid()
    plt.show()
