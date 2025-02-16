import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from utils import *

class MoNuSegDataset(Dataset):
    """
    MoNuSeg dataset for nuclei segmentation in histopathological images.
    """

    def __init__(self, dir_path, image_names, patch_size, size=None, normalizer=None, transform=None):
        """
        Initializes dataset, handles loading images and masks, applying transformations, and patchifying.
        
        Args:
            dir_path (str): Directory containing images and masks.
            image_names (list): List of image filenames.
            patch_size (int): Size of the patches.
            size (int, optional): Resize images and masks to this size. Default is None.
            normalizer (object, optional): Normalizer instance for stain normalization. Default is None.
            transform (callable, optional): Data augmentation transformations. Default is None.
        """
    
        images = []
        masks= []
        for i in range(len(image_names)):
            img_path = os.path.join(f'{dir_path}/img', image_names[i])
            image = transforms.ToTensor()(Image.open(img_path).convert("RGB"))
            
            mask_path = os.path.join(f'{dir_path}/mask', image_names[i].replace(".tif", ".png"))
            mask = transforms.ToTensor()(Image.open(mask_path).convert("L"))
            # mask = mask / 255.0
            if size:
                image = transforms.functional.resize(image, [size, size])
                mask = transforms.functional.resize(mask, [size, size])

            if normalizer:
                 # Apply stain normalization if a normalizer is provided
                image, H, E = normalizer.normalize(I=image * 255.0, stains=True)
                image = image.permute(2, 0, 1).float() / 255.0
                
            images.append(image)
            masks.append(mask)
        
        # Patchify images and masks
        self.images = torch.cat([patchify(tensor, patch_size, patch_size) for tensor in images], 0)
        self.masks = torch.cat([patchify(tensor, patch_size, patch_size) for tensor in masks], 0)
        
        self.transform = transform
     
    def __getitem__(self, index):
        image = self.images[index]
        mask = self.masks[index]

        if self.transform is not None:
             # Apply transformations (data augmentation)
            augumentations = self.transform(image=image.permute(1, 2, 0).numpy(), mask=mask.permute(1, 2, 0).numpy())
            image = torch.tensor(augumentations["image"]).permute(2, 0, 1)
            mask = torch.tensor(augumentations["mask"]).permute(2, 0, 1)

        return image, mask
    
    def __len__(self):
        return self.images.shape[0]