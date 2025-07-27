import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
from PIL import Image

class HybridTransform:
    def __init__(self, transforms, probs=None):
        self.transforms = transforms
        self.probs = probs or [1.0] * len(transforms)
        
    def __call__(self, img):
        for transform, prob in zip(self.transforms, self.probs):
            if random.random() < prob:
                img = transform(img)
        return img

class CutMix:
    def __init__(self, alpha=1.0, prob=1.0):
        self.alpha = alpha
        self.prob = prob
        
    def __call__(self, images, targets):
        if random.random() > self.prob:
            return images, targets
            
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(images.size(0))
        target_a = targets
        target_b = targets[rand_index]
        
        bbx1, bby1, bbx2, bby2 = self.rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
        
        return images, (target_a, target_b, lam)
    
    def rand_bbox(self, size, lam):
        W = size[2]
        H = size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int32(W * cut_rat)
        cut_h = np.int32(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)

        return bbx1, bby1, bbx2, bby2

class MixUp:
    def __init__(self, alpha=0.8, prob=0.8):
        self.alpha = alpha
        self.prob = prob
        
    def __call__(self, images, targets):
        if random.random() > self.prob:
            return images, targets
            
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(images.size(0))
        mixed_images = lam * images + (1 - lam) * images[rand_index]
        target_a, target_b = targets, targets[rand_index]
        
        return mixed_images, (target_a, target_b, lam)

class RandomErasing:
    def __init__(self, prob=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3)):
        self.prob = prob
        self.scale = scale
        self.ratio = ratio
        
    def __call__(self, img):
        if random.random() > self.prob:
            return img
            
        img = np.array(img)
        h, w, c = img.shape
        area = h * w
        
        for _ in range(10):  # Try max 10 times
            erase_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)
            
            erase_h = int(round(np.sqrt(erase_area * aspect_ratio)))
            erase_w = int(round(np.sqrt(erase_area / aspect_ratio)))
            
            if erase_h < h and erase_w < w:
                x1 = random.randint(0, h - erase_h)
                y1 = random.randint(0, w - erase_w)
                img[x1:x1+erase_h, y1:y1+erase_w] = np.random.randint(
                    0, 255, (erase_h, erase_w, c)
                )
                return Image.fromarray(img)
                
        return Image.fromarray(img)

def get_transforms(dataset, split, img_size):
    """Get dataset-specific transformations"""
    if dataset == "imagenet":
        return imagenet_transforms(split, img_size)
    elif dataset == "coco":
        return coco_transforms(split, img_size)
    elif dataset == "ade20k":
        return ade20k_transforms(split, img_size)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def imagenet_transforms(split, img_size=224):
    """ImageNet transformations as per paper"""
    if split == "train":
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
            T.RandomHorizontalFlip(),
            HybridTransform([
                T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                T.RandomGrayscale(p=0.2),
                T.RandomSolarize(threshold=128, p=0.1),
                T.RandomPosterize(bits=4, p=0.1),
                T.RandomAdjustSharpness(sharpness_factor=2, p=0.1),
            ], probs=[0.8, 0.2, 0.1, 0.1, 0.1]),
            RandomErasing(prob=0.25),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # validation
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def coco_transforms(split, img_size=(800, 1333)):
    """COCO transformations for detection/segmentation"""
    if split == "train":
        return T.Compose([
            T.Resize(img_size),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # validation
        return T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

def ade20k_transforms(split, img_size=512):
    """ADE20K transformations for segmentation"""
    if split == "train":
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.5, 2.0)),
            T.RandomHorizontalFlip(),
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  
        return T.Compose([
            T.Resize(img_size),
            T.CenterCrop(img_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
