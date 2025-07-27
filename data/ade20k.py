import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from .datasets import ade20k_transforms

class ADE20KDataset(Dataset):
   
    def __init__(self, root, split='training', transform=None, target_size=512):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_size = target_size
  
        self.image_dir = os.path.join(root, 'images', split)
        self.mask_dir = os.path.join(root, 'annotations', split)
        
        self.files = []
        for filename in os.listdir(self.image_dir):
            if filename.endswith('.jpg'):
                basename = filename[:-4]
                self.files.append({
                    'image': os.path.join(self.image_dir, filename),
                    'label': os.path.join(self.mask_dir, basename + '.png')
                })
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        datafiles = self.files[index]
        
        image = Image.open(datafiles['image']).convert('RGB')
        
        label = Image.open(datafiles['label'])
        label = np.array(label).astype(np.int32)
        label[label == 0] = 255  
        label -= 1  
        label[label == 254] = 255 
        label = Image.fromarray(label.astype(np.uint8))
        
        if self.transform:
            image = self.transform(image)
            label = self.transform(label) 
        
        
        label = torch.from_numpy(np.array(label)).long()
        
        return image, label

def build_ade20k_dataloader(root, batch_size=16, num_workers=8, 
                            target_size=512, split='training'):
    
    transform = ade20k_transforms(split, target_size)
                              
    dataset = ADE20KDataset(
        root=root,
        split=split,
        transform=transform,
        target_size=target_size
    )
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size,
        shuffle=(split == 'training'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'training')
    )
    
    return loader
