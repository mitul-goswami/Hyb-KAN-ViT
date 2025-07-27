from pycocotools.coco import COCO
import torch
import torch.utils.data as data
import os
import cv2
import numpy as np
from PIL import Image
from .datasets import coco_transforms

class COCODataset(data.Dataset):
    def __init__(self, root, ann_file, transform=None, target_size=(800, 1333)):
        self.root = root
        self.coco = COCO(ann_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transform = transform
        self.target_size = target_size
        
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        
        img_info = coco.loadImgs(img_id)[0]
        path = img_info['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        
        num_objs = len(anns)
        boxes = []
        masks = []
        labels = []
        
        for i in range(num_objs):
            x, y, w, h = anns[i]['bbox']
            boxes.append([x, y, x+w, y+h])

            mask = coco.annToMask(anns[i])
            masks.append(mask)
     
            labels.append(anns[i]['category_id'])
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(np.stack(masks), dtype=torch.uint8)
        image_id = torch.tensor([img_id])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd
        }
        
   
        if self.transform:
            img = self.transform(img)
            
        return img, target

def build_coco_dataloader(root, ann_dir, batch_size=16, num_workers=8, 
                          target_size=(800, 1333)):
   
    train_ann = os.path.join(ann_dir, "instances_train2017.json")
    val_ann = os.path.join(ann_dir, "instances_val2017.json")
    
    
    train_transform = coco_transforms("train", target_size)
    val_transform = coco_transforms("val", target_size)
    
    
    train_dataset = COCODataset(
        root=os.path.join(root, "train2017"),
        ann_file=train_ann,
        transform=train_transform,
        target_size=target_size
    )
    
    val_dataset = COCODataset(
        root=os.path.join(root, "val2017"),
        ann_file=val_ann,
        transform=val_transform,
        target_size=target_size
    )
    
    
    def collate_fn(batch):
        images = []
        targets = []
        for img, target in batch:
            images.append(img)
            targets.append(target)
        return torch.stack(images), targets
    
   
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, 
        collate_fn=collate_fn, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader
