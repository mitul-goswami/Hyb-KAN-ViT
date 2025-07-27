import torch
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, random_split
from .datasets import imagenet_transforms, CutMix, MixUp

def build_imagenet_dataloader(root, batch_size=1024, img_size=224, 
                              num_workers=16, val_split=0.1):
   
    train_transform = imagenet_transforms("train", img_size)
    val_transform = imagenet_transforms("val", img_size)
    test_transform = imagenet_transforms("val", img_size)  
    
    train_dataset = ImageNet(root=root, split='train', transform=train_transform)
    test_dataset = ImageNet(root=root, split='val', transform=test_transform)
    
    num_val = int(len(train_dataset) * val_split)
    num_train = len(train_dataset) - num_val
    train_dataset, val_dataset = random_split(
        train_dataset, [num_train, num_val],
        generator=torch.Generator().manual_seed(42)
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    cutmix = CutMix(alpha=1.0, prob=1.0)
    mixup = MixUp(alpha=0.8, prob=0.8)
    
    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        targets = torch.tensor([item[1] for item in batch])
        
        if random.random() < 0.5:
            images, targets = cutmix(images, targets)
        else:
            images, targets = mixup(images, targets)
        
        return images, targets
    
    train_loader.collate_fn = collate_fn
    
    return train_loader, val_loader, test_loader
