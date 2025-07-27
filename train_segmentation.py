import torch
import torch.nn as nn
from data.ade20k import build_ade20k_dataloader
from models import HybKANViT
from models.heads import SegmentationHead
import os
import time
import numpy as np

class UPerNet(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.decode_head = SegmentationHead(384, num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        return self.decode_head(features)

def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = build_ade20k_dataloader(
        root="/path/to/ade20k",
        split='training',
        target_size=512
    )
    
    val_loader = build_ade20k_dataloader(
        root="/path/to/ade20k",
        split='validation',
        target_size=512
    )
    
    backbone = HybKANViT(
        img_size=512,
        patch_size=16,
        in_chans=3,
        num_classes=150,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        hybrid_type=1,
        wavelet_type='dog'
    )
    
    model = UPerNet(backbone, 150)
    model.to(device)
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=6e-5,
        weight_decay=0.01
    )
    
    total_iter = 160000
    warmup_iter = 1500
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda iter: iter / warmup_iter if iter < warmup_iter else (1 - (iter - warmup_iter) / (total_iter - warmup_iter)
    )
    
    iter_count = 0
    best_miou = 0.0
    
    while iter_count < total_iter:
        model.train()
        start_time = time.time()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            iter_count += 1
            
            if iter_count % 100 == 0:
                print(f"Iter [{iter_count}/{total_iter}] Loss: {loss.item():.4f}")
            
            if iter_count >= total_iter:
                break
        
        train_loss /= len(train_loader)
        val_miou = validate(model, val_loader, device)
        
        print(f"Iter [{iter_count}/{total_iter}] Time: {time.time()-start_time:.2f}s "
              f"Train Loss: {train_loss:.4f} Val mIoU: {val_miou:.4f}")
        
        if val_miou > best_miou:
            best_miou = val_miou
            torch.save({
                'iter': iter_count,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'miou': val_miou
            }, "best_segmentation_model.pth")
        
        if iter_count % 5000 == 0:
            torch.save({
                'iter': iter_count,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"segmentation_checkpoint_iter_{iter_count}.pth")

def validate(model, val_loader, device):
    model.eval()
    total_miou = 0.0
    count = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            miou_val = miou(outputs, labels, 150)
            total_miou += miou_val.item()
            count += 1
    
    return total_miou / count

def miou(outputs, labels, num_classes):
    outputs = outputs.argmax(1)
    ious = []
    for cls in range(num_classes):
        pred_inds = (outputs == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum().float()
        union = (pred_inds | target_inds).sum().float()
        iou = (intersection + 1e-8) / (union + 1e-8)
        ious.append(iou)
    return torch.mean(torch.stack(ious))

if __name__ == "__main__":
    main()
