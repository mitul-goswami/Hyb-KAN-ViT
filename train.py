import torch
import torch.nn as nn
import torch.optim as optim
from data.datasets import build_imagenet_dataloader
from models import HybKANViT
from utils.train_utils import ExponentialMovingAverage, CosineLRScheduler, clip_gradients
from utils.metrics import accuracy
import os
import time
import numpy as np

def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, _ = build_imagenet_dataloader(
        root="/path/to/imagenet",
        batch_size=1024,
        img_size=224,
        num_workers=16
    )
    
    model = HybKANViT(
        img_size=224,
        patch_size=16,
        in_chans=3,
        num_classes=1000,
        embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        hybrid_type=1,
        wavelet_type='dog'
    ).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=5e-4,
        weight_decay=0.05,
        betas=(0.9, 0.98),
        eps=1e-8
    )
    
    ema = ExponentialMovingAverage(model, decay=0.9998)
    scheduler = CosineLRScheduler(optimizer, warmup_epochs=10, total_epochs=300)
    
    scaler = torch.cuda.amp.GradScaler()
    
    for epoch in range(300):
        model.train()
        start_time = time.time()
        train_loss = 0.0
        top1_acc = 0.0
        top5_acc = 0.0
        
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            clip_gradients(model, 1.0)
            scaler.step(optimizer)
            scaler.update()
            ema.update()
            
            train_loss += loss.item()
            top1, top5 = accuracy(outputs, targets, topk=(1, 5))
            top1_acc += top1.item()
            top5_acc += top5.item()
            
            if i % 100 == 0:
                print(f"Epoch [{epoch+1}/300] Batch [{i}/{len(train_loader)}] Loss: {loss.item():.4f}")
        
        scheduler.step(epoch)
        
        train_loss /= len(train_loader)
        top1_acc /= len(train_loader)
        top5_acc /= len(train_loader)
        
        val_top1, val_top5 = validate(model, val_loader, device)
        
        print(f"Epoch [{epoch+1}/300] Time: {time.time()-start_time:.2f}s "
              f"Train Loss: {train_loss:.4f} Top1: {top1_acc:.2f}% Top5: {top5_acc:.2f}% "
              f"Val Top1: {val_top1:.2f}% Val Top5: {val_top5:.2f}%")
        
        if (epoch+1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"checkpoint_epoch_{epoch+1}.pth")

def validate(model, val_loader, device):
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, top1_preds = outputs.max(1)
            _, top5_preds = outputs.topk(5, 1, True, True)
            
            top1_correct += top1_preds.eq(labels).sum().item()
            top5_correct += top5_preds.eq(labels.unsqueeze(1)).sum().item()
            total += labels.size(0)
    
    top1_acc = 100. * top1_correct / total
    top5_acc = 100. * top5_correct / total
    return top1_acc, top5_acc

if __name__ == "__main__":
    main()
