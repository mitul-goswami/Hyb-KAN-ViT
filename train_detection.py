import torch
import torch.nn as nn
from data.coco import build_coco_dataloader
from models import HybKANViT
from models.heads import DetectionHead
from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.backbone_utils import BackboneWithFPN
from utils.metrics import map_metrics
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
    
    train_loader, val_loader = build_coco_dataloader(
        root="/path/to/coco",
        ann_dir="/path/to/coco/annotations",
        batch_size=16,
        target_size=(800, 1333)
    )
    
    backbone = HybKANViT(
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
    )
    
    return_layers = {'blocks': '0'}
    in_channels_list = [384]
    out_channels = 256
    backbone = BackboneWithFPN(backbone, return_layers, in_channels_list, out_channels)
    
    model = MaskRCNN(
        backbone=backbone,
        num_classes=91,
        box_detections_per_img=100,
        box_score_thresh=0.05
    )
    model.roi_heads.box_predictor = DetectionHead(1024, 91)
    model.roi_heads.mask_predictor = DetectionHead(1024, 91)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=0.0001, weight_decay=0.05)
    
    for epoch in range(36):
        model.train()
        start_time = time.time()
        train_loss = 0.0
        
        for i, (images, targets) in enumerate(train_loader):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            
            train_loss += losses.item()
            
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/36] Batch [{i}/{len(train_loader)}] Loss: {losses.item():.4f}")
        
        train_loss /= len(train_loader)
        val_ap = validate(model, val_loader, device)
        
        print(f"Epoch [{epoch+1}/36] Time: {time.time()-start_time:.2f}s "
              f"Train Loss: {train_loss:.4f} Val AP: {val_ap:.4f}")
        
        if (epoch+1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, f"detection_checkpoint_epoch_{epoch+1}.pth")

def validate(model, val_loader, device):
    model.eval()
    aps = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            
            for i, output in enumerate(outputs):
                pred_boxes = output['boxes'].cpu()
                pred_scores = output['scores'].cpu()
                pred_labels = output['labels'].cpu()
                gt_boxes = targets[i]['boxes'].cpu()
                gt_labels = targets[i]['labels'].cpu()
                
                if len(gt_boxes) > 0:
                    ap = map_metrics(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels)
                    aps.append(ap)
    
    return torch.mean(torch.tensor(aps))

if __name__ == "__main__":
    main()
