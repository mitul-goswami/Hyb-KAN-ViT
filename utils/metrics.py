import torch

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def map_metrics(pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels, iou_threshold=0.5):
    aps = []
    for iou in [0.5, 0.75]:
        ap = 0.0
        for cls in torch.unique(gt_labels):
            cls_pred_inds = pred_labels == cls
            cls_gt_inds = gt_labels == cls
            if not torch.any(cls_pred_inds) or not torch.any(cls_gt_inds):
                continue
            ious = box_iou(pred_boxes[cls_pred_inds], gt_boxes[cls_gt_inds])
            max_ious, _ = ious.max(1)
            tp = max_ious >= iou_threshold
            fp = ~tp
            tp_cumsum = torch.cumsum(tp, dim=0)
            fp_cumsum = torch.cumsum(fp, dim=0)
            recall = tp_cumsum / (cls_gt_inds.sum() + 1e-8)
            precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
            ap += compute_ap(recall, precision)
        aps.append(ap / len(torch.unique(gt_labels)))
    return aps

def miou(pred, target, num_classes):
    ious = []
    pred = pred.argmax(1)
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = (pred_inds & target_inds).sum()
        union = (pred_inds | target_inds).sum()
        ious.append((intersection + 1e-8) / (union + 1e-8))
    return torch.mean(torch.tensor(ious))

def box_iou(boxes1, boxes2):
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    return inter / (area1[:, None] + area2 - inter)

def compute_ap(recall, precision):
    recall = recall.cpu().numpy()
    precision = precision.cpu().numpy()
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
