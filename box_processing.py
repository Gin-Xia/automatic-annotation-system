import torch
import torchvision.ops as ops
import matplotlib.pyplot as plt



# def compute_elbow_threshold(scores):
#     if scores.numel() < 2:
#         print("Too few scores to compute elbow.")
#         return 0.0
#
#     sorted_scores, _ = torch.sort(scores, descending=True)
#     diffs = sorted_scores[:-1] - sorted_scores[1:]
#     print("diffs:", diffs)
#
#     if diffs.numel() == 0:
#         print("No score differences to compute elbow.")
#         return 0.0
#
#     max_gap_idx = torch.argmax(diffs)
#     return sorted_scores[max_gap_idx].item()

def compute_elbow_threshold_from_logits(logits, apply_sigmoid=True):
    """
    使用 logits 计算 elbow 阈值，可选是否返回 sigmoid(score)
    """
    if logits.numel() < 2:
        print("Too few logits to compute elbow.")
        return 0.0

    print("logits.shape:", logits.shape)
    sorted_scores, _ = torch.sort(logits, descending=True)
    logits = logits.view(-1)
    sorted_logits, _ = torch.sort(logits, descending=True)
    diffs = sorted_logits[:-1] - sorted_logits[1:]
    # print("diffs:", diffs)

    if diffs.numel() == 0:
        print("No score differences to compute elbow.")
        return 0.0

    elbow_idx = torch.argmax(diffs)
    elbow_logit = sorted_logits[elbow_idx].item()

    print("elbow_logit:", elbow_logit)
    print("elbow_logit_sigmoid:", torch.sigmoid(torch.tensor(elbow_logit)).item())
    return torch.sigmoid(torch.tensor(elbow_logit)).item() if apply_sigmoid else elbow_logit




def filter_boxes_by_score(logits, pred_boxes, min_threshold=0.007, show_plot=True):
    """
    过滤低置信度目标，并动态计算阈值
    参数:
        logits (Tensor): 目标检测输出的 logits，形状为 [num_boxes, 1]
        pred_boxes (Tensor): 预测的边界框，形状为 [num_boxes, 4]
        min_threshold (float): 最低的置信度阈值，避免全被过滤

    返回:
        valid_boxes (Tensor): 过滤后的框
        valid_scores (Tensor): 过滤后的置信度分数
    """
    scores = logits.squeeze(-1).sigmoid()  # 转换为概率
    max_score = scores.max().item()

    # 动态设置 threshold
    if max_score > 0.01:
        threshold = max(min_threshold, scores.mean() - scores.std())
    else:
        print("Scores too low, applying logits normalization...")
        scores = (logits - logits.mean()).squeeze(-1).sigmoid()
        threshold = max(min_threshold, scores.mean() - scores.std())

    print(f"Using threshold: {threshold:.4f}")

    # 可视化分数分布 + 阈值线
    # if show_plot:
    #     plt.figure(figsize=(6, 4))
    #     plt.hist(scores.cpu().flatten().numpy(), bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    #     plt.axvline(threshold, color='red', linestyle='--', label=f"Threshold = {threshold:.4f}")
    #     plt.xlabel("Score")
    #     plt.ylabel("Count")
    #     plt.title("Distribution of OWL-ViT Scores")
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()

    if show_plot:
        flat_scores = scores.cpu().flatten().numpy()

        # 额外计算 quantile 和 elbow
        quantile_thresh = scores.quantile(0.85).item()
        elbow_thresh = compute_elbow_threshold_from_logits(logits, apply_sigmoid=True)

        plt.figure(figsize=(6, 4))
        plt.hist(flat_scores, bins=50, alpha=0.7, color='skyblue', edgecolor='black')

        plt.axvline(threshold, color='red', linestyle='--', label=f"Mean - Std = {threshold:.4f}")
        plt.axvline(quantile_thresh, color='green', linestyle='-.', label=f"85% Quantile = {quantile_thresh:.4f}")
        plt.axvline(elbow_thresh, color='orange', linestyle=':', label=f"Elbow = {elbow_thresh:.4f}")

        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.title("Distribution of OWL-ViT Scores")
        plt.legend()
        plt.tight_layout()
        plt.show()

    valid_indices = scores > threshold  # 过滤低置信度框
    valid_boxes = pred_boxes[valid_indices]  # 过滤框
    valid_scores = scores[valid_indices]  # 过滤置信度分数

    return valid_boxes, valid_scores


def convert_boxes(pred_boxes, image_shape):
    """将归一化的坐标转换为像素坐标"""
    H, W = image_shape
    x_center, y_center, w, h = pred_boxes[:, 0], pred_boxes[:, 1], pred_boxes[:, 2], pred_boxes[:, 3]

    x_min = (x_center - w / 2) * W
    y_min = (y_center - h / 2) * H
    x_max = (x_center + w / 2) * W
    y_max = (y_center + h / 2) * H

    return torch.stack([x_min.clamp(0, W), y_min.clamp(0, H), x_max.clamp(0, W), y_max.clamp(0, H)], dim=1)

def apply_nms(boxes, scores, iou_threshold=0.3):
    """NMS 过滤"""
    keep_indices = ops.nms(boxes, scores, iou_threshold)
    return boxes[keep_indices], scores[keep_indices]

def compute_iou(box1, box2):
    """计算 IOU"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0

def merge_high_iou_boxes(boxes, scores, iou_threshold=0.7):
    """合并高 IOU 框"""
    merged = []
    used = set()

    for i in range(len(boxes)):
        if i in used:
            continue
        overlapping = [(boxes[i], scores[i])]
        used.add(i)

        for j in range(i + 1, len(boxes)):
            if j in used:
                continue
            if compute_iou(boxes[i], boxes[j]) > iou_threshold:
                overlapping.append((boxes[j], scores[j]))
                used.add(j)

        total_score = sum(score for _, score in overlapping)
        merged_box = sum(box * (score / total_score) for box, score in overlapping)
        merged.append(merged_box)

    return torch.stack(merged)
