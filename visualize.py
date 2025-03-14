import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

def visualize_results(image, boxes, masks, category, scores):
    """
    可视化 MobileSAM 结果：
    - 在原始图像上绘制所有检测框
    - 叠加所有分割掩码（使用不同颜色）

    参数：
    - image: numpy 数组，形状 (H, W, C)，应为 uint8 格式
    - boxes: numpy 数组，形状 (N, 4)，每行 [x_min, y_min, x_max, y_max]，像素坐标
    - masks: numpy 数组，形状 (N, H, W)，二值掩码 (0 or 1)
    - category: str，类别名称
    - scores: numpy 数组，形状 (N,)，每个框的置信度
    """
    image_vis = image.copy()
    image_vis = (image_vis * 255).astype(np.uint8)

    # 画检测框
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = map(int, box)
        cv2.rectangle(image_vis, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        cv2.putText(image_vis, f"{category} {float(scores[i]):.2f}",
                    (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    # 创建彩色 mask 叠加层
    color_mask = np.zeros_like(image_vis, dtype=np.uint8)

    # 颜色选择方案：使用随机颜色 or 预定义 colormap
    cmap = plt.get_cmap("tab10")  # 可以换成 "jet", "viridis" 等
    num_masks = len(masks)

    for i, mask in enumerate(masks):
        color = np.array(cmap(i / num_masks)[:3]) * 255  # 获取 RGB 颜色并转换为 0-255
        color = color.astype(np.uint8)

        # 叠加 mask
        color_mask[mask == 1] = color

    # 透明叠加
    alpha = 0.5
    image_overlay = cv2.addWeighted(image_vis, 1 - alpha, color_mask, alpha, 0)

    # 显示最终可视化图
    plt.figure(figsize=(8, 8))
    plt.imshow(image_overlay)
    plt.axis("off")
    plt.title(f"{category} - All Detections")
    plt.show()
