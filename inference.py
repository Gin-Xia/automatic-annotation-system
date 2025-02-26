import torch
import numpy as np
from PIL import Image
from transformers import OwlViTProcessor
from models.owlvit_official import OwlvitOfficial
from models.mobilesam_official import MobileSAMOfficial
from coco_loader import get_coco_dataloader  # Import the COCO DataLoader
from visualize import visualize_results

# Initialize models
owlvit_model = OwlvitOfficial()
processor = OwlViTProcessor.from_pretrained("models/owlvit-base-patch32")
owlvit_model.eval()

mobilesam_model = MobileSAMOfficial(checkpoint_path="models/mobile_sam.pt")

# Load COCO DataLoader
dataloader = get_coco_dataloader(root="datasets/coco/train2017",
                                 annotation="datasets/coco/annotations/instances_train2017.json",
                                 batch_size=1, shuffle=False)

# Limit the number of inference images
max_images = 10  # Only process the first 10 images
processed_images = 0

# Inference loop
for batch_idx, batch in enumerate(dataloader):
    if processed_images >= max_images:
        break

    images, bboxes, category_names = batch  # Unpack batch
    images = images.to(torch.float32)
    inputs = processor(text=category_names, images=images, return_tensors="pt", do_rescale=False)  # 直接传入 Tensor

    with torch.no_grad():
        outputs = owlvit_model(inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"])

    pred_boxes = outputs[1]  # Assuming model returns (logits, pred_boxes)

    # 获取图像尺寸
    image_height, image_width = images.shape[-2], images.shape[-1]

    # 归一化 bbox 转换为像素坐标 (x, y, w, h)
    pred_boxes[:, :, 0] *= image_width  # x
    pred_boxes[:, :, 1] *= image_height  # y
    pred_boxes[:, :, 2] *= image_width  # w
    pred_boxes[:, :, 3] *= image_height  # h

    # 确保 pred_boxes 形状一致 (B, 4)
    if pred_boxes.dim() == 3:  # 处理 batch 维度
        pred_boxes = pred_boxes[:, 0, :]  # 只取 top-1 预测框

    pred_boxes = pred_boxes.cpu().numpy()  # 转换为 NumPy
    # print("outputs: ",outputs[1])

    for img_idx in range(len(category_names)):  # Process each image in batch
        best_box = pred_boxes[img_idx]  # 取第 img_idx 个 bbox
        image_np = images[img_idx].permute(1, 2, 0).cpu().numpy()  # Convert [C, H, W] → [H, W, C]

        # 确保 bbox 形状为 [4,]
        best_box = np.array(best_box).reshape(-1, 4)

        masks, scores = mobilesam_model.predict(image_np, best_box)

        # 计算全局图片索引
        image_number = batch_idx * dataloader.batch_size + img_idx + 1
        print(f"Image {image_number}: Category: {category_names[img_idx]}, Confidence: {scores[0]}")
        visualize_results(image_np, best_box, masks, category_names[img_idx], scores[0])

    processed_images += len(category_names)


