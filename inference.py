import torch
import numpy as np
import time
import logging
from transformers import OwlViTProcessor
from models.owlvit_official import OwlvitOfficial
from models.mobilesam_official import MobileSAMOfficial
from coco_loader import get_coco_dataloader
from flickr_loader import get_flickr_inference_dataloader
from visualize import visualize_results
import box_processing


# ========== 设置日志 ==========
logging.basicConfig(
    filename="inference.log",
    filemode="w",
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

# ========== 设置设备 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Using device: {device}")
print(f"Using device: {device}")

# ========== 初始化模型 ==========
owlvit_model = OwlvitOfficial().to(device)
processor = OwlViTProcessor.from_pretrained("models/owlvit-base-patch32")
owlvit_model.eval()

mobilesam_model = MobileSAMOfficial(checkpoint_path="models/mobile_sam.pt")

# ========== 加载 COCO 数据集 ==========
dataloader = get_coco_dataloader(
    root="datasets/coco/train2017",
    annotation="datasets/coco/annotations/instances_train2017.json",
    batch_size=1,
    shuffle=False
)

# dataloader = get_flickr_inference_dataloader(
#     image_dir="datasets/flicker30k/flickr30k-images",
#     annotation_dir="datasets/flicker30k/Sentences",
#     batch_size=1,
#     shuffle=False
# )


# ========== 推理配置 ==========
max_images = 4
processed_images = 0

# ========== 开始总计时 ==========
if torch.cuda.is_available():
    torch.cuda.synchronize()
total_start_time = time.time()

# ========== 推理主循环 ==========
for batch_idx, batch in enumerate(dataloader):
    if processed_images >= max_images:
        break

    images, bboxes, category_names = batch
    print("images.size:",images.size())
    category_names = [[category_names[0][0]]]
    images = images.to(device=device, dtype=torch.float32)

    inputs = processor(text=category_names, images=images, return_tensors="pt", do_rescale=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = owlvit_model(inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"])

    logits = outputs[0]
    pred_boxes = outputs[1]
    #
    # logits = logits.squeeze(0).squeeze(-1)
    #
    # print("logits.shape:", logits.shape)
    # logits = logits.sigmoid()
    # # print(logits)
    print(logits.squeeze(0).squeeze(-1).sigmoid().min(), logits.squeeze(0).squeeze(-1).sigmoid().max())
    # # print(logits.sigmoid())

    valid_boxes, valid_scores = box_processing.filter_boxes_by_score(logits, pred_boxes)
    if valid_boxes.shape[0] == 0:
        logging.info("No high-confidence objects detected.")
        print("No high-confidence objects detected.")
        break

    converted_boxes = box_processing.convert_boxes(valid_boxes, images.shape[-2:])
    filtered_boxes, filtered_scores = box_processing.apply_nms(converted_boxes, valid_scores, iou_threshold=0.3)
    final_boxes = box_processing.merge_high_iou_boxes(filtered_boxes, filtered_scores, iou_threshold=0.7)

    log_msg = f"Detected {final_boxes.shape[0]} high-confidence objects after NMS + Merge"
    logging.info(log_msg)
    print(log_msg)

    for img_idx in range(images.shape[0]):
        image_np = images[img_idx].permute(1, 2, 0).cpu().numpy()

        all_masks = []
        all_scores = []

        final_boxes = final_boxes.cpu().numpy().astype(np.float32)
        for final_box in final_boxes:
            final_box = final_box.reshape(1, 4)
            masks, sam_scores = mobilesam_model.predict(image_np, final_box, multimask_output=False)
            all_masks.append(masks)
            all_scores.append(sam_scores)

        all_masks = np.array(all_masks).squeeze()
        if all_masks.ndim == 2:
            all_masks = np.expand_dims(all_masks, axis=0)

        visualize_results(image_np, final_boxes, all_masks, category_names[0], all_scores)

        log_info = f"Processed Image {processed_images + 1}/{max_images} | Category: {category_names[0]} | Objects: {len(final_boxes)}"
        logging.info(log_info)
        print(log_info)
        print("=" * 50)

        processed_images += 1

# ========== 结束计时 ==========
if torch.cuda.is_available():
    torch.cuda.synchronize()
total_end_time = time.time()
total_time = total_end_time - total_start_time

final_msg = f"\n=== Total inference time for {processed_images} images: {total_time:.3f} seconds ==="
logging.info(final_msg)
print(final_msg)

if processed_images > 0:
    avg_time = total_time / processed_images
    avg_msg = f"Average time per image: {avg_time:.3f} seconds"
    logging.info(avg_msg)
    print(avg_msg)

