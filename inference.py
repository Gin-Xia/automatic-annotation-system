import torch
import numpy as np
from PIL import Image
from transformers import OwlViTProcessor
from models.owlvit_official import OwlvitOfficial
from models.mobilesam_official import MobileSAMOfficial
from coco_loader import get_coco_dataloader  # Import the COCO DataLoader
from visualize import visualize_results
import torchvision.ops as ops
from sklearn.cluster import KMeans
import box_processing


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
    # only detect one category when inference
    category_names = [[category_names[0][0]]]
    images = images.to(torch.float32)
    inputs = processor(text=category_names, images=images, return_tensors="pt", do_rescale=False)  # 直接传入 Tensor

    with torch.no_grad():
        outputs = owlvit_model(inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"])


    # out put of owlvit (logits, pred_boxes)
    logits = outputs[0]
    # print("logits shape:", logits.shape)
    # print("Max logits:", logits.max())
    # print("Min logits:", logits.min())
    pred_boxes = outputs[1]

    valid_boxes, valid_scores = box_processing.filter_boxes_by_score(logits, pred_boxes)
    if valid_boxes.shape[0] == 0:
        print("No high-confidence objects detected.")
        break

    print(f"Detected {valid_boxes.shape[0]} high-confidence objects.")

    converted_boxes = box_processing.convert_boxes(valid_boxes, images.shape[-2:])
    filtered_boxes, filtered_scores = box_processing.apply_nms(converted_boxes, valid_scores, iou_threshold=0.3)
    final_boxes = box_processing.merge_high_iou_boxes(filtered_boxes, filtered_scores, iou_threshold=0.7)
    print(f"Final number of boxes after merging: {final_boxes.shape[0]}")

    # For all image in batch
    for img_idx in range(images.shape[0]):
        image_np = images[img_idx].permute(1, 2, 0).cpu().numpy()  # [C, H, W] → [H, W, C]

        all_masks = []  # 存储所有掩码
        all_scores = []  # 存储所有置信度
        # For all final_box in the same image
        final_boxes = final_boxes.cpu().numpy().astype(np.float32)
        for final_box in final_boxes:
            final_box = final_box.reshape(1, 4)
            masks, sam_scores = mobilesam_model.predict(image_np, final_box, multimask_output=False)
            all_masks.append(masks)
            all_scores.append(sam_scores)

        all_masks = np.array(all_masks).squeeze()  # 变成 (N, H, W)
        # 保证 shape 始终是 (N, H, W)
        if all_masks.ndim == 2:  # 如果只有一个 mask，扩展维度
            all_masks = np.expand_dims(all_masks, axis=0)

        visualize_results(image_np, final_boxes, all_masks, category_names[0], all_scores)

        print(f"Processed Image {processed_images + 1}/{max_images}")
        print(f"Category: {category_names[0]}")
        print(f"Detected Objects: {len(final_boxes)}")
        print("=" * 50)
        processed_images += 1

