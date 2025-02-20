import torch
import json
import numpy as np
from PIL import Image
from transformers import Owlv2Processor
from pycocotools.coco import COCO
from models.owlvit_official import OwlvitOfficial
from models.mobilesam_offical import MobileSAMOfficial

# Load COCO dataset
coco_annotation_path = "annotations/instances_train2017.json"
coco = COCO(coco_annotation_path)

# Load category names
categories = {cat["id"]: cat["name"] for cat in coco.loadCats(coco.getCatIds())}

# Initialize OWL-ViT
owlvit_model = OwlvitOfficial()
processor = Owlv2Processor.from_pretrained("google/owlvit-base-patch32")
owlvit_model.eval()

# Initialize MobileSAM
mobilesam_model = MobileSAMOfficial(checkpoint_path="./mobile_sam.pt")

# Get image and annotations
image_id = coco.getImgIds()[0]  # Pick one image
image_info = coco.loadImgs(image_id)[0]
image_path = f"train2017/{image_info['file_name']}"
image = Image.open(image_path).convert("RGB")

# Get annotations for the image
ann_ids = coco.getAnnIds(imgIds=image_id)
annotations = coco.loadAnns(ann_ids)

# Process each object in the image
for ann in annotations:
    category_name = categories[ann["category_id"]]
    bbox = ann["bbox"]  # [x, y, width, height]

    # Prepare OWL-ViT input
    inputs = processor(text=[category_name], images=image, return_tensors="pt")

    # OWL-ViT inference
    with torch.no_grad():
        outputs = owlvit_model(inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"])

    # Convert OWL-ViT output to bounding box
    pred_boxes = outputs[1]  # Assuming your model returns (logits, pred_boxes)

    # Select the highest confidence box
    best_box = pred_boxes[0].cpu().numpy()

    # Prepare MobileSAM input
    image_np = np.array(image)
    masks, scores = mobilesam_model.predict(image_np, best_box)

    print(f"Category: {category_name}, Confidence: {scores[0]}")

