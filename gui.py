import streamlit as st
import os
import json
import numpy as np
import torch
from PIL import Image
from visualize import visualize_results
from flickr_loader import get_flickr_dataloader
from models.owlvit_official import OwlvitOfficial
from models.mobilesam_official import MobileSAMOfficial
from transformers import OwlViTProcessor
import box_processing

# ========== Model and Processor Initialization ==========
st.session_state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.session_state.owlvit_model = OwlvitOfficial().to(st.session_state.device).eval()
st.session_state.processor = OwlViTProcessor.from_pretrained("models/owlvit-large-patch14")
st.session_state.sam_model = MobileSAMOfficial(checkpoint_path="models/mobile_sam.pt")

# ========== Dataset Loader ==========
dataloader = get_flickr_dataloader(
    image_dir="datasets/flicker30k/flickr30k-images",
    annotation_dir="datasets/flicker30k/Annotations",
    sentence_dir="datasets/flicker30k/Sentences",
    batch_size=1,
    shuffle=False
)

data_iter = iter(dataloader)

# ========== Streamlit Page Config ==========
st.set_page_config(page_title="OWL-ViT Annotation Tool", layout="wide")
st.title("📌 OWL-ViT Auto Annotation Tool + Optional SAM")

# ========== Session State ==========
if 'image_index' not in st.session_state:
    st.session_state.image_index = 0
if 'last_result' not in st.session_state:
    st.session_state.last_result = {}

# ========== Load Current Sample ==========
try:
    images, bboxes, phrases = next(data_iter)
except StopIteration:
    st.warning("🚫 No more images available!")
    st.stop()

image = images[0].to(st.session_state.device)
prompt = str(phrases[0])
inputs = st.session_state.processor(
    text=[prompt],
    images=image,
    return_tensors="pt",
    do_rescale=False,
    truncation=True  # ✅ ensure prompt doesn't exceed max token length
)
inputs = {k: v.to(st.session_state.device) for k, v in inputs.items()}

# ========== Inference ==========
with torch.no_grad():
    logits, pred_boxes = st.session_state.owlvit_model(inputs["pixel_values"], inputs["input_ids"], inputs["attention_mask"])

valid_boxes, valid_scores = box_processing.filter_boxes_by_score(logits, pred_boxes)
converted_boxes = box_processing.convert_boxes(valid_boxes, image.shape[-2:])
filtered_boxes, filtered_scores = box_processing.apply_nms(converted_boxes, valid_scores, iou_threshold=0.3)
final_boxes = box_processing.merge_high_iou_boxes(filtered_boxes, filtered_scores, iou_threshold=0.3)


image_np = image.permute(1, 2, 0).cpu().numpy()
final_boxes = final_boxes.cpu().numpy().astype(np.float32)

# ========== Display Result ==========
st.image(image_np, caption=f"Image ID: {st.session_state.image_index}", channels="RGB")
st.markdown(f"#### Inference Result: {len(final_boxes)} candidate boxes")

# ========== Box Interaction ==========
selected_indices = st.multiselect("Select boxes to keep (by index)", options=list(range(len(final_boxes))), default=list(range(len(final_boxes))))
use_sam = st.checkbox("Enable SAM Mask", value=False)

if use_sam:
    all_masks = []
    for box in final_boxes:
        masks, _ = st.session_state.sam_model.predict(image_np, box.reshape(1, 4), multimask_output=False)
        all_masks.append(masks.squeeze())
    all_masks = np.array(all_masks)
else:
    all_masks = np.zeros((len(final_boxes), image_np.shape[0], image_np.shape[1]), dtype=bool)

visualize_results(image_np, final_boxes, all_masks, [prompt] * len(final_boxes), filtered_scores)

# ========== Save to JSON ==========
if st.button("💾 Save Annotations"):
    results = []
    for idx in selected_indices:
        results.append({
            "bbox": final_boxes[idx].tolist(),
            "phrase": prompt,
            "use_sam": bool(use_sam)
        })
    json_path = f"output_annotations/image_{st.session_state.image_index}.json"
    os.makedirs("output_annotations", exist_ok=True)
    with open(json_path, "w") as f:
        json.dump({
            "file_name": f"{st.session_state.image_index}.jpg",
            "annotations": results
        }, f, indent=2)
    st.success(f"✅ Saved to {json_path}")

if st.button("➡️ Next Image"):
    st.session_state.image_index += 1
    st.rerun()
