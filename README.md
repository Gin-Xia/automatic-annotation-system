# Automatic Annotation System

## **Project Overview**
The **Automatic Annotation System** is a deep learning pipeline that integrates **OWL-ViT** and **MobileSAM** to automatically annotate images. It processes images, detects objects using OWL-ViT, and refines segmentation masks with MobileSAM.

## **Features**
- **Batch Inference Support**: Process multiple images in parallel.
- **Text-Guided Object Detection**: OWL-ViT enables open-vocabulary object detection.
- **High-Precision Segmentation**: MobileSAM generates fine-grained masks.
- **COCO Dataset Support**: Uses the COCO dataset for training and evaluation.
- **Visualization Tools**: Visualize results with bounding boxes and segmentation masks.

## **Project Structure**
```
automatic-annotation-system/
│── datasets/               # Dataset management
│   ├── coco/               # COCO dataset structure
│   │   ├── annotations/    # COCO annotations (JSON files)
│   │   ├── train2017/      # Training images
│   │   ├── val2017/        # Validation images
│── models/                 # Model definitions and weights
│   ├── owlvit-base-patch32/  # Pretrained OWL-ViT model files
│   │   ├── config.json
│   │   ├── merges.txt
│   │   ├── model.safetensors
│   │   ├── preprocessor_config.json
│   │   ├── pytorch_model.bin
│   │   ├── special_tokens_map.json
│   │   ├── tokenizer_config.json
│   │   ├── vocab.json
│   ├── mobile_sam.pt       # Pretrained MobileSAM model weights
│   ├── mobilesam_official.py
│   ├── owlvit.py
│   ├── owlvit_official.py
│   ├── text_encoder.py
│   ├── vision_encoder.py
│── .gitignore              # Git ignore file
│── coco_loader.py          # DataLoader for COCO dataset
│── inference.py            # Script for running inference
│── train.py                # Script for training models
│── visualize.py            # Visualization utilities
│── README.md               # Project documentation
```

## **Installation**
### **1. Clone the Repository**
```bash
git clone https://github.com/yourusername/automatic-annotation-system.git
cd automatic-annotation-system
```

### **2. Create and Activate Virtual Environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On macOS/Linux
.venv\Scripts\activate     # On Windows
```

### **3. Install Dependencies**
```bash
pip install -r requirements.txt
```

## **Usage**
### **Running Inference**
To run inference on COCO images using OWL-ViT and MobileSAM:
```bash
python inference.py
```
- The script loads images, detects objects, and generates segmentation masks.
- Outputs include **bounding boxes, segmentation masks, and confidence scores**.

### **Training the Model**
To train a custom model:
```bash
python train.py
```
Modify the `train.py` script to specify dataset paths and training parameters.

## **Configuration**
Modify `coco_loader.py` to specify dataset paths:
```python
dataloader = get_coco_dataloader(root="datasets/coco/train2017",
                                 annotation="datasets/coco/annotations/instances_train2017.json",
                                 batch_size=8, shuffle=True)
```

## **Visualization**
Results can be visualized using `visualize.py`. To display predictions:
```python
from visualize import visualize_results
visualize_results(image_np, best_box, masks, category_name, confidence)
```

## **Dependencies**
- `torch`
- `transformers`
- `numpy`
- `PIL`
- `mobile-sam`
- `torchvision`

## **License**
This project is licensed under the MIT License.


