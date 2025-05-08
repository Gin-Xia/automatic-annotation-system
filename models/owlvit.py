import torch
import torch.nn as nn
from transformers import ViTModel, CLIPTextModel


class Owlvit(nn.Module):
    def __init__(self):
        super(Owlvit, self).__init__()

        # ViT base-patch32
        self.vision_encoder = ViTModel.from_pretrained('google/vit-base-patch32-224-in21k')

        # CLIP Text Encoder
        self.text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32')

        self.multi_modal_fusion = nn.Linear(self.vision_encoder.config.hidden_size, 512)

        self.detection_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 4 + 1)  # 4 bbox coordinates + 1 class confidence
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        """
            Args:
                pixel_values: Tensor of shape [batch_size, 3, height, width]
                              Input images as pixel tensors
                input_ids: Tensor of shape [batch_size, sequence_length]
                           Token IDs representing input text
                attention_mask: Tensor of shape [batch_size, sequence_length]
                                Attention masks for text input

            Output: Tensor of shape [batch_size, 5]
                Contains [x, y, w, h, confidence]
                The first four values are bounding box coordinates, and the last value is the class confidence or objectness score
        """

        # Encode visual features
        vision_outputs = self.vision_encoder(pixel_values=pixel_values)
        vision_pooled = vision_outputs.pooler_output

        # Encode text features
        text_outputs = self.text_encoder(input_ids=input_ids, attention_mask=attention_mask)
        text_pooled = text_outputs.pooler_output

        # Fuse visual and text features (element-wise multiplication)
        combined_features = vision_pooled * text_pooled

        # Multi-modal fusion layer
        fused_features = self.multi_modal_fusion(combined_features)

        # Detection head predicts bounding box coordinates and confidence
        detection_output = self.detection_head(fused_features)

        return detection_output
