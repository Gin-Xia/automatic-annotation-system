import torch
import torch.nn as nn
from transformers import Owlv2ForObjectDetection


class OwlvitOfficial(nn.Module):
    def __init__(self, pretrained_model_name="google/owlvit-base-patch32"):
        super(OwlvitOfficial, self).__init__()

        # Load the official OWL-ViT model
        self.model = Owlv2ForObjectDetection.from_pretrained(pretrained_model_name)

    def forward(self, pixel_values, input_ids, attention_mask):
        """
        Args:
            pixel_values: Tensor of shape [batch_size, 3, height, width]
            input_ids: Tensor of shape [batch_size, sequence_length]
            attention_mask: Tensor of shape [batch_size, sequence_length]

        Output:
            logits: Tensor [batch_size, num_queries, num_classes]
            pred_boxes: Tensor [batch_size, num_queries, 4]
        """
        outputs = self.model(pixel_values=pixel_values,
                             input_ids=input_ids,
                             attention_mask=attention_mask)

        logits = outputs.logits
        pred_boxes = outputs.pred_boxes

        return logits, pred_boxes
