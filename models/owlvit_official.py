import torch.nn as nn
from transformers import OwlViTForObjectDetection


class OwlvitOfficial(nn.Module):
    def __init__(self, pretrained_model_name="models/owlvit-large-patch14"):
        super(OwlvitOfficial, self).__init__()

        # Load the official OWL-ViT model
        self.model = OwlViTForObjectDetection.from_pretrained(pretrained_model_name)

    def forward(self, pixel_values, input_ids, attention_mask):
        """
        Args:
            pixel_values: Tensor of shape [batch_size, 3, height, width]
            input_ids: Tensor of shape [batch_size, sequence_length]
            attention_mask: Tensor of shape [batch_size, sequence_length]

        Output:
            logits: Tensor [batch_size, num_queries, num_classes]
            pred_boxes: Tensor [batch_size, num_queries, 4], normalized [x_center, y_center, w, h]
        """
        outputs = self.model(pixel_values=pixel_values,
                             input_ids=input_ids,
                             attention_mask=attention_mask)

        logits = outputs.logits
        pred_boxes = outputs.pred_boxes

        return logits, pred_boxes

#
# Try fine-tuning
# class OwlvitOfficial(nn.Module):
#     def __init__(self, pretrained_model_name="models/owlvit-large-patch14"):
#         super(OwlvitOfficial, self).__init__()
#
#         self.model = OwlViTForObjectDetection.from_pretrained(pretrained_model_name)
#
#         self.logits_mlp = None
#         self.box_mlp = nn.Sequential(
#             nn.Linear(4, 4),
#             nn.ReLU(),
#             nn.Linear(4, 4)
#         )
#
#     def forward(self, pixel_values, input_ids, attention_mask):
#         outputs = self.model(pixel_values=pixel_values,
#                              input_ids=input_ids,
#                              attention_mask=attention_mask)
#
#         logits = outputs.logits  # [batch_size, num_queries, num_classes]
#         pred_boxes = outputs.pred_boxes  # [batch_size, num_queries, 4]
#
#         # lazy build logits_mlp
#         if self.logits_mlp is None:
#             num_classes = logits.shape[-1]
#             self.logits_mlp = nn.Sequential(
#                 nn.Linear(num_classes, num_classes),
#                 nn.ReLU(),
#                 nn.Linear(num_classes, num_classes)
#             ).to(logits.device)
#
#         refined_logits = self.logits_mlp(logits)
#         refined_pred_boxes = self.box_mlp(pred_boxes)
#
#         return refined_logits, refined_pred_boxes
