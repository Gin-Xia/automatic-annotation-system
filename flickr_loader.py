from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import re
from flickr30k_entities_utils import get_sentence_data  # 你已经有了这个函数

def clean_sentence(sentence):
    """移除 [man/ph1/person] 这样的标注，得到干净句子"""
    return re.sub(r"\[([^\[]+?)/ph\d+/.+?\]", r"\1", sentence)

class Flickr30kInferenceDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_ids = [os.path.splitext(f)[0] for f in os.listdir(image_dir) if f.endswith('.jpg')]
        self.transform = transform or transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_dir, image_id + ".jpg")
        txt_path = os.path.join(self.annotation_dir, image_id + ".txt")

        # 加载图片
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        # 只取第一句话 + 清理掉标注
        try:
            sentences = get_sentence_data(txt_path)
            raw_sentence = sentences[0]["sentence"]  # 原始带 [man/ph1/person] 的句子
            prompt = clean_sentence(raw_sentence)    # 转为自然语言 prompt
        except:
            prompt = "a person"  # fallback
        return image_tensor, prompt, image_id  # image_id 用于保存/可视化命名


def get_flickr_inference_dataloader(image_dir, annotation_dir, batch_size=1, shuffle=False):
    dataset = Flickr30kInferenceDataset(image_dir, annotation_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
