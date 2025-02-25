from .base_model import BaseModel
from transformers import AlignProcessor, AlignModel
import torch
from PIL import Image

class ALIGNModel():
    def __init__(self, gpu):
        self.img_size = 224
        self.model = AlignModel.from_pretrained("kakaobrain/align-base").to(gpu)
        self.processor = AlignProcessor.from_pretrained("kakaobrain/align-base")

    def score(self, images, texts):
        inputs = self.processor(images=images, text=texts, return_tensors="pt")
        result = self.model(**inputs)
        return result.logits_per_image