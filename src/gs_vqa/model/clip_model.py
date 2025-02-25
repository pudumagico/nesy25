from .base_model import BaseModel
from transformers import CLIPModel as TCLIPModel, CLIPImageProcessor, CLIPTokenizer
import open_clip
import torch
from PIL import Image

class CLIPModel(BaseModel):
    def __init__(self, gpu, model="openai/clip-vit-base-patch32", snapshot=None, use_open=False, image_kwargs={}):
        super().__init__(img_size=224, gpu=gpu)
        self.use_open = use_open

        if use_open:
            self.model, _, self.image_processor = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
            self.model.eval().to(gpu)
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        else:
            self.model = TCLIPModel.from_pretrained(snapshot if snapshot is not None else model).to(gpu)
            self.image_processor = CLIPImageProcessor.from_pretrained(model, **image_kwargs)
            self.tokenizer = CLIPTokenizer.from_pretrained(model)

    def preprocess_images(self, images):
        return self.image_processor(images, return_tensors="pt", do_resize=False, do_center_crop=False).to(self.gpu)

    def preprocess_texts(self, texts):
        return self.tokenizer(texts, return_tensors="pt", padding=True).to(self.gpu)

    def score(self, images, texts):
        if self.use_open:
            with torch.no_grad():
                processed_images = []
                for image in images:
                    self.image_processor(Image.fromarray(image.permute(1, 2, 0).cpu().numpy())).to(self.gpu)
                    processed_images.append(image)
                processed_images = torch.stack(processed_images).to(self.gpu).type(torch.float32)
                text = self.tokenizer(texts).to(self.gpu)
                image_features = self.model.encode_image(processed_images)
                text_features = self.model.encode_text(text)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                return (100.0 * image_features @ text_features.T).softmax(dim=-1)

        image_inputs = self.preprocess_images(images)
        text_inputs = self.preprocess_texts(texts)
        
        result = self.model(**image_inputs, **text_inputs)
        return result["logits_per_image"]
    
    def get_image_features(self, images):
        image_inputs = self.preprocess_images(images)
        return self.model.get_image_features(**image_inputs)
    
    def get_text_features(self, texts):
        text_inputs = self.preprocess_texts(texts)
        return self.model.get_text_features(**text_inputs)
        