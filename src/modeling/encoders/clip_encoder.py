import os

from PIL import Image
from typing import List

import torch
from transformers import CLIPProcessor, CLIPModel


class ClipEncoder:

    def __init__(self, model_path: str = None):
        
        # Set default for model_name, and overwrite if any
        self.model_name = "openai/clip-vit-base-patch32"
        if model_path is not None:
            model_config = model_path + '/config.json'
            if os.path.isfile(model_config):
                self.model_name = model_path

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(self.model_name)

    def encode_images(self, images: List[Image.Image]):
        with torch.no_grad():
            inputs = self.processor(text=None, images=images, return_tensors="pt", padding=True)
            image_embeds = self.encoder.get_image_features(**inputs.to(self.device)).to("cpu")
            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
        return image_embeds

    def encode_text(self, text: str):
        """
        Encode the query in a 512 dim vector.
            :param text: The search query - A single text string
            :return: The feature vector
        """
        with torch.no_grad():
            inputs = self.processor(text=text, images=None, return_tensors="pt", padding=True)
            text_embeds = self.encoder.get_text_features(**inputs.to(self.device)).to("cpu")
            text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        return text_embeds



