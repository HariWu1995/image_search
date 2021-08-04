from typing import List, Generator
import os

from PIL import Image


class ImageLoader:
    
    def __init__(self, image_dir_path: str, traverse: bool=False, 
                       exclude_hidden_directories: bool=True):
        self.image_dir_path = image_dir_path
        self.traverse = traverse
        self.exclude_hidden = exclude_hidden_directories
        self.accepted_formats = (".png", ".jpg", ".jpeg")
        self.batch_size = 3

    def search_tree(self):
        image_files = []
        if not self.image_dir_path:
            return image_files

        if self.traverse:
            for root, dirs, files in os.walk(self.image_dir_path):
                if self.exclude_hidden:
                    dirs[:] = [d for d in dirs if d[0]!='.']
                image_files.extend([os.path.join(root, f) 
                                    for f in files if f.lower().endswith(self.accepted_formats)])
        else:
            image_files = [os.path.join(self.image_dir_path, f) for f in os.listdir(self.image_dir_path)
                           if f.lower().endswith(self.accepted_formats)]
        return image_files

    def open_images(self, image_paths: List[str]) -> Generator:
        for idx in range(0, len(image_paths), self.batch_size):
            yield [self.open_image(file) 
                    for file in image_paths[idx:min(idx+self.batch_size, len(image_paths))]]

    @staticmethod
    def open_image(image_path: str) -> Image.Image:
        return Image.open(image_path).convert('RGB')
