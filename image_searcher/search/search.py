# pylint:disable=no-member
import logging
from tqdm import tqdm
from PIL import Image
from typing import List
from inputimeout import inputimeout, TimeoutOccurred

import torch

from image_searcher.interfaces.image_loader import ImageLoader
from image_searcher.interfaces.result_interface import RankedImage
from image_searcher.interfaces.stored_embeddings import StoredEmbeddings
from image_searcher.encoders.clip_encoder import ClipEncoder


class Search:

    def __init__(self, image_dir_path: str=None, 
                       traverse: bool=False, 
                       save_path: str=None):

        assert (image_dir_path is not None or save_path is not None), \
            "At least one of the paths (image and save path) needs to be specified"
        self.loader = ImageLoader(image_dir_path=image_dir_path, traverse=traverse)

        logging.info("Loading CLIP Encoder")
        self.encoder = ClipEncoder()

        logging.info("Loading pre-computed embeddings")
        self.stored_embeddings = StoredEmbeddings(save_path=save_path if save_path else image_dir_path)
        logging.info(f"{len(self.stored_embeddings.get_image_paths())} files are indexed.")

        logging.info(f"Re-indexing the image files in {image_dir_path}")
        self.reindex()

        if image_dir_path:
            logging.info(f"Excluding files from search if not in {image_dir_path}")
            self.stored_embeddings.exclude_extra_files(filter_path=image_dir_path)

        logging.info(f"Setup over, Searcher is ready to be queried")

    def reindex(self):
        waiting_list = set(self.loader.search_tree()) - set(self.stored_embeddings.get_image_paths())
        if not waiting_list:
            return

        for idx, image_path in enumerate(tqdm(waiting_list)):
            try:
                images = [self.loader.open_image(image_path)]
                self.stored_embeddings.add_embedding(image_path, self.encoder.encode_images(images))
                if idx % 1000 == 0:
                    self.stored_embeddings.update_file()
            except Exception as e:
                logging.warning(e)
                logging.warning(f"Image {image_path} has failed to process - adding it to fail list.")
                try:
                    option = inputimeout(prompt='Do you want to store this value as 0s? [y/N]', 
                                         timeout=5)
                except TimeoutOccurred:
                    option = 'n'
                if option.lower() == 'y':
                    self.stored_embeddings.add_embedding(image_path, torch.zeros((1, 512)))

        self.stored_embeddings.update_file()

    def rank_results(self, query, n: int=10) -> List[RankedImage]:
        if isinstance(query, str):
            query_embeds = self.encoder.encode_text(query)
        else:
            query_embeds = self.encoder.encode_images([query])
        image_embeds = self.stored_embeddings.get_embedding_tensor()
        image_paths = self.stored_embeddings.get_image_paths()

        scores = torch.matmul(query_embeds, image_embeds.t()) * 100
        scores = scores.softmax(dim=1).squeeze().numpy().astype(float)
        rankings = sorted(list(zip(image_paths, scores)), key=lambda x: x[1], reverse=True)[:n]

        return [RankedImage(image_path=path, score=score) for path, score in rankings]
