import logging
import yaml

from PIL import Image

from flask import Flask
from flask import request, jsonify
from flask_cors import CORS

from image_searcher import Search
from api.flask_config import FlaskConfig


class RunFlaskCommand:

    logging.basicConfig(level=logging.INFO)

    def __init__(self, config_path: str):
        self.searcher = None
        self.config = FlaskConfig(**yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))

    def search_by_text(self):
        user_query = request.args.get("q")
        logging.info(f"Text query: {user_query}")
        result = self.searcher.rank_results(user_query, n=self.config.n)
        return jsonify(results=result)

    def search_by_image(self):
        image = request.files['image']
        user_query = Image.open(image.stream)
        logging.info(f"Image query")
        result = self.searcher.rank_results(user_query, n=self.config.n)
        return jsonify(results=result)

    def run(self, start=True):
        app = Flask(__name__, static_folder=None)
        self.searcher = Search(image_dir_path=self.config.image_dir_path,
                               traverse=self.config.traverse,
                               save_path=self.config.save_path)
        CORS(app)
        app.add_url_rule(rule="/search_by_text", endpoint="search_by_text", view_func=self.search_by_text, methods=["GET"])
        app.add_url_rule(rule="/search_by_image", endpoint="search_by_image", view_func=self.search_by_image, methods=["POST", "GET"])

        if start:
            app.run(port=self.config.port,
                    host=self.config.host,
                    debug=self.config.debug,
                    threaded=self.config.threaded)
        return app


if __name__ == "__main__":
    command = RunFlaskCommand(config_path="api_config.yml")
    command.run()
