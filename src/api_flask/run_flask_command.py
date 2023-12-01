import logging
import yaml

from PIL import Image

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

from src.modeling.search.search import Search
from src.api_flask.flask_config import FlaskConfig
from src.utils.converters import image2base64


class RunFlaskCommand:

    logging.basicConfig(level=logging.INFO)

    def __init__(self, config_path: str):
        self.searcher = None
        self.config = FlaskConfig(**yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader))

    def format_ranking(self, result):
        for r in result:
            r.image_path = str(image2base64(Image.open(r.image_path)))[2:-1]
        return result

    def search_by_text(self):
        user_query = request.args.get("q")
        logging.info(f"Text query: {user_query}")
        result = self.searcher.rank_results(user_query, n=self.config.n)
        return jsonify(results=self.format_ranking(result))

    def search_by_image(self):
        image = request.files['image']
        user_query = Image.open(image.stream).convert('RGB')
        logging.info(f"Image query")
        result = self.searcher.rank_results(user_query, n=self.config.n)
        return jsonify(results=self.format_ranking(result))

    def ping(self):
        logging.info(f"Ping")
        return jsonify(results={'text': 'h3ll0 w0rld'})

    def kill(self):
        logging.info(f"Terminate")
        quit()

    def run(self, start=True):
        app = Flask(__name__, static_folder=None)
        self.searcher = Search(image_dir_path=self.config.image_dir_path,
                               traverse=self.config.traverse,
                               save_path=self.config.save_path)
        CORS(app)
        app.add_url_rule(rule="/search_by_text", endpoint="search_by_text", view_func=self.search_by_text, methods=["GET"])
        app.add_url_rule(rule="/search_by_image", endpoint="search_by_image", view_func=self.search_by_image, methods=["POST", "GET"])
        app.add_url_rule(rule="/ping", endpoint="ping", view_func=self.ping, methods=["POST", "GET"])
        app.add_url_rule(rule="/kill", endpoint="kill", view_func=self.kill, methods=["POST", "GET"])
        # app.add_url_rule(rule="/text", endpoint="text", view_func=self.text, methods=["POST", "GET"])
        # app.add_url_rule(rule="/image", endpoint="image", view_func=self.image, methods=["POST", "GET"])

        if start:
            app.run(port=self.config.port,
                    host=self.config.host,
                    debug=self.config.debug,
                    threaded=self.config.threaded)
        return app


if __name__ == "__main__":
    command = RunFlaskCommand(config_path="configuration/api_config.yml")
    command.run()
