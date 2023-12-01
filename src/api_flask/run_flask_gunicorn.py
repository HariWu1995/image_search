from flask import Flask
from src.api_flask.run_flask_command import RunFlaskCommand


def create_app(config_path: str) -> Flask:
    """
    Create the Flask app and return it (without starting).
    Entry point for gunicorn
    """
    command = RunFlaskCommand(config_path)
    return command.run(start=False)


if __name__ == "__main__":
    create_app(config_path="api_config.yml")

