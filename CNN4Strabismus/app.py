from flask import Flask

from CNN4Strabismus.blueprints.page  import page_blueprint
from CNN4Strabismus.blueprints.video import video_blueprint
from CNN4Strabismus.blueprints.photo import photo_blueprint

def create_app():
    """
    Create a Flask application using the app factory pattern.

    :return: Flask app
    """
    app = Flask(__name__, instance_relative_config=True)

    app.config.from_object('config.settings')
    app.config.from_pyfile('settings.py', silent=True)

    app.register_blueprint(page_blueprint)
    app.register_blueprint(photo_blueprint)
    app.register_blueprint(video_blueprint)

    print(f'app.url_map = {app.url_map}')

    return app
