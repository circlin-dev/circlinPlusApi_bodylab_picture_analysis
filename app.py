from api import api
from flask import Flask
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)  # For Cross-Domain problem

APP_ROOT = "/home/ubuntu/circlinMembersApi_python/bodylab_picture_analysis"
logging.basicConfig(filename=f'{APP_ROOT}/execution_log.log',
                    filemode='a+',
                    format=' [%(filename)s:%(lineno)s:%(funcName)s()]- %(asctime)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

# For Blueprint api activation.
app.register_blueprint(api, url_prefix="/api")


@app.route('/testing')
def testing():
    return "Hello, Circlin!!!"


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
