from global_things.functions import slack_error_notification
from bodylab_picture_analyze import analysis

from flask import Flask, request
from flask_cors import CORS
import json
import logging

app = Flask(__name__)
CORS(app) #For Cross-Domain problem

APP_ROOT="/home/ubuntu/circlinMembersApi_python/bodylab_picture_analysis"
logging.basicConfig(filename=f'{APP_ROOT}/execution_log.log', filemode='a+', format=' [%(filename)s:%(lineno)s:%(funcName)s()]- %(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)

@app.route('/testing')
def testing():
    return "Hello, Circlin!!!"

@app.route('/', methods = ['POST'])
def index():
    print('Accessed to test server.')
    #Input: 이미지 주소, 유저 정보(id)
    req = request.json
    print('Request: ', req)  #req: ImmutableMultiDict

    #파라미터 읽어들이기
    url = req.get('url')   #이미지 주소
    user_id = req.get('id')   #회원 id

    result = analysis(url, id)
    result = json.loads(result)
    result_message = result['message']
    if result_message == 'Too many people.':
        print(result_message)
        slack_error_notification(user_id=user_id, api='/', error_log=f"Error while analyzing original image '{url}': {result_message}")
        return result, 400
    elif result_message == 'Cannot find image.':
        print(result_message)
        slack_error_notification(user_id=user_id, api='/', error_log=f"Error while analyzing original image '{url}': {result_message}")
        return result, 400
    elif result_message == 'No person detected.':
        print(result_message)
        slack_error_notification(user_id=user_id, api='/', error_log=f"Error while analyzing original image '{url}': {result_message}")
        return result, 400
    elif result_message == 'Unacceptable file extension.':
        print(result_message)
        slack_error_notification(user_id=user_id, api='/', error_log=f"Error while analyzing original image '{url}': {result_message}")
        return result, 400
    elif result_message == 'Bad pose. Unable to detect the whole body joints.':
        print(result_message)
        slack_error_notification(user_id=user_id, api='/', error_log=f"Error while analyzing original image '{url}': {result_message}")
        return result, 400
    elif result_message == 'Bad pose. Head or hip width is 0.':
        print(result_message)
        slack_error_notification(user_id=user_id, api='/', error_log=f"Error while analyzing original image '{url}': {result_message}")
        return result, 400
    elif result_message == 'Bad pose. Invalid body length.':
        print(result_message)
        slack_error_notification(user_id=user_id, api='/', error_log=f"Error while analyzing original image '{url}': {result_message}")
        return result, 400
    elif result_message == 'Error while upload output image into S3 for original image.':
        print(result_message)
        slack_error_notification(user_id=user_id, api='/', error_log=f"Error while analyzing original image '{url}': {result_message}")
        return result, 500
    else:
        return result, 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
