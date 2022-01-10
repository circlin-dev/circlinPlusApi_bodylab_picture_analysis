from flask import Flask, request
from flask_cors import CORS
import json
import logging

from global_things.variables import SLACK_NOTIFICATION_WEBHOOK
from global_things.functions import error_notification_slack
from bodylab_picture_analyze import analysis

app = Flask(__name__)
CORS(app) #For Cross-Domain problem

APP_ROOT="/home/ubuntu/circlinMembersApi_python/bodylab_picture_analysis"
logging.basicConfig(filename=f'{APP_ROOT}/execution_log.log', filemode='a+', format=' [%(filename)s:%(lineno)s:%(funcName)s()]- %(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)


#Input(써클인에서 전송하는 파라미터): 이미지(파일 제목), 유저 ID
#Output: 이미지 처리 결과 파일(output_'유저 ID'_'현재 연월일'_'파일명.확장자'), 신체 비율 수치

@app.route('/testing')
def testing():
    return "Hello, Circlin!!!"

#1. 도메인 하나를 얻어, DNS 등록을 시켜야 한다.
@app.route('/', methods = ['POST'])
def index():
    print('Accessed to test server.')
    #Input: 이미지 주소, 유저 정보(id)
    req = request.json
    print('Request: ', req)  #req: ImmutableMultiDict

    #파라미터 읽어들이기
    url = req.get('url')   #이미지 주소
    uid = req.get('uid')   #회원 id

    result = analysis(url, uid)

    if json.loads(result)['message'] == 'Too many people.':
        print(json.loads(result)['message'])
        return result, 400
    elif json.loads(result)['message'] == 'Cannot find image.':
        print(json.loads(result)['message'])
        return result, 400
    elif json.loads(result)['message'] == 'No person detected.':
        print(json.loads(result)['message'])
        return result, 400
    elif json.loads(result)['message'] == 'Unacceptable file extension.':
        print(json.loads(result)['message'])
        return result, 400
    elif json.loads(result)['message'] == 'Bad pose. Unable to detect the whole body joints.':
        print(json.loads(result)['message'])
        return result, 400
    elif json.loads(result)['message'] == 'Bad pose. Head or hip width is 0.':
        print(json.loads(result)['message'])
        return result, 400
    elif json.loads(result)['message'] == 'Bad pose. Invalid body length.':
        print(json.loads(result)['message'])
        return result, 400
    elif json.loads(result)['message'] == 'Image upload error.':
        print(json.loads(result)['message'])
        return result, 500
    else:
        return result, 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
