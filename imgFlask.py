from flask import Flask, request, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import json
import ftplib
import time
import datetime
import cv2
import numpy as np
from OpenSSL import SSL
from urllib.request import urlopen

import noonBody_Start as nbs

app = Flask(__name__)
CORS(app)   #For Cross-Domain problem

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
    req = request.form
    print('Request: ', req)  #req: ImmutableMultiDict

    #파라미터 읽어들이기
    url = req.get('url')   #이미지 주소
    uid = req.get('uid')   #회원 id
    feedPk = req.get('feedPk') #feedPK

    result = nbs.analysis(url, uid, feedPk)

    if json.loads(result)['message'] == 'Too many people.':
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
    else:
        return result, 200

def isFileExists(file, out, flag):
    #squared4_ 파일이 없으면 -1을 반대 방향에 감하여 영상 제작.
    if os.path.getsize(out) == 0 and flag == 'w1':
        os.system(f"rm {out}")
        os.system(f"ffmpeg -i {file} -vf 'scale=iw/2:(ih/2)-1',setsar=1 {out}") #width not divisible by 2 ==> 변환 후의 가로, 세로 길이가 짝수여야 한다. #setsar=1 : 영상이 500:500을 벗어나는 현상을 막아준다.
    elif os.path.getsize(out) == 0 and flag == 'h1':
        os.system(f"rm {out}")
        os.system(f"ffmpeg -i {file} -vf 'scale=(iw/2)-1:ih/2',setsar=1 {out}")
    else:
        pass

@app.route('/videoResizing', methods = ['POST'])
def videoResizing():
    req = request.form
    print("Video Resizing start!")
    print('Request: ', req)  #req: ImmutableMultiDict
    #1. JSON 객체에 든 파라미터 가져오기 ==> 영상은 1개(url), 주소는 2개(url, url_thumb)인 것은, url_thumb는 별도로 정방형으로 변환한 영상을 저장하기 위한 주소이다. 실제로 영상이 있는 주소가 아니다.
    url = req.get('url') #Aspect Ratio에 맞게 한 후 축소하기
    url_thumb = req.get('url_thumb') #정방형(500:500)으로 나누기
    ftp_server = req.get('ftp_server')
    ftp_user_name = req.get('ftp_user_name')
    ftp_user_pass = req.get('ftp_user_pass')
    feedPk = req.get('feedPk')
    uid = req.get('uid')
    print("(1) Got values in a requested JSON object")

    #2. 비디오 확장자 목록 정의
    videoFile_extension = [ #sample video source: https://filesamples.com/
        'mp4', 'MP4',
        'm4a', 'M4A',
        'mpeg', 'MPEG',
        'mov', 'MOV', #for Apple Quicktime
        '3gp', '3GP', #for Android
        'm4v', 'M4V', # for Android
        'wmv', 'WMV', #for Windows
        'mkv', 'MKV', #mov,mp4,m4a,3gp,3g2,mj2,
        'avi', 'AVI',
        'flv', 'FLV',
    ]

    #3. path 변수에 해당하는 경로에 파일 다운로드 & 500*500으로 변환
    path = '/var/www/detectron2/video_resizing'
    fileName = url.split('/')[-1]  #'파일제목.확장자'
    filePath = os.path.join(path, fileName) #'경로/파일제목.확장자'
    if fileName.split('.')[1] == '3gp' or fileName.split('.')[1] == '3GP':
        output_filePath = os.path.join(path, 'resized', f"{fileName.split('.')[0]}.mp4")
    else:
        output_filePath = os.path.join(path, 'resized', fileName)

    # fileName_thumb = url_thumb.split('/')[-1]
    # filePath_thumb = os.path.join(path, fileName_thumb)
    # output_filePath_thumb = os.path.join(path, 'resized', fileName_thumb)

    os.chdir(path)
    os.system(f"curl -O {url}")
    print(f"(3) Done downloading video? : {os.path.isfile(filePath)}")

    if (os.path.isfile(filePath)) and (fileName.split('.')[1] in videoFile_extension):
        print("(4) Downloading done, acceptable video extension.")
        """
        #변환 사이즈: 500 * 500
        target_w = 500
        target_h = 500
        """
        video_cv = cv2.VideoCapture(filePath)
        iw = video_cv.get(cv2.CAP_PROP_FRAME_WIDTH) #input width
        ih = video_cv.get(cv2.CAP_PROP_FRAME_HEIGHT) #input height

        tw = int(iw/2) #target width
        th = int(ih/2) #target height

        if tw%2 == 1 and th%2 == 0:
            flag = 'w1'
            os.system(f"ffmpeg -i {filePath} -crf 20 -vf 'scale=(iw/2)-1:ih/2',setsar=1 {output_filePath}") #width not divisible by 2 ==> 변환 후의 가로, 세로 길이가 짝수여야 한다. #setsar=1 : 영상이 500:500을 벗어나는 현상을 막아준다.
            isFileExists(filePath, output_filePath, flag)
        elif tw%2 == 0 and th%2 == 1:
            flag = 'h1'
            os.system(f"ffmpeg -i {filePath} -crf 20 -vf 'scale=iw/2:(ih/2)-1',setsar=1 {output_filePath}") #width not divisible by 2 ==> 변환 후의 가로, 세로 길이가 짝수여야 한다. #setsar=1 : 영상이 500:500을 벗어나는 현상을 막아준다.
            isFileExists(filePath, output_filePath, flag)
        elif tw%2 == 1 and th%2 == 1:
            flag = 'wh1'
            os.system(f"ffmpeg -i {filePath} -crf 20 -vf 'scale=(iw/2)-1:(ih/2)-1',setsar=1 {output_filePath}")
        elif tw%2 == 0 and th%2 == 0:
            flag = 'wh0'
            os.system(f"ffmpeg -i {filePath} -crf 20 -vf 'scale=iw/2:ih/2',setsar=1 {output_filePath}") #width not divisible by 2 ==> 변환 후의 가로, 세로 길이가 짝수여야 한다. #setsar=1 : 영상이 500:500을 벗어나는 현상을 막아준다.
        # else:
        #     os.system(f"ffmpeg -i {filePath} -crf 20 -vf 'scale=500:500',setsar=1 {output_filePath}")
        # targetAspectRatio_thumb = "500:500"

        #변환 후 별도 폴더에 저장
        #os.system(f"ffmpeg -i {filePath} -vf 'scale=iw:ih' {output_filePath}") #원본 비율 그대로 변환(파일 사이즈 자체는 줄어듬)
        #os.system(f"ffmpeg -i {filePath} -vf scale={targetAspectRatio_thumb},setsar=1 {output_filePath_thumb}")
        print("(5) Resizing done!")

        #3-1. Create ftplib object
        f = ftplib.FTP()

        #3-2. Connect & Login
        f.connect(ftp_server, 21) #Connect
        print("(6) Connected ftp server...")
        f.login(ftp_user_name, ftp_user_pass)  #Login
        print("(7) Login successed...")

        #5. Move to the original file directory
        #url 예시: https://cyld20182.speedgabia.com/Image/SNS/21217/21217_1604235660.kr
        f.cwd(f"/{url.split('/')[3]}/{url.split('/')[4]}/{uid}")
        #6. Store file in the uid directory.
        outputFile = open(output_filePath, 'rb')
        f.storbinary(f"STOR {output_filePath.split('/')[-1]}", outputFile)

        #7. Close file, FTP connection
        outputFile.close()
        f.close()

        result_dict = {
            'success': 'y',
            'message': 'success'
        }
        return json.dumps(result_dict, ensure_ascii=False), 200
    else:
        print("(8) Download error or unacceptable video extension.")
        result_dict = {
            'success': 'n',
            'message': 'Invalid video extension'
        }
        return json.dumps(result_dict, ensure_ascii=False), 401

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
    #app.debug = True
    #app.run(host='0.0.0.0', port=443, ssl_context=('/etc/letsencrypt/live/www.circlinad.co.kr/cert.pem', '/etc/letsencrypt/live/www.circlinad.co.kr/privkey.pem'))
