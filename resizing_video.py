# import os
# import json
# from flask import request
# import ftplib
# import cv2
#
#
# def is_file_exists(file, out, flag):
#     # squared4_ 파일이 없으면 -1을 반대 방향에 감하여 영상 제작.
#     if os.path.getsize(out) == 0 and flag == 'w1':
#         os.system(f"rm {out}")
#         os.system(
#             f"ffmpeg -i {file} -vf 'scale=iw/2:(ih/2)-1',setsar=1 {out}")  # width not divisible by 2 ==> 변환 후의 가로, 세로 길이가 짝수여야 한다. #setsar=1 : 영상이 500:500을 벗어나는 현상을 막아준다.
#     elif os.path.getsize(out) == 0 and flag == 'h1':
#         os.system(f"rm {out}")
#         os.system(f"ffmpeg -i {file} -vf 'scale=(iw/2)-1:ih/2',setsar=1 {out}")
#     else:
#         pass
#
#
# #@app.route('/video-resizing', methods=['POST'])
# def video_resizing():
#     req = request.form
#     # 1. JSON 객체에 든 파라미터 가져오기 ==> 영상은 1개(url), 주소는 2개(url, url_thumb)인 것은, url_thumb는 별도로 정방형으로 변환한 영상을 저장하기 위한 주소이다. 실제로 영상이 있는 주소가 아니다.
#     url = req.get('url')  # Aspect Ratio에 맞게 한 후 축소하기
#     url_thumb = req.get('url_thumb')  # 정방형(500:500)으로 나누기
#     ftp_server = req.get('ftp_server')
#     ftp_user_name = req.get('ftp_user_name')
#     ftp_user_pass = req.get('ftp_user_pass')
#     feedPk = req.get('feedPk')
#     uid = req.get('uid')
#
#     # 2. 비디오 확장자 목록 정의
#     videoFile_extension = [  # sample video source: https://filesamples.com/
#         'mp4', 'MP4',
#         'm4a', 'M4A',
#         'mpeg', 'MPEG',
#         'mov', 'MOV',  # for Apple Quicktime
#         '3gp', '3GP',  # for Android
#         'm4v', 'M4V',  # for Android
#         'wmv', 'WMV',  # for Windows
#         'mkv', 'MKV',  # mov,mp4,m4a,3gp,3g2,mj2,
#         'avi', 'AVI',
#         'flv', 'FLV',
#     ]
#
#     # 3. path 변수에 해당하는 경로에 파일 다운로드 & 500*500으로 변환
#     path = '/var/www/detectron2/video_resizing'
#     fileName = url.split('/')[-1]  # '파일제목.확장자'
#     filePath = os.path.join(path, fileName)  # '경로/파일제목.확장자'
#     if fileName.split('.')[1] == '3gp' or fileName.split('.')[1] == '3GP':
#         output_filePath = os.path.join(path, 'resized', f"{fileName.split('.')[0]}.mp4")
#     else:
#         output_filePath = os.path.join(path, 'resized', fileName)
#
#     # fileName_thumb = url_thumb.split('/')[-1]
#     # filePath_thumb = os.path.join(path, fileName_thumb)
#     # output_filePath_thumb = os.path.join(path, 'resized', fileName_thumb)
#
#     os.chdir(path)
#     os.system(f"curl -O {url}")
#
#     if (os.path.isfile(filePath)) and (fileName.split('.')[1] in videoFile_extension):
#         """
#         #변환 사이즈: 500 * 500
#         target_w = 500
#         target_h = 500
#         """
#         video_cv = cv2.VideoCapture(filePath)
#         iw = video_cv.get(cv2.CAP_PROP_FRAME_WIDTH)  # input width
#         ih = video_cv.get(cv2.CAP_PROP_FRAME_HEIGHT)  # input height
#
#         tw = int(iw / 2)  # target width
#         th = int(ih / 2)  # target height
#
#         if tw % 2 == 1 and th % 2 == 0:
#             flag = 'w1'
#             os.system(
#                 f"ffmpeg -i {filePath} -crf 20 -vf 'scale=(iw/2)-1:ih/2',setsar=1 {output_filePath}")  # width not divisible by 2 ==> 변환 후의 가로, 세로 길이가 짝수여야 한다. #setsar=1 : 영상이 500:500을 벗어나는 현상을 막아준다.
#             is_file_exists(filePath, output_filePath, flag)
#         elif tw % 2 == 0 and th % 2 == 1:
#             flag = 'h1'
#             os.system(
#                 f"ffmpeg -i {filePath} -crf 20 -vf 'scale=iw/2:(ih/2)-1',setsar=1 {output_filePath}")  # width not divisible by 2 ==> 변환 후의 가로, 세로 길이가 짝수여야 한다. #setsar=1 : 영상이 500:500을 벗어나는 현상을 막아준다.
#             is_file_exists(filePath, output_filePath, flag)
#         elif tw % 2 == 1 and th % 2 == 1:
#             flag = 'wh1'
#             os.system(f"ffmpeg -i {filePath} -crf 20 -vf 'scale=(iw/2)-1:(ih/2)-1',setsar=1 {output_filePath}")
#         elif tw % 2 == 0 and th % 2 == 0:
#             flag = 'wh0'
#             os.system(
#                 f"ffmpeg -i {filePath} -crf 20 -vf 'scale=iw/2:ih/2',setsar=1 {output_filePath}")  # width not divisible by 2 ==> 변환 후의 가로, 세로 길이가 짝수여야 한다. #setsar=1 : 영상이 500:500을 벗어나는 현상을 막아준다.
#         # else:
#         #     os.system(f"ffmpeg -i {filePath} -crf 20 -vf 'scale=500:500',setsar=1 {output_filePath}")
#         # targetAspectRatio_thumb = "500:500"
#
#         # 변환 후 별도 폴더에 저장
#         # os.system(f"ffmpeg -i {filePath} -vf 'scale=iw:ih' {output_filePath}") #원본 비율 그대로 변환(파일 사이즈 자체는 줄어듬)
#         # os.system(f"ffmpeg -i {filePath} -vf scale={targetAspectRatio_thumb},setsar=1 {output_filePath_thumb}")
#
#         # 3-1. Create ftplib object
#         f = ftplib.FTP()
#
#         # 3-2. Connect & Login
#         f.connect(ftp_server, 21)  # Connect
#         f.login(ftp_user_name, ftp_user_pass)  # Login
#
#         # 5. Move to the original file directory
#         # url 예시: https://cyld20182.speedgabia.com/Image/SNS/21217/21217_1604235660.kr
#         f.cwd(f"/{url.split('/')[3]}/{url.split('/')[4]}/{uid}")
#         # 6. Store file in the uid directory.
#         outputFile = open(output_filePath, 'rb')
#         f.storbinary(f"STOR {output_filePath.split('/')[-1]}", outputFile)
#
#         # 7. Close file, FTP connection
#         outputFile.close()
#         f.close()
#
#         result_dict = {
#             'success': 'y',
#             'message': 'success'
#         }
#         return json.dumps(result_dict, ensure_ascii=False), 200
#     else:
#         result_dict = {
#             'success': 'n',
#             'message': 'Invalid video extension'
#         }
#         return json.dumps(result_dict, ensure_ascii=False), 401
