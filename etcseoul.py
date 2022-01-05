<<<<<<< HEAD
=======
# 기본 설정
# detectron2 logger 설정
>>>>>>> 424c3e5d07c4b85e742c80b90e58d819a0abdea3
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

<<<<<<< HEAD
=======
# 자주 사용하는 라이브러리 import
>>>>>>> 424c3e5d07c4b85e742c80b90e58d819a0abdea3
import numpy as np
import cv2
import random
import time
import datetime
import json
import ftplib
import os
from urllib.request import urlopen

<<<<<<< HEAD
=======
# 자주 사용하는 detectron2 유틸 import
>>>>>>> 424c3e5d07c4b85e742c80b90e58d819a0abdea3
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

<<<<<<< HEAD

=======
#본 코드의 이미지 처리는 CPU로 처리함(GPU 및 CUDA가 이 VM에 설치되어있지 않음).
>>>>>>> 424c3e5d07c4b85e742c80b90e58d819a0abdea3
def analysis(dir, targetDir, inputFileName):
    start_time = time.time()

    #2. 이미지 데이터 read
    print('img_url: ', dir)
    im = cv2.imread(dir)

    #3. Segmentation & KeyPoints extraction.
    #3-1. Segmentation
    cfg_Seg = get_cfg() # Segmentation용 configuration 추가
    cfg_Seg.MODEL.DEVICE = 'cpu'
    cfg_Seg.merge_from_list(['MODEL.DEVICE','cpu'])  #이 코드를 추가하시면 cpu모드로 구동하게 됩니다.
    cfg_Seg.merge_from_file("/var/www/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") #cd detectron2 로 경로 변경한 경우 이것으로 실행.
    cfg_Seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this mdoel

    #detectron2 model zoo에서 모델 선택(다양한 모델을 사용할 수 있음)
    cfg_Seg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    predictor_Seg = DefaultPredictor(cfg_Seg)
    outputs_Seg = predictor_Seg(im)
    #print("Detected classes: ", outputs_Seg["instances"].pred_classes)

    #Removing bounding box, class label, predicting score.
    outputs_Seg['instances'].remove('pred_boxes') #Removing bounding box.
    outputs_Seg['instances'].remove('pred_classes') #Removing class label.
    outputs_Seg['instances'].remove('scores') #Removing predicting score.

    # `Visualizer`를 이용하면 Segmentation 예측 결과를 손쉽게 그릴 수 있습니다.
    v_Seg = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg_Seg.DATASETS.TRAIN[0]), scale=1.2)
    v_Seg = v_Seg.draw_instance_predictions(outputs_Seg["instances"].to("cpu"))

    v_Seg = v_Seg.get_image()[:, :, ::-1] #v_Seg 변수에 Segmentation 결과를 저장하고, 이 위에 keypoints를 그린다.


    # #################################################################################
    #3-2. KeyPoints detection
    cfg_Key = get_cfg() #Keypoints용 configuration 추가
    cfg_Seg.MODEL.DEVICE = 'cpu'
    cfg_Key.merge_from_list(['MODEL.DEVICE','cpu'])    #이 코드를 추가하시면 cpu모드로 구동하게 됩니다.
    #cfg_Key.merge_from_file("./detectron2_repo/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml")
    cfg_Key.merge_from_file("/var/www/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml") # cd detectron2 로 경로 변경한 경우 이것으로 실행.
    cfg_Key.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model

    # keypoint 모델 선택(다양한 모델을 사용할 수 있음).
    cfg_Key.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
    predictor_Key = DefaultPredictor(cfg_Key)
    outputs_Key = predictor_Key(v_Seg)

    #Removing bounding box, class label, predicting score.
    outputs_Key['instances'].remove('pred_boxes') #Removing bounding box.
    outputs_Key['instances'].remove('pred_classes') #Removing class label.
    outputs_Key['instances'].remove('scores') #Removing predicting score.

    #`Visualizer`를 이용하면 Segmentation 예측 결과를 손쉽게 그릴 수 있습니다.
    v_Key = Visualizer(v_Seg[:, :, ::-1], MetadataCatalog.get(cfg_Key.DATASETS.TRAIN[0]), scale=1.2)
    v_Key = v_Key.draw_instance_predictions(outputs_Key["instances"].to("cpu"))
    #cv2.imshow(v_Key.get_image()[:, :, ::-1]) #최종 이미지 확인
    #cv2.imwrite('./img_output/output.png', v_Key.get_image()[:, :, ::-1])

    #4. Keypoint, Segmentation으로부터 신체 비율 & 부위별 면적 계산하기
    #4-1. 신체 비율
    #이미지 상에서 검출된 모든 Keypoints
    keypoints_whole = outputs_Key['instances'].get('pred_keypoints')
    #print('keypoints_whole: ', keypoints_whole)

    #이미지에 인식된 사람 수 체크: keypoints를 담은 배열의 개수 확인
    #이상이 없으면 try,
    try:
        howManyPerson = len(keypoints_whole)
        howManyPerson == 1
    except:
        if howManyPerson > 1:
            msg = '인식된 사람이 한 명 이상입니다. 사진 속에 다른 사람이나 거울에 비친 자신의 모습이 있지는 않은지 확인해 주세요.'
            print('msg2: ', msg)
        elif howManyPerson == 0:
            msg = '인식된 사람이 없습니다. 정확한 분석을 위해 사람이 없거나, 너무 작게 나왔거나, 다른 물체에 많이 가려져 있는 사진은 피해주세요.'
            print('msg3: ', msg)

    """
    outputs['instances'].get('pred_keypoints)')
    {
    "name": "person", # (specific category)
    "supercategory": "person", #
    "id": 1, # class id
    "keypoints": [
        "nose",           #0 #[x, y, score]
        "left_eye",       #1
        "right_eye",      #2
        "left_ear",       #3
        "right_ear",      #4
        "left_shoulder",  #5
        "right_shoulder", #6
        "left_elbow",     #7
        "right_elbow",    #8
        "left_wrist",     #9
        "right_wrist",    #10
        "left_hip",       #11
        "right_hip",      #12
        "left_knee",      #13
        "right_knee",     #14
        "left_ankle",     #15
        "right_ankle"     #16
        ]
    }

    head_width= #4 - #3
    shoulder_width = #6 - #5
    hip_width = #12 - #11
    """
    #x: 왼쪽으로 갈수록 x값이 커짐(좌측 좌표값이 우측 좌표값보다 큼)(역방향)
    #y: 위로 갈수록 y값이 커짐(위쪽 좌표값이 아래쪽 좌표값보다 큼)(정방향)

    #모든 좌표값은 torch.tensor이므로 float로 변환해 주어야 함.
    keypoints_person = keypoints_whole[0]

    left_ear = keypoints_person[3]
    left_ear_x = left_ear[0]
    left_ear_y = left_ear[1]

    right_ear = keypoints_person[4]
    right_ear_x = right_ear[0]
    right_ear_y = right_ear[1]

    left_shoulder = keypoints_person[5]
    left_shoulder_x = left_shoulder[0]
    left_shoulder_y = left_shoulder[1]

    right_shoulder = keypoints_person[6]
    right_shoulder_x = right_shoulder[0]
    right_shoulder_y = right_shoulder[1]

    left_hip = keypoints_person[11]
    left_hip_x = left_hip[0]
    left_hip_y = left_hip[1]

    right_hip = keypoints_person[12]
    right_hip_x = right_hip[0]
    right_hip_y = right_hip[1]

    #몸통 길이
    #어깨 x,y좌표 평균 ~ 힙 x,y좌표 평균
    # body_upper_x = abs(right_shoulder_x + left_shoulder_x) / 2
    # body_below_x = abs(right_hip_x + left_hip_x) / 2
    # body_width = abs(body_upper_x - body_below_x)

    body_upper_y = abs(right_shoulder_y + left_shoulder_y) / 2
    body_below_y = abs(right_hip_y + left_hip_y) / 2
    body_length = abs(body_upper_y - body_below_y)  #####상체(몸통) 길이
    body_length = body_length.item() #형변환: torch.tensor -> float
    body_length = round(body_length, 2) #소수점 둘째 자리까지만 출력


    #왼팔 길이
    left_elbow = keypoints_person[7]
    left_elbow_y = left_elbow[1]
    left_wrist = keypoints_person[9]
    left_wrist_y = left_wrist[1]

    left_arm_upper = abs(left_shoulder_y - left_elbow_y)
    left_arm_upper = left_arm_upper.item()
    left_arm_upper = round(left_arm_upper, 2)

    left_arm_below = abs(left_elbow_y - left_wrist_y)
    left_arm_below = left_arm_below.item()
    left_arm_below = round(left_arm_below, 2)

    left_arm = left_arm_upper + left_arm_below   ####왼팔
    #left_arm = left_arm.item() #형변환: torch.tensor -> float
    #left_arm = round(left_arm, 2) #소수점 둘째 자리까지만 출력

    #오른팔 길이
    right_elbow = keypoints_person[8]
    right_elbow_y = right_elbow[1]
    right_wrist = keypoints_person[10]
    right_wrist_y = right_wrist[1]

    right_arm_upper = abs(right_shoulder_y - right_elbow_y)
    right_arm_upper =  right_arm_upper.item()
    right_arm_upper = round(right_arm_upper, 2)

    right_arm_below = abs(right_elbow_y - right_wrist_y)
    right_arm_below = right_arm_below.item()
    right_arm_below = round(right_arm_below, 2)

    right_arm = right_arm_upper + right_arm_below   ####오른팔
    #right_arm = right_arm.item() #형변환: torch.tensor -> float
    #right_arm = round(right_arm, 2) #소수점 둘째 자리까지만 출력

    #왼다리 길이

    #오른다리 길이
    right_ankle = keypoints_person[16]
    right_ankle_y = right_ankle[1]
    right_leg_length = abs(right_hip_y - right_ankle_y)  ####다리 길이
    right_leg_length = right_leg_length.item() #형변환: torch.tensor -> float
    right_leg_length = round(right_leg_length, 2) #소수점 둘째 자리까지만 출력

    ###상체 비율( 몸통 / 몸통 + 다리)
    #upper_ratio = body_length / (body_length + right_leg_length)
    #upper_ratio = round(upper_ratio, 2) #소수점 둘째 자리까지만 출력

    ###하체 비율 ( 다리 / 몸통 + 다리 )
    #leg_ratio = right_leg_length / (body_length + right_leg_length)
    #leg_ratio = round(leg_ratio, 2)

    ####왼팔 비율(왼팔 / 몸통 )
    #left_arm_ratio = left_arm / body_length
    #left_arm_ratio = round(left_arm_ratio, 2)

    ####오른팔 비율(오른팔 / 몸통)
    #right_arm_ratio = right_arm / body_length
    #right_arm_ratio = round(right_arm_ratio, 2)

    #torch.tensor들 float으로 변환 & 소수점 둘째자리까지만 계산
    try:
        head_width = abs(left_ear_x - right_ear_x) #type: torch.tensor
        #print(head_width)
        head_width = head_width.item() #형변환: torch.tensor -> float
        head_width = round(head_width, 2) #소수점 둘째 자리까지만 출력
        msg = '분석이 완료되었습니다.'
    except:
        if head_width <= 0 :
            msg = '머리 너비가 제대로 인식되지 않았어요. 양쪽 귀가 모두 사진에 담기도록 정면을 바라본 상태에서 촬영해 주세요.'

    try:
        shoulder_width = abs(left_shoulder_x - right_shoulder_x) #type: torch.tensor
        #print(shoulder_width)
        shoulder_width = shoulder_width.item() #형변환: torch.tensor -> float
        shoulder_width = round(shoulder_width, 2) #소수점 둘째 자리까지만 출력
        msg = '분석이 완료되었습니다.'
    except:
        if shoulder_width <= 0:
            msg = '어깨 너비가 제대로 인식되지 않았어요. 몸이 틀어지지 않았는지 다시 확인해 주시고, 양쪽 어깨가 모두 사진에 담기도록 정면을 바라본 상태에서 촬영해 주세요. '

    try:
        hip_width = abs(left_hip_x - right_hip_x) #type: torch.tensor
        #print(hip_width)
        hip_width = hip_width.item() #형변환: torch.tensor -> float
        hip_width = round(hip_width, 2) #소수점 둘째 자리까지만 출력
        msg = '분석이 완료되었습니다.'
    except:
        if hip_width <= 0:
            msg = '엉덩이 너비가 제대로 인식되지 않았어요. 몸이 틀어지지 않았는지 확인해 주시고, 정면을 바라본 상태에서 촬영해 주세요.'

    #어깨비율, 엉덩이비율   #####기준: head_width
    shoulder_head = shoulder_width / head_width #어깨 너비가 커질수록 shoulder_head는 거진다.
    shoulder_head = round(shoulder_head, 2) #소수점 둘째 자리까지만 출력
    hip_head = hip_width / head_width #엉덩이 너비가 커질수록 hip_head는 커진다.
    hip_head = round(hip_head, 2)  #소수점 둘째 자리까지만 출력

    #상체 비율, 하체 비율   #####기준: head_width
    body_head = body_length / head_width
    body_head = round(body_head, 2)
    leg_head = right_leg_length / head_width
    leg_head = round(leg_head, 2)  #소수점 둘째 자리까지만 출력

    #오른팔, 왼팔 비   #####기준: head_width
    leftarm_head = left_arm / head_width
    leftarm_head = round(leftarm_head, 2)
    rightarm_head = right_arm / head_width
    rightarm_head = round(rightarm_head, 2)

    """
    time_ymd = datetime.datetime.now().strftime('%Y%m%d') #저장하는 현재 날짜
    #처리 결과를 '경로 + (output_회원 ID_날짜_이미지 이름)'로 이미지 파일로 저장.
    cv2.imwrite('/var/www/detectron2/img_output/output_'+'USER_ID_'+time_ymd+'_'+img_name, v_Key.get_image()[:, :, ::-1])

    print("Segmentation & Keypoints Image was saved successfully!")

    #6. 이미지 1장 처리에 걸리는 시간 측정(초 단위). #GPU 및 CUDA가 설치된다면 더 빠르게 가능할 것이다.
    timeEnd = time.time() - timeStart
    print("Time passed for fully processing 1 image: ",timeEnd) #현재 시각 - 시작 시간
    """
    dt = datetime.datetime.now()
    nt = dt.strftime('%Y%m%d%H%M%S')

    #결과물 이미지 저장 경로
    OUTPUT_PATH = os.path.join('/var/www/detectron2/img_input/etcseoul/', targetDir, 'output', 'output_'+inputFileName)
    #결과물 이미지 저장
    cv2.imwrite(OUTPUT_PATH, v_Key.get_image()[:, :, ::-1])
    #output = cv2.resize(v_Key.get_image()[:, :, ::-1], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR) ######아래 줄과 바꿔치기하기

    # #FTP 전송
    ftp = ftplib.FTP()
    ftp.connect("cyld20182.speedgabia.com", 21)
    ftp.login("cyld20182", "teamcyld2018!")
    ftp.cwd('Image/ETC')
    try:
        ftp.mkd(targetDir) #ftp에서 폴더 생성
    except:
        1+1

    ftp.cwd(targetDir) #ftp에서 해당 폴더로 이동
    #os.path.join("detectron2/img_input/etcseoul/", targetDir, "output")
    #os.listdir()  #전체 파일 조회(for문을 써야 하나) #ftp.close 전까지 반복문 안에 들어가야.
    myfile = open(OUTPUT_PATH, 'rb')
    #outfile = targetDir + inputFileName
    print('myfile: ', myfile)
    # #print('outfile: ', outfile)
    ftp.storbinary('STOR ' +'output_'+inputFileName, myfile) # storbinary(파일 이름 설정(outfile), 실제로 보내질 경로 및 파일명(myfile))
    myfile.close()
    ftp.close
    print("complete!")

    result_dict = {
    'shoulder_ratio': shoulder_head,
    'hip_ratio': hip_head,
    'upper_ratio': body_head, #upper_ratio,
    'leg_ratio': leg_head, #leg_ratio,
    'left_arm_ratio': leftarm_head,
    'right_arm_ratio': rightarm_head
    }

    print("RESULT::::::: ", result_dict)

    #print("Everything done!")
    #print(result_dict)
    end_time = time.time()
    #print("Time spent: ", end_time - start_time)
    return json.dumps(result_dict)
