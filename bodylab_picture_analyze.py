import numpy as np
import cv2
import time
import datetime
import json
import os
from urllib.request import urlopen
from global_things.constants import AMAZON_URL, BUCKET_NAME, BUCKET_BODY_IMAGE_OUTPUT_PATH, BODY_IMAGE_OUTPUT_PATH
from global_things.functions import upload_output_to_s3

# 출력 형식에 관한 자세한 내용은 다음 주소를 참고하세요: # https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
setup_logger()


# 본 코드의 이미지 처리는 CPU로 처리함(GPU 및 CUDA가 이 VM에 설치되어있지 않음).
def analysis(url, user_id):
    start_time = time.time()

    # 1. 이미지 데이터 read
    try:
        req = urlopen(url)
    except Exception as e:
        result_dict = {
            'url': url,
            'user_id': user_id,
            'message': 'Cannot find image.',
            'result': False
        }
        print(result_dict)
        return json.dumps(result_dict)

    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    im = cv2.imdecode(arr, -1)

    # 3. Segmentation & KeyPoints extraction.
    # 3-1. Segmentation
    cfg_seg = get_cfg()  # Segmentation 을 위한 configuration 추가
    cfg_seg.MODEL.DEVICE = 'cpu'
    cfg_seg.merge_from_list(['MODEL.DEVICE','cpu'])  #이 코드를 추가하시면 cpu모드로 구동하게 됩니다.
    # cfg_seg.merge_from_file("/var/www/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml") #cd detectron2 로 경로 변경한 경우 이것으로 실행.
    cfg_seg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model.

    # detectron2 model zoo에서 모델 선택(다양한 모델을 사용할 수 있음)
    cfg_seg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    predictor_Seg = DefaultPredictor(cfg_seg)

    try:
        outputs_Seg = predictor_Seg(im)
        print("Detected classes: ", outputs_Seg["instances"].pred_classes)
    except:
        result_dict = {
            'message': 'Unacceptable file extension.',
            'result': False
        }
        print(result_dict)
        return json.dumps(result_dict)

    # Copy outputs_Seg and filter 'person' class only.
    person_seg = outputs_Seg['instances'][outputs_Seg['instances'].pred_classes == 0]
    print("Detected persons: ", len(person_seg))
    print("Detected persons information: ", person_seg)

    if len(person_seg) == 1:
        # Remain only one highest probability of person.
        person_seg_trustful = person_seg[person_seg.scores == person_seg.scores.max()]
        print("Person who has the highest probability: ", person_seg_trustful.to("cpu"))

        # Remove bounding box, class label, predicting score from the image.
        person_seg_trustful.remove('pred_boxes')  # Removing bounding box.
        person_seg_trustful.remove('pred_classes')  # Removing class label.
        person_seg_trustful.remove('scores')  # Removing predicting score.

        # `Visualizer`를 이용하면 Segmentation 예측 결과를 손쉽게 그릴 수 있습니다.
        v_seg = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg_seg.DATASETS.TRAIN[0]), scale=1.2)
        v_seg = v_seg.draw_instance_predictions(person_seg_trustful.to("cpu"))  # 20210506: outputs_Seg["instances"] -> outputs_Seg["instances"][0]
        v_seg = v_seg.get_image()[:, :, ::-1]   # v_Seg 변수에 Segmentation 결과를 저장하고, keypoints를 그리기 위해 다음 섹션에서 사용합니다.
    elif len(person_seg) > 1:
        score_gap = person_seg.scores[0] - person_seg.scores[1]  # 최대, 차대 점수 차
        if score_gap >= 0.01:
            # 사람 형태가 모두가 확실한 것은 아니라고 가정. ==> 상위 1명만 Segmentation 그린다.
            # Remain only one highest probability of person.
            person_seg_trustful = person_seg[person_seg.scores == person_seg.scores.max()]
            print("Person who has the highest probability: ", person_seg_trustful.to("cpu"))

            # Remove bounding box, class label, predicting score from the image.
            person_seg_trustful.remove('pred_boxes')  # Removing bounding box.
            person_seg_trustful.remove('pred_classes')  # Removing class label.
            person_seg_trustful.remove('scores')  # Removing predicting score.

            # `Visualizer`를 이용하면 Segmentation 예측 결과를 손쉽게 그릴 수 있습니다.
            v_seg = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg_seg.DATASETS.TRAIN[0]), scale=1.2)
            v_seg = v_seg.draw_instance_predictions(person_seg_trustful.to("cpu"))    # 20210506: outputs_Seg["instances"] -> outputs_Seg["instances"][0]
            v_seg = v_seg.get_image()[:, :, ::-1]  # v_Seg 변수에 Segmentation 결과를 저장하고, keypoints를 그리기 위해 다음 섹션에서 사용합니다.
        else:
            # 사람 간 형태가 모두 확실하다고 가정. ==> 검출된 모든 사람에 Segmentation을 그린다.
            # Remove bounding box, class label, predicting score from the image.
            person_seg.remove('pred_boxes')  # Removing bounding box.
            person_seg.remove('pred_classes')  # Removing class label.
            person_seg.remove('scores')  # Removing predicting score.

            # `Visualizer`를 이용하면 Segmentation 예측 결과를 손쉽게 그릴 수 있습니다.
            v_seg = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg_seg.DATASETS.TRAIN[0]), scale=1.2)
            v_seg = v_seg.draw_instance_predictions(person_seg.to("cpu"))    # 20210506: outputs_Seg["instances"] -> outputs_Seg["instances"][0]
            v_seg = v_seg.get_image()[:, :, ::-1]  # v_Seg 변수에 Segmentation 결과를 저장하고, keypoints를 그리기 위해 다음 섹션에서 사용합니다.
    else:
        # if len(person_Seg) == 0  ==>  Error occurs at person_Seg_trustful.
        result_dict = {
            'message': 'No person detected.',
            'result': False
        }
        print(result_dict)
        return json.dumps(result_dict)

    # #################################################################################
    # 3-2. KeyPoints detection
    cfg_key = get_cfg()  # Keypoints용 configuration 추가
    cfg_seg.MODEL.DEVICE = 'cpu'
    cfg_key.merge_from_list(['MODEL.DEVICE','cpu'])    # 이 코드를 추가하시면 cpu모드로 구동하게 됩니다.
    # cfg_key.merge_from_file("/var/www/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml") # cd detectron2 로 경로 변경한 경우 이것으로 실행.
    cfg_key.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg_key.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model

    # keypoint 모델 선택(다양한 모델을 사용할 수 있음).
    cfg_key.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
    predictor_key = DefaultPredictor(cfg_key)
    outputs_key = predictor_key(v_seg)

    # Copy outputs_Seg and filter keypoints of the 'person' object who has highest probability. #Keypoints are only for person, so class filtering doesn't needed.
    try:
        person_key = outputs_key['instances'][outputs_key['instances'].scores == outputs_key['instances'].scores.max()]
        print("person_key: ", person_key)
    except:
        # if len(outputs_Key['instances'].pred_classes) == 0  ==>  Error occurs at person_Key because no keypoints to draw detected.
        result_dict = {
            'message': 'No person detected.',
            'result': False
        }
        print(result_dict)
        return json.dumps(result_dict)

    """Removing bounding box, class label, predicting score."""
    person_key.remove('pred_boxes')  # Removing bounding box.
    person_key.remove('pred_classes')  # Removing class label.
    person_key.remove('scores')  # Removing predicting score.

    # `Visualizer`를 이용하면 Keypoint 예측 결과를 손쉽게 그릴 수 있습니다.
    v_key = Visualizer(v_seg[:, :, ::-1], MetadataCatalog.get(cfg_key.DATASETS.TRAIN[0]), scale=1.2)
    v_key = v_key.draw_instance_predictions(person_key.to("cpu"))  # 20210506: outputs_key["instances"] -> outputs_key["instances"][0]

    # 4. Keypoint, Segmentation으로부터 신체 비율 & 부위별 면적 계산하기
    keypoints_whole = person_key.get('pred_keypoints')  # Every keypoints on an image
    keypoints_person = keypoints_whole[0]  # keypoints of one highest probability person
    howManyPerson = len(keypoints_whole)  # Number of keypoint tensors(Number of 2-D tensors)(1 2-D tensor is consists of 17 1-D tensors.)
    howManyKeyPoints = len(keypoints_person)  # Number of keypoints of one highest probability person
    print("how many person: ", howManyPerson)
    print("Number of keypoints: ", howManyKeyPoints)
    print("Location of keypoints of one highest person: ", keypoints_person)
    if howManyKeyPoints < 17:
        print("Keypoints error: len(keypoints_person) < 17")
        result_dict = {
            'message': 'Bad pose: Unable to detect the whole body joints.',
            'result': False
        }
        return json.dumps(result_dict)
    elif howManyKeyPoints == 17:
        pass

    if howManyPerson > 1:
        result_dict = {
            'message': 'Too many people.',
            'result': False
        }
        print('Person error: ', result_dict)
        return json.dumps(result_dict)
    elif howManyPerson == 0:
        result_dict = {
            'message': 'No person detected.',
            'result': False
        }
        print('Person error: ', result_dict)
        return json.dumps(result_dict)
    else:
        pass

    """
    "keypoints": [
        "nose",           #0 #[x, y, score] #x: 왼쪽으로 갈수록 x값이 커짐(좌측 좌표값이 우측 좌표값보다 큼)(역방향) #y: 위로 갈수록 y값이 커짐(위쪽 좌표값이 아래쪽 좌표값보다 큼)(정방향)
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

    head_width= #4 - #3
    shoulder_width = #6 - #5
    hip_width = #12 - #11
    """

    # 모든 좌표값은 torch.tensor이므로 float로 변환해 주어야 함.
    nose = keypoints_person[0]
    nose_x = nose[0]
    nose_y = nose[1]

    left_ear = keypoints_person[3]
    left_ear_x = left_ear[0]
    left_ear_y = left_ear[1]

    right_ear = keypoints_person[4]
    right_ear_x = right_ear[0]
    right_ear_y = right_ear[1]

    # 어깨단
    left_shoulder = keypoints_person[5]
    left_shoulder_x = left_shoulder[0]
    left_shoulder_y = left_shoulder[1]

    right_shoulder = keypoints_person[6]
    right_shoulder_x = right_shoulder[0]
    right_shoulder_y = right_shoulder[1]

    center_shoulder_x = abs((right_shoulder_x + left_shoulder_x) / 2).item()
    center_shoulder_y = abs((right_shoulder_y + left_shoulder_y) / 2).item()

    # 엉덩이단
    left_hip = keypoints_person[11]
    left_hip_x = left_hip[0]
    left_hip_y = left_hip[1]

    right_hip = keypoints_person[12]
    right_hip_x = right_hip[0]
    right_hip_y = right_hip[1]

    center_hip_x = abs((right_hip_x + left_hip_x) / 2).item()
    center_hip_y = abs((right_hip_y + left_hip_y) / 2).item()

    # 발목단
    right_ankle = keypoints_person[15]
    right_ankle_x = right_ankle[0]
    right_ankle_y = right_ankle[1]

    left_ankle = keypoints_person[16]
    left_ankle_x = left_ankle[0]
    left_ankle_y = left_ankle[1]

    center_ankle_x = abs((right_ankle_x + left_ankle_x) / 2).item()
    center_ankle_y = abs((right_ankle_y + left_ankle_y) / 2).item()

    # torch.tensor들 float으로 변환 & 소수점 둘째자리까지만 계산
    head_width = abs(left_ear_x - right_ear_x)  # type: torch.tensor
    print(head_width)
    head_width = head_width.item()  # 형변환: torch.tensor -> float
    head_width = round(head_width, 2)  # 소수점 둘째 자리까지만 출력

    shoulder_width = abs(left_shoulder_x - right_shoulder_x)  # type: torch.tensor
    print(shoulder_width)
    shoulder_width = shoulder_width.item()  # 형변환: torch.tensor -> float
    shoulder_width = round(shoulder_width, 2)  # 소수점 둘째 자리까지만 출력

    # try:
    hip_width = abs(left_hip_x - right_hip_x)  # type: torch.tensor
    print(hip_width)
    hip_width = hip_width.item()  # 형변환: torch.tensor -> float
    hip_width = round(hip_width, 2)  # 소수점 둘째 자리까지만 출력

    try:
        shoulder_head = shoulder_width / head_width # 어깨 너비가 커질수록 shoulder_head는 거진다.
        shoulder_head = round(shoulder_head, 2) # 소수점 둘째 자리까지만 출력
        hip_head = hip_width / head_width # 엉덩이 너비가 커질수록 hip_head는 커진다.
        hip_head = round(hip_head, 2)  # 소수점 둘째 자리까지만 출력
    except:
        result_dict = {
            'message': 'Bad pose. Head or hip width is 0',
            'result': True
        }
        print('Pose error: ', result_dict)
        return json.dumps(result_dict)

    # 길이값
    # 코~가슴 어깨 중앙 = h1
    nose_to_shoulder_center = abs(nose_y - center_shoulder_y).item()  # type: #torch.tensor
    # 어깨 중앙 ~ 골반 중앙 = h2
    shoulder_center_to_hip_center = abs(center_shoulder_y - center_hip_y)
    # 골반 중앙 ~ 발목 중앙 = h3
    hip_center_to_ankle_center = abs(center_hip_y - center_ankle_y)
    # 전신 길이 = h1 + h2 + h3
    whole_body_length = nose_to_shoulder_center + shoulder_center_to_hip_center + hip_center_to_ankle_center
    # 상체 + 하체 길이 = h2 + h3
    upper_body_lower_body = shoulder_center_to_hip_center + hip_center_to_ankle_center

    """
    #6. 이미지 1장 처리에 걸리는 시간 측정(초 단위). #GPU 및 CUDA가 설치된다면 더 빠르게 가능할 것이다.
    timeEnd = time.time() - timeStart
    print("Time passed for fully processing 1 image: ",timeEnd) #현재 시각 - 시작 시간
    """
    dt = datetime.datetime.now()
    nt = dt.strftime('%Y%m%d%H%M%S')

    output = cv2.resize(v_key.get_image()[:, :, ::-1], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    if str(user_id) not in os.listdir(f"{BODY_IMAGE_OUTPUT_PATH}"):
        os.makedirs(f"{BODY_IMAGE_OUTPUT_PATH}/{user_id}")
    file_name = f'{user_id}_{nt}.jpg'
    local_image_path = f'{BODY_IMAGE_OUTPUT_PATH}/{user_id}/{file_name}'
    cv2.imwrite(local_image_path, output)

    # 7. S3에 분석 결과 이미지(output_path) 원본 저장
    object_name = f"{BUCKET_BODY_IMAGE_OUTPUT_PATH}/{user_id}/{file_name}"
    if upload_output_to_s3(local_image_path, BUCKET_NAME, object_name) is True:
        pass
    else:
        result_dict = {
            'message': f'Error while upload output image into S3 for original image.',
            'result': False
        }
        print('Pose error: ', result_dict)
        return json.dumps(result_dict)
    s3_path_body_output = f"{AMAZON_URL}/{object_name}"
    if os.path.exists(local_image_path):
        os.remove(local_image_path)

    if shoulder_head == 0 or hip_head == 0 or shoulder_width == 0 or hip_width == 0 or nose_to_shoulder_center == 0 or shoulder_center_to_hip_center == 0 or hip_center_to_ankle_center == 0 or whole_body_length == 0 or upper_body_lower_body == 0:
        print(f'shoulder_head: {shoulder_head}')
        print(f'hip_head: {hip_head}')
        print(f'shoulder_width: {shoulder_width}')
        print(f'hip_width: {hip_width}')
        print(f'nose_to_shoulder_center: {nose_to_shoulder_center}')
        print(f'shoulder_center_to_hip_center: {shoulder_center_to_hip_center}')
        print(f'hip_center_to_ankle_center: {hip_center_to_ankle_center}')
        print(f'whole_body_length: {whole_body_length}')
        print(f'upper_body_lower_body: {upper_body_lower_body}')
        result_dict = {
            'message': 'Bad pose. Invalid body length.',
            'result': False
        }
        print('Pose error: ', result_dict)
        return json.dumps(result_dict)
    else:
        pass

    # 8. 써클인 서버에 분석 결과 이미지 주소 + 수치 데이터 + Input 시 받은 회원 정보 전송
    result_dict = {
        'result': True,
        'original_img_url': url,
        'output_url': s3_path_body_output,
        'shoulder_ratio': shoulder_head,
        'hip_ratio': hip_head,
        'shoulder_width': shoulder_width,
        'hip_width': hip_width,
        'nose_to_shoulder_center': nose_to_shoulder_center,  # 코~가슴 어깨 중앙 = h1
        'shoulder_center_to_hip_center': shoulder_center_to_hip_center,  # 어깨 중앙 ~ 골반 중앙 = h2
        'hip_center_to_ankle_center': hip_center_to_ankle_center,  # 골반 중앙 ~ 발목 중앙 = h3
        'whole_body_length': whole_body_length,  # 전신 길이 = h1 + h2 + h3
        'upper_body_lower_body': upper_body_lower_body,  # 상체 + 하체 길이 = h2 + h3
        'message': 'Done analysis'  # 성공 혹은 실패 여부
    }

    print("Everything done!")
    print(result_dict)
    end_time = time.time()
    print("Time spent: ", end_time - start_time)
    return json.dumps(result_dict)
