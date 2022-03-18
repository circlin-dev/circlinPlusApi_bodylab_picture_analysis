import boto3
import cv2
from global_things.constants import AMAZON_URL, BUCKET_NAME, BUCKET_IMAGE_PATH_BODY_OUTPUT, LOCAL_SAVE_PATH_BODY_OUTPUT
from datetime import datetime
import filetype
import json
import numpy as np
import os
import pandas as pd
from PIL import Image
import pyheif
import time
import shutil
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import Normalizer
from urllib.request import urlopen
import uuid

# 출력 형식에 관한 자세한 내용은 다음 주소를 참고하세요: # https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.utils.logger import setup_logger
setup_logger()


# region free-trial
def validate_and_save_to_s3(file_type, save_path, bucket_path, file):
    unique_name = uuid.uuid4()
    invalid_mimes = ['heic', 'HEIC', 'heif', 'HEIF']

    mime = get_image_information(file)['mime_type'].split('/')
    if mime[0] != 'image':
        result = {
            'result': False,
            'error': 'Invalid file type(Requested file is not an image).'
        }
        return result

    if file_type == 'input':
        if mime[1] in invalid_mimes:
            new_secure_file, extension = heic_to_jpg(file)
            if os.path.exists(new_secure_file):
                shutil.move(new_secure_file, save_path)
            # 처리 불가한 이미지 확장자 파일은, 변환 후 원본 파일은 지워야 한다.
            if os.path.exists(file):
                os.remove(file)
            converted_file = f"{save_path}/{new_secure_file}"
            encrypted_file_name = f"{unique_name}.{extension}"
            encrypted_file_path = f"{save_path}/{encrypted_file_name}"
            if os.path.exists(converted_file):
                os.rename(converted_file, encrypted_file_path)
        else:
            # 처리 가능한 이미지 확장자 파일은, 원본을 다른 디렉터리로 옮겨 작업한다.
            if os.path.exists(file):
                shutil.move(file, save_path)
            moved_file = f"{save_path}/{file}"
            extension = file.split('.')[-1]
            encrypted_file_name = f"{unique_name}.{extension}"
            encrypted_file_path = f"{save_path}/{encrypted_file_name}"
            if os.path.exists(moved_file):
                os.rename(moved_file, encrypted_file_path)

        image_height, image_width, image_channel = cv2.imread(encrypted_file_path, cv2.IMREAD_COLOR).shape
        object_name = f"{bucket_path}/{encrypted_file_name}"
        upload_result = upload_file_to_s3(encrypted_file_path, BUCKET_NAME, object_name)

        if upload_result is False:
            if os.path.exists(encrypted_file_path):
                os.remove(encrypted_file_path)
            result = {
                'result': False,
                'error': 'S3 Upload error: Failed to upload image.'
            }
            return result
        else:
            result = {
                'result': True,
                'pathname': f"{AMAZON_URL}/{bucket_path}/{encrypted_file_name}",
                'original_name': encrypted_file_name,
                'mime_type': get_image_information(encrypted_file_path)['mime_type'],
                'size': get_image_information(encrypted_file_path)['size'],
                'width': image_width,
                'height': image_height,
                # For Server
                'file_name': encrypted_file_name,
                'local_path': encrypted_file_path,
                'object_name': object_name,
            }
            if os.path.exists(encrypted_file_path):
                os.remove(encrypted_file_path)
            return result
    else:
        image_height, image_width, image_channel = cv2.imread(file, cv2.IMREAD_COLOR).shape
        file_name = file.split('/')[-1]
        object_name = f"{bucket_path}/{file_name}"
        upload_result = upload_file_to_s3(file, BUCKET_NAME, object_name)

        if upload_result is False:
            if os.path.exists(file):
                os.remove(file)
            result = {
                'result': False,
                'error': 'S3 Upload error: Failed to upload image.'
            }
            return result
        else:
            result = {
                'result': True,
                'pathname': f"{AMAZON_URL}/{object_name}",
                'original_name': file_name,
                'mime_type': get_image_information(file)['mime_type'],
                'size': get_image_information(file)['size'],
                'width': image_width,
                'height': image_height,
                # For Server
                'file_name': file_name,
                'local_path': file,
                'object_name': object_name,
            }
            if os.path.exists(file):
                os.remove(file)
            return result


def trial_analysis(url, output_save_path, file_name):
    # 1. 이미지 데이터 read
    try:
        req = urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        # im = cv2.imdecode(arr, -1)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        result_dict = {
            'url': url,
            'message': f'Cannot find image({str(e)})',
            'result': False
        }
        return result_dict

    # 3. Segmentation & KeyPoints extraction.
    # 3-1. Segmentation
    cfg_seg = get_cfg()  # Segmentation 을 위한 configuration 추가
    cfg_seg.MODEL.DEVICE = 'cpu'
    cfg_seg.merge_from_list(['MODEL.DEVICE', 'cpu'])  #이 코드를 추가하시면 cpu모드로 구동하게 됩니다.
    # cfg_seg.merge_from_file("/var/www/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg_seg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95  # set threshold for this model.

    # detectron2 model zoo에서 모델 선택(다양한 모델을 사용할 수 있음)
    cfg_seg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    predictor_seg = DefaultPredictor(cfg_seg)

    try:
        outputs_seg = predictor_seg(im)
    except Exception as e:
        result_dict = {
            'message': f'Cannot open image url: {str(e)}',
            'result': False
        }

        return result_dict

    # Copy outputs_Seg and filter 'person' class only.
    person_seg = outputs_seg['instances'][outputs_seg['instances'].pred_classes == 0]

    if len(person_seg) == 1:
        # Remain only one highest probability of person.
        person_seg_trustful = person_seg[person_seg.scores == person_seg.scores.max()]

        # Remove bounding box, class label, predicting score from the image.
        person_seg_trustful.remove('pred_boxes')  # Removing bounding box.
        person_seg_trustful.remove('pred_classes')  # Removing class label.
        person_seg_trustful.remove('scores')  # Removing predicting score.

        # `Visualizer`를 이용하면 Segmentation 예측 결과를 손쉽게 그릴 수 있습니다.
        v_seg = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg_seg.DATASETS.TRAIN[0]), scale=1.2)
        v_seg = v_seg.draw_instance_predictions(person_seg_trustful.to("cpu"))
        v_seg = v_seg.get_image()[:, :, ::-1]   # v_Seg 변수에 Segmentation 결과를 저장하고, keypoints를 그리기 위해 다음 섹션에서 사용합니다.
    elif len(person_seg) > 1:
        score_gap = person_seg.scores[0] - person_seg.scores[1]  # 최대, 차대 점수 차
        if score_gap >= 0.01:
            # 사람 형태가 모두가 확실한 것은 아니라고 가정. ==> 상위 1명만 Segmentation 그린다.
            # Remain only one highest probability of person.
            person_seg_trustful = person_seg[person_seg.scores == person_seg.scores.max()]

            # Remove bounding box, class label, predicting score from the image.
            person_seg_trustful.remove('pred_boxes')  # Removing bounding box.
            person_seg_trustful.remove('pred_classes')  # Removing class label.
            person_seg_trustful.remove('scores')  # Removing predicting score.

            # `Visualizer`를 이용하면 Segmentation 예측 결과를 손쉽게 그릴 수 있습니다.
            v_seg = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg_seg.DATASETS.TRAIN[0]), scale=1.2)
            v_seg = v_seg.draw_instance_predictions(person_seg_trustful.to("cpu"))
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
        return result_dict

    # #################################################################################
    # 3-2. KeyPoints detection
    cfg_key = get_cfg()  # Keypoints용 configuration 추가
    cfg_key.MODEL.DEVICE = 'cpu'
    cfg_key.merge_from_list(['MODEL.DEVICE', 'cpu'])    # 이 코드를 추가하시면 cpu모드로 구동하게 됩니다.
    # cfg_key.merge_from_file("/var/www/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml") # cd detectron2 로 경로 변경한 경우 이것으로 실행.
    cfg_key.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg_key.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95  # set threshold for this model

    # keypoint 모델 선택(다양한 모델을 사용할 수 있음).
    cfg_key.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
    predictor_key = DefaultPredictor(cfg_key)
    outputs_key = predictor_key(v_seg)

    # Copy outputs_Seg and filter keypoints of the 'person' object who has highest probability. #Keypoints are only for person, so class filtering doesn't needed.
    try:
        person_key = outputs_key['instances'][outputs_key['instances'].scores == outputs_key['instances'].scores.max()]
    except:
        # if len(outputs_Key['instances'].pred_classes) == 0  ==>  Error occurs at person_Key because no keypoints to draw detected.
        result_dict = {
            'message': 'No person detected.',
            'result': False
        }

        return result_dict

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
    if howManyKeyPoints < 17:
        result_dict = {
            'message': 'Bad pose: Unable to detect the whole body joints.',
            'result': False
        }
        return result_dict
    elif howManyKeyPoints == 17:
        pass

    if howManyPerson > 1:
        result_dict = {
            'message': 'Too many people.',
            'result': False
        }

        return result_dict
    elif howManyPerson == 0:
        result_dict = {
            'message': 'No person detected.',
            'result': False
        }
        return result_dict
    else:
        pass

    """
    "key points": [
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

    # 모든 좌표는 torch.tensor 이므로 float 형으로 변환해 주어야 함.
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

    # torch.tensor 를 float 으로 변환 & 소수점 둘째 자리 까지만 계산
    head_width = abs(left_ear_x - right_ear_x)  # type: torch.tensor
    head_width = head_width.item()  # 형변환: torch.tensor -> float
    head_width = round(head_width, 2)

    shoulder_width = abs(left_shoulder_x - right_shoulder_x)  # type: torch.tensor
    shoulder_width = shoulder_width.item()  # 형변환: torch.tensor -> float
    shoulder_width = round(shoulder_width, 2)

    # try:
    hip_width = abs(left_hip_x - right_hip_x)  # type: torch.tensor
    hip_width = hip_width.item()  # 형변환: torch.tensor -> float
    hip_width = round(hip_width, 2)

    try:
        shoulder_head = shoulder_width / head_width
        shoulder_head = round(shoulder_head, 2)
        hip_head = hip_width / head_width
        hip_head = round(hip_head, 2)
    except:
        result_dict = {
            'message': 'Bad pose. Head or hip width is 0',
            'result': False
        }
        return result_dict

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
    shoulder_center_to_ankle_center = shoulder_center_to_hip_center + hip_center_to_ankle_center

    # output = cv2.resize(v_key.get_image()[:, :, ::-1], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    output = v_key.get_image()[:, :, ::-1]
    local_image_path = f'{output_save_path}/{file_name}'
    cv2.imwrite(local_image_path, output)

    if shoulder_head == 0 or hip_head == 0 or shoulder_width == 0 or hip_width == 0 or nose_to_shoulder_center == 0 or shoulder_center_to_hip_center == 0 or hip_center_to_ankle_center == 0 or whole_body_length == 0 or shoulder_center_to_ankle_center == 0:
        result_dict = {
            'message': 'Bad pose. Invalid body length.',
            'result': False
        }
        return result_dict
    else:
        pass

    # 8. 써클인 서버에 분석 결과 이미지 주소 + 수치 데이터 + Input 시 받은 회원 정보 전송
    result_dict = {
        'result': True,
        'pathname': local_image_path,
        'head_width': float(head_width),
        'shoulder_ratio': float(shoulder_head),
        'hip_ratio': float(hip_head),
        'shoulder_width': float(shoulder_width),
        'hip_width': float(hip_width),
        'nose_to_shoulder_center': float(nose_to_shoulder_center),  # 코~가슴 어깨 중앙 = h1
        'shoulder_center_to_hip_center': float(shoulder_center_to_hip_center),  # 어깨 중앙 ~ 골반 중앙 = h2
        'hip_center_to_ankle_center': float(hip_center_to_ankle_center),  # 골반 중앙 ~ 발목 중앙 = h3
        'whole_body_length': float(whole_body_length),  # 전신 길이 = h1 + h2 + h3
        'shoulder_center_to_ankle_center': float(shoulder_center_to_ankle_center)  # 상체 + 하체 길이 = h2 + h3
    }

    return result_dict


def find_similar_entertainer(cursor, gender: str, user_height: float, user_weight: float, user_hip_ratio: float, user_shoulder_ratio: float):
    sql = f"""
        SELECT 
                name, gender, height, weight, head_width,
                shoulder_width, shoulder_ratio, hip_width, hip_ratio,
                nose_to_shoulder_center, shoulder_center_to_hip_center,
                hip_center_to_ankle_center, shoulder_center_to_ankle_center,
                whole_body_length
        FROM
            bodylab_trials
        WHERE
            is_standard = 1
        AND gender='{gender}'"""
    cursor.execute(sql)
    entertainers = cursor.fetchall()

    # Convert tuple as dataframe.
    col = ['name', 'gender', 'height', 'weight', 'head_width',
           'shoulder_width', 'shoulder_ratio', 'hip_width', 'hip_ratio',
           'nose_to_shoulder_center', 'shoulder_center_to_hip_center',
           'hip_center_to_ankle_center', 'shoulder_center_to_ankle_center',
           'whole_body_length']
    # col = ['name', 'gender', 'height', 'shoulder_ratio', 'hip_ratio']
    df = pd.DataFrame(data=entertainers, columns=col)

    # 1안. 키, 체중을 기준으로 오름차순 정렬한 후 유사한 수치인 사람들을 찾기
    my_specification = {'name': 'circlin', 'height': user_height, 'weight': user_weight, 'hip_ratio': user_hip_ratio, 'shoulder_ratio': user_shoulder_ratio}
    specification_list = []
    for entertainer in entertainers:
        ent_specification = {'name': entertainer[0], 'height': entertainer[2], 'weight': entertainer[3], 'shoulder_ratio': entertainer[6],
                             'hip_ratio': entertainer[8]}
        specification_list.append(ent_specification)
    specification_list.append(my_specification)

    specification_list = sorted(specification_list, key=lambda x: (x['height'], x['weight']))

    my_index = 0
    for index, dictionary in enumerate(specification_list):
        if dictionary['name'] == 'circlin':
            my_index = index
            break

    # 키가 가장 비슷한 사람들 4명을 구하고 본인과 한 리스트에 배열하기
    if my_index == 0:
        compare_target = specification_list[:5]
    elif my_index == 1:
        compare_target = specification_list[0:5]
    elif my_index == len(specification_list) - 2:
        compare_target = specification_list[-5:]
    elif my_index == len(specification_list) - 1:
        compare_target = specification_list[-5:]  # or -5:-2 ?
    else:
        compare_target = specification_list[my_index - 2:my_index + 3]

    # closest_index = euclidean_list.index(min(euclidean_list))
    # # similar_entertainer = json.loads(df.to_json(orient='records'))[closest_index]


    # # 2안. 좌표평면에 mapping 후 유클리드 거리 유사도 측정
    # # 좌표평면에 mapping 후 유클리드 거리 유사도 측정
    user = np.array([my_specification['height'], my_specification['shoulder_ratio'], my_specification['hip_ratio']])
    vectorized_entertainers = []
    for target in compare_target:
        if target['name'] == 'circlin':
            pass
        else:
            np_ent = np.asarray([target['height'], target['shoulder_ratio'], target['hip_ratio']])
            np_ent = np.array([float(x) for x in np_ent])
            vectorized_entertainers.append(np_ent)

    data_normalizer = Normalizer(norm='l2').fit(vectorized_entertainers)  # L2 normalization for euclidean distance.
    entertainer_normalized = data_normalizer.transform(vectorized_entertainers)
    user_normalized = data_normalizer.transform(user.reshape(1, -1))

    euclidean_list = []
    for normalized in entertainer_normalized:
        euclidean_list.append(euclidean_distances(user_normalized.reshape(1, -1), normalized.reshape(1, -1)))

    closest_index = euclidean_list.index(min(euclidean_list))
    # similar_entertainer = json.loads(df.to_json(orient='records'))[closest_index]
    similar_entertainer = compare_target[closest_index]

    # 연예인 정보로 추천 프로그램 찾기
    sql = f"""
        SELECT
               p.id,
               p.title AS program,
               c.name AS coach,
               e.title AS exercise,
               f.pathname AS program_thumbnail,
               (SELECT JSON_ARRAYAGG(pathname) FROM files WHERE files.original_file_id = p.thumbnail_id) AS program_thumbnails
        FROM
             bodylab_trials bt
        INNER JOIN
                 bodylab_trial_programs btp
            ON 
                bt.id = btp.bodylab_trial_id
        INNER JOIN
                 programs p
            ON 
                btp.program_id = p.id
        INNER JOIN
            coaches c
            ON 
                p.coach_id = c.id
        INNER JOIN
            program_exercises pe
            ON
                p.id = pe.program_id
        INNER JOIN
            exercises e
            ON
                e.id = pe.exercise_id
        INNER JOIN files f
            ON
                f.id = p.thumbnail_id
        WHERE bt.name='{similar_entertainer["name"]}'"""
    cursor.execute(sql)
    program = cursor.fetchall()

    col_program = ['id', 'title',
                   'coach', 'exercise',
                   'thumbnail', 'thumbnails']

    df_program = pd.DataFrame(data=program, columns=col_program)
    df_program['thumbnails'] = df_program['thumbnails'].apply(lambda x: json.loads(x))
    recommended_programs = json.loads(df_program.to_json(orient='records'))
    return similar_entertainer, recommended_programs
# endregion


# region upload file
def upload_file_to_s3(file_name, bucket_name, object_name):
    s3_client = boto3.client('s3')

    try:
        s3_client.upload_file(file_name, bucket_name, object_name)
    except Exception as e:
        return False

    return True
# endregion


# region file type
def heic_to_jpg(path):
    heif_file = pyheif.read(path)
    new_image = Image.frombytes(
        heif_file.mode,
        heif_file.size,
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    new_path = f"{path.split('.')[0]}.jpg"
    new_image.save(new_path, "JPEG")
    if os.path.exists(path):
        os.remove(path)
    return new_path, 'jpg'


def get_image_information(path):
    result = {
        'mime_type': filetype.guess(path).mime,
        'size': int(os.path.getsize(path))
    }
    return result


def generate_resized_image(LOCAL_SAVE_PATH, user_id, now, extension, original_image_path):
    # file_name = user_id + now + extension
    # local_image_path = BODY_IMAGE_INPUT_PATH, user_id, file_name
    # 새롭게 생성되는 resized file들은 file_name = user_id + now + {width}w + extension
    original_image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
    height, width, channel = original_image.shape
    new_widths = [1080, 750, 640, 480, 320, 240, 150]
    new_image_list = []
    for new_width in new_widths:
        new_height = int(new_width * height / width)

        if new_width > width:  # 확대
            resized_image = cv2.resize(original_image,
                                       dsize=(new_width, new_height),
                                       interpolation=cv2.INTER_LINEAR)
        else:  # 축소(<) or 유지(=)
            resized_image = cv2.resize(original_image,
                                       dsize=(new_width, new_height),
                                       interpolation=cv2.INTER_AREA)
        original_name = f'bodylab_body_output_{user_id}_{now}.{extension}'
        file_name = f'bodylab_body_output_{user_id}_{now}_{new_width}w.{extension}'
        resized_image_path = f'{LOCAL_SAVE_PATH}/{user_id}/{file_name}'

        object_name = f'{BUCKET_IMAGE_PATH_BODY_OUTPUT}/{user_id}/{file_name}'

        cv2.imwrite(resized_image_path, resized_image)
        image_dict = {
            # For DB when INSERT
            'pathname': f'{AMAZON_URL}/{object_name}',
            'original_name': original_name,
            'mime_type': get_image_information(resized_image_path)['mime_type'],
            'size': get_image_information(resized_image_path)['size'],
            'width': new_width,
            'height': new_height,
            # For Server
            'file_name': file_name,
            'local_path': resized_image_path,
            'object_name': object_name,
        }
        new_image_list.append(image_dict)
    return new_image_list
# endregion


# 본 코드의 이미지 처리는 CPU로 처리함(GPU 및 CUDA가 이 VM에 설치되어있지 않음).
def analysis(url, user_id):
    start_time = time.time()

    # 1. 이미지 데이터 read
    try:
        req = urlopen(url)
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        # im = cv2.imdecode(arr, -1)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        result_dict = {
            'url': url,
            'user_id': user_id,
            'message': f'Cannot find image({str(e)})',
            'result': False
        }
        return json.dumps(result_dict)

    # 3. Segmentation & KeyPoints extraction.
    # 3-1. Segmentation
    cfg_seg = get_cfg()  # Segmentation 을 위한 configuration 추가
    cfg_seg.MODEL.DEVICE = 'cpu'
    cfg_seg.merge_from_list(['MODEL.DEVICE', 'cpu'])  #이 코드를 추가하시면 cpu모드로 구동하게 됩니다.
    # cfg_seg.merge_from_file("/var/www/detectron2/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg_seg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg_seg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95  # set threshold for this model.

    # detectron2 model zoo에서 모델 선택(다양한 모델을 사용할 수 있음)
    cfg_seg.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x/137849600/model_final_f10217.pkl"
    predictor_seg = DefaultPredictor(cfg_seg)

    try:
        outputs_seg = predictor_seg(im)
    except Exception as e:
        result_dict = {
            'message': f'Cannot open image url: {str(e)}',
            'result': False
        }
        return json.dumps(result_dict)

    # Copy outputs_Seg and filter 'person' class only.
    person_seg = outputs_seg['instances'][outputs_seg['instances'].pred_classes == 0]

    if len(person_seg) == 1:
        person_seg_trustful = person_seg[person_seg.scores == person_seg.scores.max()]
        # Remove bounding box, class label, predicting score from the image.
        person_seg_trustful.remove('pred_boxes')  # Removing bounding box.
        person_seg_trustful.remove('pred_classes')  # Removing class label.
        person_seg_trustful.remove('scores')  # Removing predicting score.

        # `Visualizer`를 이용하면 Segmentation 예측 결과를 손쉽게 그릴 수 있습니다.
        v_seg = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg_seg.DATASETS.TRAIN[0]), scale=1.2)
        v_seg = v_seg.draw_instance_predictions(person_seg_trustful.to("cpu"))
        v_seg = v_seg.get_image()[:, :, ::-1]   # v_Seg 변수에 Segmentation 결과를 저장하고, keypoints를 그리기 위해 다음 섹션에서 사용합니다.
    elif len(person_seg) > 1:
        score_gap = person_seg.scores[0] - person_seg.scores[1]
        if score_gap >= 0.01:
            # 사람 형태가 모두가 확실한 것은 아니라고 가정. ==> 상위 1명만 Segmentation 그린다.
            # Remain only one highest probability of person.
            person_seg_trustful = person_seg[person_seg.scores == person_seg.scores.max()]

            # Remove bounding box, class label, predicting score from the image.
            person_seg_trustful.remove('pred_boxes')  # Removing bounding box.
            person_seg_trustful.remove('pred_classes')  # Removing class label.
            person_seg_trustful.remove('scores')  # Removing predicting score.

            # `Visualizer`를 이용하면 Segmentation 예측 결과를 손쉽게 그릴 수 있습니다.
            v_seg = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg_seg.DATASETS.TRAIN[0]), scale=1.2)
            v_seg = v_seg.draw_instance_predictions(person_seg_trustful.to("cpu"))
            v_seg = v_seg.get_image()[:, :, ::-1]  # v_Seg 변수에 Segmentation 결과를 저장하고, keypoints를 그리기 위해 다음 섹션에서 사용합니다.
        else:
            # 사람 간 형태가 모두 확실하다고 가정. ==> 검출된 모든 사람에 Segmentation을 그린다.
            # Remove bounding box, class label, predicting score from the image.
            person_seg.remove('pred_boxes')  # Removing bounding box.
            person_seg.remove('pred_classes')  # Removing class label.
            person_seg.remove('scores')  # Removing predicting score.

            # `Visualizer`를 이용하면 Segmentation 예측 결과를 손쉽게 그릴 수 있습니다.
            v_seg = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg_seg.DATASETS.TRAIN[0]), scale=1.2)
            v_seg = v_seg.draw_instance_predictions(person_seg.to("cpu"))
            v_seg = v_seg.get_image()[:, :, ::-1]  # v_Seg 변수에 Segmentation 결과를 저장하고, keypoints를 그리기 위해 다음 섹션에서 사용합니다.
    else:
        # if len(person_Seg) == 0  ==>  Error occurs at person_Seg_trustful.
        result_dict = {
            'message': 'No person detected.',
            'result': False
        }
        return json.dumps(result_dict)

    # #################################################################################
    # 3-2. KeyPoints detection
    cfg_key = get_cfg()  # Keypoints용 configuration 추가
    cfg_key.MODEL.DEVICE = 'cpu'
    cfg_key.merge_from_list(['MODEL.DEVICE', 'cpu'])    # 이 코드를 추가하시면 cpu모드로 구동하게 됩니다.
    # cfg_key.merge_from_file("/var/www/detectron2/configs/COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml") # cd detectron2 로 경로 변경한 경우 이것으로 실행.
    cfg_key.merge_from_file(model_zoo.get_config_file("COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml"))
    cfg_key.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.95  # set threshold for this model

    # keypoint 모델 선택(다양한 모델을 사용할 수 있음).
    cfg_key.MODEL.WEIGHTS = "detectron2://COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x/137849621/model_final_a6e10b.pkl"
    predictor_key = DefaultPredictor(cfg_key)
    outputs_key = predictor_key(v_seg)

    # Copy outputs_Seg and filter keypoints of the 'person' object who has highest probability. #Keypoints are only for person, so class filtering doesn't needed.
    try:
        person_key = outputs_key['instances'][outputs_key['instances'].scores == outputs_key['instances'].scores.max()]
    except:
        result_dict = {
            'message': 'No person detected.',
            'result': False
        }
        return json.dumps(result_dict)

    # Removing bounding box, class label, predicting score.
    person_key.remove('pred_boxes')
    person_key.remove('pred_classes')
    person_key.remove('scores')

    # `Visualizer`를 이용하면 Keypoint 예측 결과를 손쉽게 그릴 수 있습니다.
    v_key = Visualizer(v_seg[:, :, ::-1], MetadataCatalog.get(cfg_key.DATASETS.TRAIN[0]), scale=1.2)
    v_key = v_key.draw_instance_predictions(person_key.to("cpu"))

    # 4. Keypoint, Segmentation으로부터 신체 비율 & 부위별 면적 계산하기
    keypoints_whole = person_key.get('pred_keypoints')  # Every keypoints on an image
    keypoints_person = keypoints_whole[0]  # keypoints of one highest probability person
    howManyPerson = len(keypoints_whole)  # Number of keypoint tensors(Number of 2-D tensors)(1 2-D tensor is consists of 17 1-D tensors.)
    howManyKeyPoints = len(keypoints_person)  # Number of keypoints of one highest probability person
    if howManyKeyPoints < 17:
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
        return json.dumps(result_dict)
    elif howManyPerson == 0:
        result_dict = {
            'message': 'No person detected.',
            'result': False
        }
        return json.dumps(result_dict)
    else:
        pass

    """
    "key points": [
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

    # torch.tensor 를 float 으로 변환 & 소수점 둘째 자리 까지만 계산
    head_width = abs(left_ear_x - right_ear_x)  # type: torch.tensor
    head_width = head_width.item()  # 형변환: torch.tensor -> float
    head_width = round(head_width, 2)

    shoulder_width = abs(left_shoulder_x - right_shoulder_x)  # type: torch.tensor
    shoulder_width = shoulder_width.item()  # 형변환: torch.tensor -> float
    shoulder_width = round(shoulder_width, 2)

    # try:
    hip_width = abs(left_hip_x - right_hip_x)  # type: torch.tensor
    hip_width = hip_width.item()  # 형변환: torch.tensor -> float
    hip_width = round(hip_width, 2)

    try:
        shoulder_head = shoulder_width / head_width
        shoulder_head = round(shoulder_head, 2)
        hip_head = hip_width / head_width
        hip_head = round(hip_head, 2)
    except:
        result_dict = {
            'message': 'Bad pose. Head or hip width is 0',
            'result': False
        }
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
    shoulder_center_to_ankle_center = shoulder_center_to_hip_center + hip_center_to_ankle_center


    now = datetime.now().strftime('%Y%m%d%H%M%S')

    # output = cv2.resize(v_key.get_image()[:, :, ::-1], dsize=(0, 0), fx=0.3, fy=0.3, interpolation=cv2.INTER_LINEAR)
    output = v_key.get_image()[:, :, ::-1]
    if str(user_id) not in os.listdir(f"{LOCAL_SAVE_PATH_BODY_OUTPUT}"):
        os.makedirs(f"{LOCAL_SAVE_PATH_BODY_OUTPUT}/{user_id}")
    extension = url.split('.')[-1]
    file_name = f'bodylab_body_output_{user_id}_{now}.{extension}'
    local_image_path = f'{LOCAL_SAVE_PATH_BODY_OUTPUT}/{user_id}/{file_name}'
    cv2.imwrite(local_image_path, output)

    """output 가로 리사이징 후 저장"""
    body_image_height, body_image_width, body_image_channel = cv2.imread(local_image_path, cv2.IMREAD_COLOR).shape

    # 7. S3에 분석 결과 이미지(output_path) 원본 저장
    object_name = f"{BUCKET_IMAGE_PATH_BODY_OUTPUT}/{user_id}/{file_name}"
    if upload_file_to_s3(local_image_path, BUCKET_NAME, object_name) is True:
        pass
    else:
        result_dict = {
            'message': f'Error while upload output image into S3 for original image.',
            'result': False
        }
        return json.dumps(result_dict)
    s3_path_body_output = f"{AMAZON_URL}/{object_name}"
    body_output_image_dict = {
        'pathname': s3_path_body_output,
        'original_name': file_name,
        'mime_type': get_image_information(local_image_path)['mime_type'],
        'size': get_image_information(local_image_path)['size'],
        'width': body_image_width,
        'height': body_image_height,
        # For Server
        'file_name': file_name,
        'local_path': local_image_path,
        'object_name': object_name,
    }

    resized_body_output_image_list = generate_resized_image(LOCAL_SAVE_PATH_BODY_OUTPUT, user_id, now, extension, local_image_path)
    for resized_image in resized_body_output_image_list:
        upload_result = upload_file_to_s3(resized_image['local_path'], BUCKET_NAME, resized_image['object_name'])
        if upload_result is True:
            pass
        else:
            result_dict = {
                'message': f'Failed to upload body image into S3({upload_result})',
                'result': False
            }
            return json.dumps(result_dict, ensure_ascii=False)
        if os.path.exists(resized_image['local_path']):
            os.remove(resized_image['local_path'])
        if os.path.exists(local_image_path):
            os.remove(local_image_path)

    if shoulder_head == 0 or hip_head == 0 or shoulder_width == 0 or hip_width == 0 or nose_to_shoulder_center == 0 or shoulder_center_to_hip_center == 0 or hip_center_to_ankle_center == 0 or whole_body_length == 0 or shoulder_center_to_ankle_center == 0:
        result_dict = {
            'message': 'Bad pose. Invalid body length.',
            'result': False
        }
        return json.dumps(result_dict, ensure_ascii=False)
    else:
        pass

    # 8. 써클인 서버에 분석 결과 이미지 주소 + 수치 데이터 + Input 시 받은 회원 정보 전송
    result_dict = {
        'result': True,
        'original_img_url': url,
        'output_url': s3_path_body_output,
        'shoulder_ratio': float(shoulder_head),
        'hip_ratio': float(hip_head),
        'shoulder_width': float(shoulder_width),
        'hip_width': float(hip_width),
        'nose_to_shoulder_center': float(nose_to_shoulder_center),  # 코~가슴 어깨 중앙 = h1
        'shoulder_center_to_hip_center': float(shoulder_center_to_hip_center),  # 어깨 중앙 ~ 골반 중앙 = h2
        'hip_center_to_ankle_center': float(hip_center_to_ankle_center),  # 골반 중앙 ~ 발목 중앙 = h3
        'whole_body_length': float(whole_body_length),  # 전신 길이 = h1 + h2 + h3
        'shoulder_center_to_ankle_center': float(shoulder_center_to_ankle_center),  # 상체 + 하체 길이 = h2 + h3
        'body_output_image_dict': body_output_image_dict,
        'resized_body_output_image_list': resized_body_output_image_list,
        'message': 'Done analysis'  # 성공 혹은 실패 여부
    }
    return json.dumps(result_dict, ensure_ascii=False)
