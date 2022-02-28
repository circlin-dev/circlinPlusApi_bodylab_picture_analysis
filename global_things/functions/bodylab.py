from global_things.constants import SLACK_NOTIFICATION_WEBHOOK, AMAZON_URL, BUCKET_NAME, BUCKET_IMAGE_PATH_BODY_OUTPUT
import boto3
import cv2
import filetype
import json
import mimetypes
import os
from PIL import Image
import pyheif
import requests
import shutil
import uuid


# region bodylab
def upload_file_to_s3(file_name, bucket_name, object_name):
    s3_client = boto3.client('s3')

    try:
        s3_client.upload_file(file_name, bucket_name, object_name)
    except Exception as e:
        return False

    return True


def validate_and_save_to_s3(save_path, bucket_path, file):
    unique_name = uuid.uuid4()
    invalid_mimes = ['heic', 'HEIC', 'heif', 'HEIF']

    mime = get_image_information(file)['mime_type'].split('/')
    if mime[0] != 'image':
        result = {
            'result': False,
            'error': 'Invalid file type(Requested file is not an image).'
        }
        return result

    if mime[1] in invalid_mimes:
        new_secure_file, extension = heic_to_jpg(file)

        # if os.path.exists(new_secure_file):
        #     shutil.move(new_secure_file, save_path)
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
        # if os.path.exists(file):
        #     shutil.move(file, save_path)
        moved_file = f"{save_path}/{file}"
        extension = file.split('.')[-1]
        encrypted_file_name = f"{unique_name}.{extension}"
        encrypted_file_path = f"{save_path}/{encrypted_file_name}"
        if os.path.exists(moved_file):
            os.rename(moved_file, encrypted_file_path)

    image_height, image_width, image_channel = cv2.imread(encrypted_file_path, cv2.IMREAD_COLOR)
    object_name = f"{bucket_path}/{encrypted_file_name}"
    upload_result = upload_file_to_s3(encrypted_file_name, BUCKET_NAME, object_name)

    if upload_result is False:
        if os.path.exists(encrypted_file_path):
            os.remove(encrypted_file_path)
        result = {
            'result': False,
            'error': 'S3 Upload error: Failed to upload input image.'
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


def get_image_information(path):
  result = {
    'mime_type': mimetypes.guess_type(path)[0],
    'size': int(os.path.getsize(path))
  }
  return result
# endregion
