from global_things.constants import SLACK_NOTIFICATION_WEBHOOK, AMAZON_URL, BUCKET_IMAGE_PATH_BODY_OUTPUT
import boto3
import cv2
import json
import mimetypes
import os
import requests

# region Slack error notification
def slack_error_notification(user_ip: str='', user_id: str='', nickname: str='', api: str='', error_log: str=''):
  if user_ip == '':
    user_ip = "Server error"

  send_notification_request = requests.post(
    SLACK_NOTIFICATION_WEBHOOK,
    json.dumps({
      "channel": "#circlin-members-log",
      "username": "써클인 멤버스 - python",
      "text": f"*써클인 멤버스(python - 이미지 분석)에서 오류가 발생했습니다.* \n \
사용자 IP: `{user_ip}` \n \
닉네임 (ID): `{nickname}({user_id})`\n \
API URL: `{api}` \n \
```{error_log}```",
      "icon_url": "https://www.circlin.co.kr/new/assets/favicon/apple-icon-180x180.png"
    }
))

  return send_notification_request
# endregion


# region bodylab
def upload_output_to_s3(file_name, bucket_name, object_name):
  s3_client = boto3.client('s3')

  try:
    s3_client.upload_file(file_name, bucket_name, object_name)
  except Exception as e:
    return e

  return True


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
