from global_things.functions.general import login_to_db
from global_things.functions.bodylab import upload_file_to_s3, validate_and_save_to_s3
from global_things.functions.slack import slack_error_notification
from global_things.constants import BUCKET_NAME, BUCKET_IMAGE_PATH_BODY_TRIAL_INPUT, BUCKET_IMAGE_PATH_BODY_TRIAL_OUTPUT, LOCAL_SAVE_PATH_BODY_TRIAL_INPUT, LOCAL_SAVE_PATH_BODY_TRIAL_OUTPUT
from bodylab_picture_analyze import analysis, trial_analysis
from flask import Flask, request
from flask_cors import CORS
import json
import logging
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)  # For Cross-Domain problem

APP_ROOT = "/home/ubuntu/circlinMembersApi_python/bodylab_picture_analysis"
logging.basicConfig(filename=f'{APP_ROOT}/execution_log.log', filemode='a+', format=' [%(filename)s:%(lineno)s:%(funcName)s()]- %(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
gunicorn_logger = logging.getLogger('gunicorn.error')
app.logger.handlers = gunicorn_logger.handlers
app.logger.setLevel(gunicorn_logger.level)


@app.route('/testing')
def testing():
    return "Hello, Circlin!!!"


@app.route('/trial-analysis', methods=['POST'])
def free_trial():
    parameters = request.form.to_dict()
    user_height = float(parameters['height'])
    user_weight = float(parameters['weight'])
    body_image = request.files.to_dict()['body_image']
    secure_file = secure_filename(body_image.filename)
    body_image.save(secure_file)

    save_input_image = validate_and_save_to_s3(LOCAL_SAVE_PATH_BODY_TRIAL_INPUT, BUCKET_IMAGE_PATH_BODY_TRIAL_INPUT, secure_file)
    if save_input_image['result'] is False:
        result = {'result': False, 'error': save_input_image['error']}
        return json.dumps(result, ensure_ascii=False), 400

    input_url = save_input_image['pathname']
    input_original_name = save_input_image['original_name']

    analysis_result = trial_analysis(input_url, LOCAL_SAVE_PATH_BODY_TRIAL_OUTPUT, input_original_name)
    if analysis_result['result'] is False:
        result = {'result': False, 'error': analysis_result['message']}
        return json.dumps(result, ensure_ascii=False), 400

    output_path = analysis_result['pathname']
    save_output_image = validate_and_save_to_s3(LOCAL_SAVE_PATH_BODY_TRIAL_OUTPUT, BUCKET_IMAGE_PATH_BODY_TRIAL_OUTPUT, output_path)
    if save_output_image['result'] is False:
        result = {'result': False, 'error': save_output_image['error']}
        return json.dumps(result, ensure_ascii=False), 500
    output_url = save_output_image['pathname']

    connection = login_to_db()
    cursor = connection.cursor()

    file_ids = []
    for data in [save_input_image, save_output_image]:
        sql = f"""
            INSERT INTO 
                files(pathname, original_name, mime_type, size, width, height)
            VALUES
                (%s, %s, %s, %s, %s, %s)"""
        values = (data['pathname'], data['original_name'],
                  data['mime_type'], data['size'],
                  data['width'], data['height'])
        cursor.execute(sql, values)
        connection.commit()
        file_ids.append(int(cursor.lastrowid))

    sql = f"""
        INSERT INTO 
            bodylab_trials(
                is_standard, file_id_body_input, file_id_body_output, height, weight, 
                shoulder_width, shoulder_ratio, hip_width, hip_ratio, nose_to_shoulder_center, 
                shoulder_center_to_hip_center, hip_center_to_ankle_center, shoulder_center_to_ankle_center, whole_body_length
            )
        VALUES
            (
                False, {file_ids[0]}, {file_ids[1]}, {user_height}, {user_weight}, 
                {analysis_result['shoulder_width']}, {analysis_result['shoulder_ratio']}, {analysis_result['hip_width']}, {analysis_result['hip_ratio']}, {analysis_result['nose_to_shoulder_center']}, 
                {analysis_result['shoulder_center_to_hip_center']}, {analysis_result['hip_center_to_ankle_center']}, {analysis_result['shoulder_center_to_ankle_center']}, {analysis_result['whole_body_length']}
            )"""
    cursor.execute(sql)
    connection.commit()
    connection.close()

    result_dict = {
        "result": True,
        "body_image_analysis": {
            "body_input_url": input_url,
            "body_output_url": output_url,
            "user": {
                "height": user_height,
                "weight": user_weight,
                "shoulder_width": analysis_result['shoulder_width'],
                "shoulder_ratio": analysis_result['shoulder_ratio'],
                "hip_width": analysis_result['hip_width'],
                "hip_ratio": analysis_result['hip_ratio'],
                "nose_to_shoulder_center": analysis_result['nose_to_shoulder_center'],
                "shoulder_center_to_hip_center": analysis_result['shoulder_center_to_hip_center'],
                "hip_center_to_ankle_center": analysis_result['hip_center_to_ankle_center'],
                "shoulder_center_to_ankle_center": analysis_result['shoulder_center_to_ankle_center'],
                "whole_body_length": analysis_result['whole_body_length']
            }
        }
    }

    return json.dumps(result_dict, ensure_ascii=False), 201


@app.route('/analysis', methods=['POST'])
def index():
    print('Accessed to test server.')
    # Input: 이미지 주소, 유저 정보(id)

    try:
        req = request.get_json()
        url = req.get('url')
        user_id = req.get('user_id')
    except Exception as e:
        slack_error_notification(api='/analysis', error_log=f"Error while handling request: {e}")
        result_dict = {
            'message': 'Error while handling request.',
            'result': False
        }
        return result_dict, 500

    result = analysis(url, user_id)
    result = json.loads(result)
    result_message = result['message']
    if result_message == 'Too many people.':
        print(result_message)
        slack_error_notification(user_id=user_id, api='/analysis', error_log=f"Error while analyzing original image '{url}': {result_message}")
        return result, 400
    elif result_message == 'Cannot find image.':
        print(result_message)
        slack_error_notification(user_id=user_id, api='/analysis', error_log=f"Error while analyzing original image '{url}': {result_message}")
        return result, 400
    elif result_message == 'No person detected.':
        print(result_message)
        slack_error_notification(user_id=user_id, api='/analysis', error_log=f"Error while analyzing original image '{url}': {result_message}")
        return result, 400
    elif result_message == 'Unacceptable file extension.':
        print(result_message)
        slack_error_notification(user_id=user_id, api='/analysis', error_log=f"Error while analyzing original image '{url}': {result_message}")
        return result, 400
    elif result_message == 'Bad pose. Unable to detect the whole body joints.':
        print(result_message)
        slack_error_notification(user_id=user_id, api='/analysis', error_log=f"Error while analyzing original image '{url}': {result_message}")
        return result, 400
    elif result_message == 'Bad pose. Head or hip width is 0.':
        print(result_message)
        slack_error_notification(user_id=user_id, api='/analysis', error_log=f"Error while analyzing original image '{url}': {result_message}")
        return result, 400
    elif result_message == 'Bad pose. Invalid body length.':
        print(result_message)
        slack_error_notification(user_id=user_id, api='/analysis', error_log=f"Error while analyzing original image '{url}': {result_message}")
        return result, 400
    elif result_message == 'Error while upload output image into S3 for original image.':
        print(result_message)
        slack_error_notification(user_id=user_id, api='/analysis', error_log=f"Error while analyzing original image '{url}': {result_message}")
        return result, 500
    else:
        return result, 200


if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=True)
