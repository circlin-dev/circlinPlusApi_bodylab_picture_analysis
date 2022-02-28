from global_things.constants import SLACK_NOTIFICATION_WEBHOOK
import json
import requests


# region Slack error notification
def slack_error_notification(user_ip: str = '', user_id: int = 0, nickname: str = '', api: str = '',
                             error_log: str = '', query: str = '', method: str = ''):
  if user_ip == '' or user_id == '':
    user_ip = "Server error"
    user_id = "Server error"

  send_notification_request = requests.post(
    SLACK_NOTIFICATION_WEBHOOK,
    json.dumps({
      "channel": "#circlin-plus-log",
      "username": "써클인 플러스 - python",
      "method": method,
      "text": f"*써클인 플러스(python)에서 오류가 발생했습니다.* \n \
사용자 IP: `{user_ip}` \n \
닉네임 (ID): `{nickname}({user_id})`\n \
API URL: `{api}` \n \
HTTP method: `{method}` \n \
```query: {query}``` \n \
```error: {error_log}```",
      "icon_url": "https://www.circlin.co.kr/new/assets/favicon/apple-icon-180x180.png"
    }, ensure_ascii=False).encode('utf-8')
  )

  return send_notification_request
# endregion