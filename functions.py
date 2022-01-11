import requests
import json

# region Slack error notification
def slack_error_notification(user_ip: str='', user_id: str='', nickname: str='', api: str='', error_log: str=''):
  if user_ip == '':
    user_ip = "Server error"

  send_notification_request = requests.post(
    "https://hooks.slack.com/services/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?",
    json.dumps({
      "channel": "#circlin-members-log",
      "username": "써클인 멤버스 - python",
      "text": f"*써클인 멤버스(python)에서 오류가 발생했습니다.* \n \
사용자 IP: `{user_ip}` \n \
닉네임 (ID): `{nickname}({user_id})`\n \
API URL: `{api}` \n \
```{error_log}```",
      "icon_url": "https://www.circlin.co.kr/new/assets/favicon/apple-icon-180x180.png"
    }
  ))

  return send_notification_request
# endregion