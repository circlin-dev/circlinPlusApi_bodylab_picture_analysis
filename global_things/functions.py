def error_notification_slack(userid, nickname, api, error):
  result_dict = {
    "channel": "#circlin-members-log",
    "username": "써클인 멤버스 - python",
    "text": f"써클인 멤버스 *파이썬 서버(바디랩 사진분석)*에서 오류가 발생했습니다.\n \
           *닉네임(ID)*: `{nickname} ({userid})`\n \
           *API URL*: `{api}`\n \
            ```{error}```",
    "icon_url": "https://www.circlin.co.kr/new/assets/favicon/apple-icon-180x180.png"
  }

  return result_dict