from global_things.functions.slack import slack_error_notification


class InvalidAPIUsage(Exception):
    status_code = 400

    def __init__(self,
                 user_ip: str = '',
                 user_id: int = 0,
                 nickname: str = '',
                 api: str = '',
                 error_message: str = '',
                 query: str = '',
                 method: str = '',
                 status_code=None,
                 payload=None,
                 result=False):
        super().__init__()
        self.user_ip = user_ip
        self.user_id = user_id
        self.nickname = nickname
        self.api = api
        self.error_message = error_message
        self.query = query
        self.method = method
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload
        self.result = result

    def to_dict(self):
        error = dict()
        error['user_ip'] = self.user_ip
        error['user_id'] = self.user_id
        error['nickname'] = self.nickname
        error['api'] = self.api
        error['error_message'] = self.error_message
        error['query'] = self.query
        error['method'] = self.method
        error['status_code'] = self.status_code
        error['payload'] = self.payload
        error['result'] = self.result
        slack_error_notification(self.user_ip, self.user_id, self.nickname, self.api, self.error_message, self.query, self.method)
        return error
