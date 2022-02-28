import pymysql
from config.database import DB_CONFIG


def login_to_db():
    conn = pymysql.connect(
        user=DB_CONFIG['user'],
        passwd=DB_CONFIG['password'],
        host=DB_CONFIG['host'],
        db=DB_CONFIG['database'],
        charset=DB_CONFIG['charset'])

    return conn
