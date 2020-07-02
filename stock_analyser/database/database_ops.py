import pymysql

from stock_analyser import settings


def create_connection_cursor():
    # Connect to the database
    connection = create_connection()
    # create cursor
    cursor = connection.cursor()
    return connection, cursor


def create_connection():
    connection = pymysql.connect(host=settings.HOST,
                                 user=settings.USER_NAME,
                                 password=settings.PASSWORD,
                                 db=settings.DATABASE_NAME)
    return connection
