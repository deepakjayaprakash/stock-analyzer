from datetime import datetime

import pandas as pd
import quandl as quandl
from django.http import JsonResponse

from stock_analyser import settings
from stock_analyser.constants import constants
from stock_analyser.database.database_ops import create_connection_cursor
from stock_analyser.helper.quandl_helper import get_quandl_data


def get_company_data(request, name):
    df = quandl.get("BSE/BOM500209", authtoken=settings.QUANDL_AUTH_TOKEN, start_date="2020-06-01",
                    end_date="2020-06-30")
    print(df.head())
    print(df.shape)
    print(df.columns)
    l = []
    for i, row in df.iterrows():
        item = {}
        item['Open'] = row['Open']
        item['Close'] = row['Close']
        item['No. of Shares'] = row['No. of Shares']
        item['Date'] = i
        l.append(item)
    response = {}
    response[name] = l
    return JsonResponse(response)


# id is the company row id
def load_company_data(request, id):
    sql = "select actual_name, symbol, quandl_code from companies where id = %d" % (id)
    company_data = pd.read_sql(sql, settings.DATABASE_URL)
    quandl_code = "BSE/" + company_data['quandl_code'].get(0)
    print("quandl_code : ", quandl_code)

    quandl_response = quandl.get(quandl_code, authtoken=settings.QUANDL_AUTH_TOKEN,
                                 start_date="2020-06-01",
                                 end_date="2020-06-30")
    print(quandl_response.head())
    connection, cursor = create_connection_cursor()
    for i, row in quandl_response.iterrows():
        print(row['Open'])
        try:
            sql = get_insert_time_series_sql(company_data, i, id, row)
            print(sql)
            cursor.execute(sql)
        except Exception as e:
            print("failed to insert this: ", tuple(row))
    connection.commit()
    response = {}
    return JsonResponse(response)


def update_company_data(request, id):
    sql = "select symbol, quandl_code, last_updated_at from companies where id = %d" % (id)
    company_data = pd.read_sql(sql, settings.DATABASE_URL)
    quandl_code = "BSE/" + company_data['quandl_code'].get(0)
    last_updated_at = company_data['last_updated_at'].get(0)
    from_date = str(last_updated_at.date())
    to_date = str(datetime.today().date())
    print("quandl_code : ", quandl_code, ", from date: ", from_date, ", to_date", to_date)

    quandl_response = get_quandl_data(quandl_code, from_date, to_date)
    print(quandl_response.head())
    connection, cursor = create_connection_cursor()
    for i, row in quandl_response.iterrows():
        try:
            sql = get_insert_time_series_sql(company_data, i, id, row)
            print(sql)
            cursor.execute(sql)
        except Exception as e:
            print("failed to insert this: ", tuple(row), "exception: ", e)
    connection.commit()
    response = {}
    return JsonResponse(response)


def get_insert_time_series_sql(company_data, i, id, row):
    sql = "INSERT INTO `time_series` (`company_id`, `symbol`, quandl_code, open, close," \
          "num_trades, num_shares, close_open_spread, trade_date, percentage_change) " \
          "VALUES (%d, '%s', '%s', %.3f, %.3f, %.3f, %.3f, %.3f, '%s', %.3f)" \
          % (id, company_data['symbol'].get(0), company_data['quandl_code'].get(0)
             , row[constants.OPEN], row[constants.CLOSE], row[constants.NUM_TRADES], row[constants.NUM_SHARES]
             , row[constants.SPREAD_C_O], i, (row[constants.SPREAD_C_O] / row[constants.OPEN] * 100))
    return sql
