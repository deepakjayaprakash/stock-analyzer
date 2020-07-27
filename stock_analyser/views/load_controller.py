from datetime import datetime
import datetime as dt

import pandas as pd
import quandl as quandl
from django.http import JsonResponse

from stock_analyser import settings
from stock_analyser.constants import constants
from stock_analyser.database.database_ops import create_connection_cursor
from stock_analyser.helper.quandl_helper import get_quandl_data


# Test API for getting quandl data
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
# API to load company time series, from and to dates hardcoded
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


# given company id, get and update data from company's last updated date to today
# important API used in watchlist also
def update_company_data(request, id):
    sql = "select symbol, quandl_code, last_updated_at from companies where id = %d" % (id)
    company_data = pd.read_sql(sql, settings.DATABASE_URL)
    quandl_code = "BSE/" + company_data['quandl_code'].get(0)
    last_updated_at = company_data['last_updated_at'].get(0)
    from_date = last_updated_at + dt.timedelta(days=1)
    from_date = str(from_date.date())
    to_date = str(datetime.today().date())
    print("quandl_code : ", quandl_code, ", from date: ", from_date, ", to_date", to_date)

    load_time_series_into_table(company_data, from_date, id, quandl_code, to_date)
    response = {}
    return JsonResponse(response)


# given company id, from date and date, get and update data
def update_company_data_with_date(request, id):
    sql = "select symbol, quandl_code from companies where id = %d" % (id)
    company_data = pd.read_sql(sql, settings.DATABASE_URL)
    quandl_code = "BSE/" + company_data['quandl_code'].get(0)
    from_date = datetime.strptime(request.GET.get('fromDate'), "%Y-%m-%d").date()
    from_date = str(from_date)
    to_date = datetime.strptime(request.GET.get('toDate'), "%Y-%m-%d").date()
    to_date = str(to_date)
    print("quandl_code : ", quandl_code, ", from date: ", from_date, ", to_date", to_date)

    load_time_series_into_table(company_data, from_date, id, quandl_code, to_date)
    response = {}
    return JsonResponse(response)


def load_time_series_into_table(company_data, from_date, id, quandl_code, to_date):
    quandl_response = get_quandl_data(quandl_code, from_date, to_date)
    print(quandl_response.head())
    connection, cursor = create_connection_cursor()
    update = True
    for i, row in quandl_response.iterrows():
        try:
            sql = get_insert_time_series_sql(company_data, i, id, row)
            cursor.execute(sql)
        except Exception as e:
            print("failed to insert this: ", tuple(row), "exception: ", e)
            update = False
    if update:
        print("updated successfully")
        sql = "update companies set last_updated_at = '%s'" % (to_date)
        cursor.execute(sql)
    connection.commit()


def load_watchlist_by_id(request, id):
    sql = "select company_ids from watchlist where id = %d" % (id)
    company_ids = pd.read_sql(sql, settings.DATABASE_URL)
    company_id_list = company_ids.split['company_ids'](",")
    print(company_id_list)
    # update_company_data(id)
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
