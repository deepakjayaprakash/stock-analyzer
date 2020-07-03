import pandas as pd
import quandl as quandl
from django.http import JsonResponse

from stock_analyser import settings
from stock_analyser.database.database_ops import create_connection_cursor


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


def load_company_data(request, id):
    sql = "select actual_name, symbol, quandl_code from companies where id = %d" % (id)
    print(sql)
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
            sql = "INSERT INTO `time_series` (`company_id`, `symbol`, quandl_code, open, close," \
                  "num_trades, num_shares, close_open_spread, trade_date, percentage_change) " \
                  "VALUES (%d, '%s', '%s', %.3f, %.3f, %.3f, %.3f, %.3f, '%s', %.3f)" \
                  % (id, company_data['symbol'].get(0), company_data['quandl_code'].get(0)
                     , row['Open'], row['Open'], row['No. of Trades'], row['No. of Trades'], row['Spread C-O'],
                     i, (row['Spread C-O'] / row['Open'] * 100))
            print(sql)
            cursor.execute(sql)
        except Exception as e:
            print("failed to insert this: ", tuple(row))
    connection.commit()
    response = {}
    return JsonResponse(response)
