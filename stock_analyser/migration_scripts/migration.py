import pandas as pd

from stock_analyser.database.database_ops import create_connection_cursor


def load_quandl_data():
    df = pd.read_csv("/home/deepak/Downloads/stock_data/BSE_metadata.csv")
    df = df[['name', 'code']]
    print(df.columns)
    connection, cursor = create_connection_cursor()
    # Insert DataFrame recrds one by one.
    for i, row in df.iterrows():
        try:
            sql = "INSERT INTO `companies` (`name`, `quandl_code`) VALUES (%s, %s)"
            cursor.execute(sql, tuple(row))
        except:
            print("failed to insert this: ", tuple(row))
    connection.commit()


def load_bse_data():
    df = pd.read_csv("/home/deepak/Downloads/stock_data/bse_securities.csv")
    df = df[['Security Id', 'Security Code', 'Industry', 'Security Name', 'ISIN No']]
    print(df.columns)
    connection, cursor = create_connection_cursor()
    # Insert DataFrame recrds one by one.
    for i, row in df.iterrows():
        try:
            select_statement = "select id from companies where quandl_code like '%" \
                               + str(row['Security Code']) + "%'"
            cursor.execute(select_statement)
            result = cursor.fetchone()

            sql = "update `companies` set `symbol` = '%s'" + ", `security_code` = '%s'" + ", `industry`= '%s'" + \
                  ", `actual_name`= '%s'" + ", `isin`= '%s' where id = %d"
            string_ci = sql % (row['Security Id'], row['Security Code'], row['Industry'], row['Security Name'],
                               row['ISIN No'], result[0])
            cursor.execute(string_ci)

        except Exception as e:
            print("failed to insert this: ", tuple(row), "e: ", e)
    connection.commit()


if __name__ == '__main__':
    # load_quandl_data()
    # load_bse_data()
    pass
