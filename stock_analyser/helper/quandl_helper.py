import quandl

from stock_analyser import settings


def get_quandl_data(quandl_code, from_date, to_date):
    quandl_response = quandl.get(quandl_code, authtoken=settings.QUANDL_AUTH_TOKEN,
                                 start_date=from_date,
                                 end_date=to_date)
    return quandl_response
