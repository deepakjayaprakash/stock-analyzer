import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from django.http import JsonResponse
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from stock_analyser import settings

matplotlib.use('TkAgg')


def simple_test():
    sql = "select trade_date,close from time_series where company_id = %d and close is not null" % (4086)
    company_data = pd.read_sql(sql, settings.DATABASE_URL)
    print(company_data.head())

    feature_train, feature_test, result_train, result_test = train_test_split(
        company_data[['trade_date']], company_data['close'], test_size=0.2)

    X_train = pd.DataFrame(columns=['trade_date'])
    X_test = pd.DataFrame(columns=['trade_date'])
    for i, row in feature_train.iterrows():
        X_train.loc[i] = row['trade_date'].toordinal()

    for i, row in feature_test.iterrows():
        X_test.loc[i] = row['trade_date'].toordinal()

    print("train size", X_train.shape)
    print("test size", X_test.shape)

    regr = linear_model.LinearRegression()

    # print(type(X_train['trade_date'].get(0)))

    regr.fit(X_train, result_train)
    print("regression_done")
    print("score: ", regr.score(X_train, result_train))
    result_pred = regr.predict(X_test)
    print("mae", mean_absolute_error(result_test, result_pred))

    test = pd.DataFrame(columns=['actual', 'pred'])
    test['actual'] = result_test
    test['pred'] = result_pred
    print(test.tail(n=40))

    # plt.scatter(company_data['trade_date'], company_data['close'], color='r')
    plt.scatter(feature_test['trade_date'], result_test, color='r')
    plt.scatter(feature_test['trade_date'], result_pred, color='b')
    plt.show()
    response = {}
    return JsonResponse(response)
