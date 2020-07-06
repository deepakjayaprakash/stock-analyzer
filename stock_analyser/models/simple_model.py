import datetime

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from django.http import JsonResponse
from sklearn import linear_model, svm, tree
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid, KNeighborsRegressor
from sklearn.neural_network import MLPRegressor

from stock_analyser import settings

matplotlib.use('TkAgg')


def simple_test(id):
    sql = "select trade_date,close, num_shares from time_series where company_id = %d and close is not null" % (id)
    company_data = pd.read_sql(sql, settings.DATABASE_URL)

    company_data_modified = pre_process_data(company_data)

    print(company_data_modified.head())
    feature_train, feature_test, result_train, result_test = train_test_split(
        company_data_modified[['trade_date', 'date_int', 'num_shares']], company_data_modified['close'], test_size=0.2)

    print("train size", feature_train.shape)
    print("test size", feature_test.shape)

    apply_regression(feature_test, feature_train, result_test, result_train, company_data_modified)
    response = {}
    return JsonResponse(response)


def get_actual_test_result():
    current_date = datetime.date.today()
    actual_test_date = pd.DataFrame(columns=['trade_date', 'date_int'])
    end_date = 200
    start = 1
    while (start < end_date):
        next_date = current_date + datetime.timedelta(days=start)
        next_date_int = next_date.toordinal()
        actual_test_date.loc[start] = [next_date, next_date_int]
        start = start + 1
    return actual_test_date


def predict(id):
    sql = "select trade_date,close, num_shares from time_series where company_id = %d and close is not null" % (id)
    company_data = pd.read_sql(sql, settings.DATABASE_URL)

    company_data_modified = pre_process_data(company_data)

    print(company_data_modified.head())
    feature_train, feature_test, result_train, result_test = train_test_split(
        company_data_modified[['trade_date', 'date_int', 'num_shares']], company_data_modified['close'], test_size=0.11)

    print("train size", feature_train.shape)
    print("test size", feature_test.shape)

    feature_list = ['date_int']
    X_train = feature_train[feature_list]

    # regr = tree.DecisionTreeRegressor()
    regr = MLPRegressor(random_state=1, max_iter=500)
    regr.fit(X_train, result_train)
    print("score: ", regr.score(X_train, result_train))

    actual_test_data = get_actual_test_result()

    result_pred = regr.predict(actual_test_data[['date_int']])
    actual_test_data['predicted'] = result_pred
    print("predcited_data: \n", actual_test_data[['trade_date', 'predicted', 'date_int']])

    # plot_prediction(company_data, result_pred, actual_test_data)
    response = {}
    return JsonResponse(response)


def get_regressor(i):
    regressor = 'linear'
    if i == 'linear':
        regressor = linear_model.LinearRegression()
    elif i == 'svr':
        regressor = svm.SVR()
    elif i == 'knn':
        regressor = KNeighborsRegressor()
    elif i == 'gradient_boost':
        regressor = GradientBoostingRegressor()
    elif i == 'decision_tree':
        regressor = tree.DecisionTreeRegressor()
    elif i == 'random_forest':
        regressor = RandomForestRegressor()
    elif i == 'mlp':
        regressor = MLPRegressor(random_state=1, max_iter=500)
    elif i == 'voting':
        regr = GradientBoostingRegressor()
        regr2 = tree.DecisionTreeRegressor()
        regressor = VotingRegressor(estimators=[('gb', regr), ('rf', regr2)])

    return regressor


def apply_regression(feature_test, feature_train, result_test, result_train, company_data):
    feature_list = ['date_int', 'num_shares']
    X_train = feature_train[feature_list]
    X_test = feature_test[feature_list]
    regressors = ['linear', 'svr', 'knn', 'gradient_boost', 'decision_tree', 'random_forest', 'mlp', 'voting']

    print("========================")
    for i in regressors:
        regr = get_regressor(i)
        run_regression_and_plot(X_test, X_train, company_data, feature_test, regr, result_test, result_train, i)
    print("========================")


def run_regression_and_plot(X_test, X_train, company_data, feature_test, regr, result_test, result_train, model_name):
    regr.fit(X_train, result_train)
    print("..................")
    print("model_name: ", model_name)
    print("score: ", regr.score(X_train, result_train))
    result_pred = regr.predict(X_test)
    print("mae", mean_absolute_error(result_test, result_pred))
    test = pd.DataFrame(columns=['actual', 'pred'])
    test['actual'] = result_test
    test['pred'] = result_pred

    # print(test.tail(n=4))
    # plot_results(company_data, feature_test, result_pred, result_test)
    print("..................")


def plot_results(company_data, feature_test, result_pred, result_test):
    ax = company_data.plot(x='trade_date', y='close', style='b-', grid=True)
    ax.set_xlabel("date")
    ax.set_ylabel("price")
    ax.scatter(feature_test['trade_date'], result_test, color='r')
    ax.scatter(feature_test['trade_date'], result_pred, color='g')
    ax.legend(['actual_data', 'test', 'predicted'])
    plt.show()


def plot_prediction(company_data, result_pred, actual_test_data):
    ax = company_data.plot(x='trade_date', y='close', style='b-', grid=True)
    ax.set_xlabel("date")
    ax.set_ylabel("price")
    ax.scatter(actual_test_data['trade_date'], result_pred, color='g')
    ax.legend(['actual_data', 'predicted'])
    plt.show()


def pre_process_data(company_data):
    company_data_modified = company_data.copy()
    company_data_modified['date_int'] = company_data.apply(lambda row: row['trade_date'].toordinal(), axis=1)
    return company_data_modified
