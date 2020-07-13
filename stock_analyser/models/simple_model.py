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
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
import numpy as np

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


def get_actual_test_data(days):
    current_date = datetime.date.today()
    actual_test_date = pd.DataFrame(columns=['trade_date', 'date_int'])
    end_date = days
    start = 1
    while (start < end_date):
        next_date = current_date + datetime.timedelta(days=start)
        next_date_int = next_date.toordinal()
        actual_test_date.loc[start] = [next_date, next_date_int]
        start = start + 1
    return actual_test_date


def predict_from_model(id):
    sql = "select trade_date,close, num_shares from time_series where company_id = %d and close is not null order by trade_date asc" % (
        id)
    company_data = pd.read_sql(sql, settings.DATABASE_URL)

    company_data_modified = pre_process_data(company_data)

    print(company_data_modified.head())

    predict_using_lstm(company_data_modified[['close']], company_data_modified)

    # predict_using_regression(company_data_modified)

    # plot_prediction(company_data, result_pred, actual_test_data)
    response = {}
    return JsonResponse(response)


def predict_using_lstm(total_data, raw_data):
    predit_last_num = 100  # number of points from most recent to use as test data

    predited_data = pd.DataFrame(columns=['predicted_price', 'actual_price', 'date'])
    predited_data['actual_price'] = raw_data[-predit_last_num:]['close']
    predited_data['date'] = raw_data[-predit_last_num:]['trade_date']

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(total_data)
    features_set = []
    labels = []
    # last 60 days prices will be 60 columns(feature set) and 61 st column will be 61st i.e day's price stored in labels

    for i in range(60, scaled_data.shape[0]):
        features_set.append(scaled_data[i - 60:i, 0])
        labels.append(scaled_data[i, 0])

    features_set, labels = np.array(features_set), np.array(labels)
    print("feature_set shape after data preperation: ", features_set.shape)
    print("label shape after data preperation: ", labels.shape)
    # converting feature set to LSTM input format,
    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))

    model = build_lstm_model(features_set, labels)

    print("train data: shape: ", features_set.shape)
    print("label data: size: ", labels.shape)

    test_features = []
    for i in range(scaled_data.shape[0] - predit_last_num, scaled_data.shape[0]):
        test_features.append(scaled_data[i - 60:i, 0])

    test_features = np.array(test_features)
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))

    predictions = model.predict(test_features)
    predictions = scaler.inverse_transform(predictions)
    predictions = predictions.flatten()  # convert nd array output to 1 d array

    predited_data['predicted_price'] = predictions
    print(predited_data)

    plot_lstm_results(predited_data)


def plot_lstm_results(predited_data):
    ax = predited_data.plot(x='date', y='actual_price', style='b-', grid=True)
    ax.set_xlabel("date")
    ax.set_ylabel("price")
    ax.scatter(predited_data['date'], predited_data['predicted_price'], color='g')
    ax.legend(['actual_data', 'predicted'])
    plt.show()


def build_lstm_model(features_set, labels):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(features_set.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(features_set, labels, epochs=1, batch_size=32)
    return model


def predict_using_regression(company_data_modified):
    feature_train, feature_test, result_train, result_test = train_test_split(
        company_data_modified[['trade_date', 'date_int', 'num_shares']], company_data_modified['close'], test_size=0.1)

    print("train size", feature_train.shape)
    print("test size", feature_test.shape)

    feature_list = ['date_int']
    X_train = feature_train[feature_list]

    regr = tree.DecisionTreeRegressor()
    regr = MLPRegressor(random_state=1, max_iter=500)
    regr.fit(X_train, result_train)
    print("score: ", regr.score(X_train, result_train))
    actual_test_data = get_actual_test_data(31)
    result_pred = regr.predict(actual_test_data[['date_int']])
    actual_test_data['predicted'] = result_pred
    print("predcited_data: \n", actual_test_data[['trade_date', 'predicted', 'date_int']])


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
    result_pred = regr.predict_from_model(X_test)
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
