import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.http import JsonResponse
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler

from stock_analyser import settings


def lstm_actual_prediction(id):
    sql = "select trade_date,close, num_shares from time_series where company_id = %d and close is not null order by trade_date asc" % (
        id)
    company_data = pd.read_sql(sql, settings.DATABASE_URL)

    company_data_modified = pre_process_data(company_data)

    print(company_data_modified.head())

    return predict_using_lstm(company_data_modified)


def predict_using_lstm(company_data_modified):
    total_data = company_data_modified[['close']]  # actual data points
    predicted_data = pd.DataFrame(
        columns=['predicted_price', 'date'])  # will contain predicted data against the trade_date
    num_future_days = 45  # number of days you want to predict in the future
    num_previous_days = 60  # number of historical points you want to consider for calculating next data point

    actual_test_data = get_actual_test_data(num_future_days)
    predicted_data['date'] = actual_test_data['trade_date']

    # scaling values into 0 to 1 range
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(total_data)

    features_set = []
    labels = []
    # last 60 days prices will be 60 columns(feature set) and 61 st column will be 61st i.e day's price stored in labels
    for i in range(num_previous_days, scaled_data.shape[0]):
        features_set.append(scaled_data[i - num_previous_days:i, 0])
        labels.append(scaled_data[i, 0])
    features_set, labels = np.array(features_set), np.array(labels)

    print("feature_set shape after data preperation: ", features_set.shape)
    print("label shape after data preperation: ", labels.shape)

    # converting feature set to LSTM input format,
    features_set = np.reshape(features_set, (features_set.shape[0], features_set.shape[1], 1))
    model = build_lstm_model(features_set, labels)

    predicted_results = prepare_test_data_and_predict(features_set, model, num_previous_days, scaler, num_future_days)
    predicted_data['predicted_price'] = predicted_results
    print("shape predicted", predicted_data.head(10))
    print_stats(predicted_data)

    prediction_using_train_data = predict_for_train_data(company_data_modified, model, scaled_data, scaler)

    plot1 = plt.figure(1)
    ax = prediction_using_train_data.plot(x='date', y='actual_price', style='b-', grid=True)
    ax.set_xlabel("date")
    ax.set_ylabel("price")
    ax.scatter(prediction_using_train_data['date'], prediction_using_train_data['predicted_price'], color='g')
    ax.legend(['actual_data', 'predicted'])

    plot2 = plt.figure(2)
    ax2 = predicted_data.plot(x='date', y='predicted_price', style='b-', grid=True)
    ax2.set_xlabel("date")
    ax2.set_ylabel("price")
    ax2.legend(['predicted'])

    plt.show()

    response = {}
    return JsonResponse(response)


def predict_for_train_data(company_data_modified, model, scaled_data, scaler):
    predit_last_num = 500  # predicting and plotting the model for last n prices of the train data itself
    prediction_using_train_data = pd.DataFrame(columns=['predicted_price', 'actual_price', 'date'])
    prediction_using_train_data['actual_price'] = company_data_modified[-predit_last_num:]['close']
    prediction_using_train_data['date'] = company_data_modified[-predit_last_num:]['trade_date']
    prediction_using_train_data = predict_using_train_data(model, predit_last_num, prediction_using_train_data,
                                                           scaled_data, scaler)
    return prediction_using_train_data


def print_stats(predicted_data):
    max_price = 0
    for i, row in predicted_data.iterrows():
        if row['predicted_price'] > max_price:
            max_price = row['predicted_price']
    print("Statistics")
    print("===========================================")
    print("max_price: \t", max_price)
    first = predicted_data['predicted_price'][1]
    last = predicted_data['predicted_price'][predicted_data.shape[0]]
    trend = ""
    if first < last:
        trend = "postive"
    else:
        trend = "negative"
    print("Trend: \t", trend)
    print("===========================================")


# test data starts from last item in the array, basically most recent 60 days prices
# predict 61st value which is tomorrow's price and then append to same array to predict next day's value
# keep repeating for the next 'num_future_days'
def prepare_test_data_and_predict(features_set, model, num_previous_days, scaler, num_future_days):
    data_length = features_set.shape[0]
    test_features = []
    test_features.append(features_set[data_length - 1].flatten())
    test_features_np_array = np.array(test_features)
    test_features_np_array = test_features_np_array.flatten()

    predicted_results = []
    for position in range(0, num_future_days - 1):
        test_sub_set = []
        test_sub_set.append(test_features_np_array[position: position + num_previous_days])
        test_sub_set = np.array(test_sub_set)
        test_sub_set = np.reshape(test_sub_set, (test_sub_set.shape[0], test_sub_set.shape[1], 1))
        predictions = model.predict(test_sub_set)
        scaled_predictions = scaler.inverse_transform(predictions)
        scaled_predictions = scaled_predictions.flatten()  # convert nd array output to 1 d array
        predicted_results.append(scaled_predictions[0])
        test_features_np_array = np.append(test_features_np_array, predictions[0])
        print("predicting ", position, " day ahead, value = ,", predictions[0], " scaled value = ",
              scaled_predictions[0])
    return predicted_results


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
    model.fit(features_set, labels, epochs=100, batch_size=32)
    return model


def predict_using_train_data(model, predit_last_num, predited_data, scaled_data, scaler):
    test_features = []
    for i in range(scaled_data.shape[0] - predit_last_num, scaled_data.shape[0]):
        test_features.append(scaled_data[i - 60:i, 0])
    test_features = np.array(test_features)
    test_features = np.reshape(test_features, (test_features.shape[0], test_features.shape[1], 1))
    predictions = model.predict(test_features)
    predictions = scaler.inverse_transform(predictions)
    predictions = predictions.flatten()  # convert nd array output to 1 d array
    predited_data['predicted_price'] = predictions
    # print(predited_data)
    return predited_data
    # plot_lstm_results(predited_data)


def pre_process_data(company_data):
    company_data_modified = company_data.copy()
    company_data_modified['date_int'] = company_data.apply(lambda row: row['trade_date'].toordinal(), axis=1)
    return company_data_modified


# gets the trade dates for next n days
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
