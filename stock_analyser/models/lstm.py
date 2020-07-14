import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from django.http import JsonResponse
from sklearn.preprocessing import MinMaxScaler

from stock_analyser import settings
from stock_analyser.models.simple_model import pre_process_data, build_lstm_model, get_actual_test_data


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
    print("shape predicted", predicted_data.head(num_future_days))

    ax = predicted_data.plot(x='date', y='predicted_price', style='b-', grid=True)
    ax.set_xlabel("date")
    ax.set_ylabel("price")
    ax.legend(['predicted'])
    plt.show()

    max_price = 0
    for i, row in predicted_data.iterrows():
        if row['predicted_price'] > max_price:
            max_price = row['predicted_price']

    print("max_price:", max_price)
    first = predicted_data['predicted_price'][1]
    last = predicted_data['predicted_price'][predicted_data.shape[0]]
    if first < last:
        print("positive trend")
    else:
        print("negative trend")

    response = {}
    return JsonResponse(response)


def prepare_test_data_and_predict(features_set, model, num_previous_days, scaler, num_future_days):
    data_length = features_set.shape[0]
    test_features = []
    print(type(features_set[data_length - 1]))
    test_features.append(features_set[data_length - 1].flatten())

    test_features_np_array = np.array(test_features)
    test_features_np_array = test_features_np_array.flatten()
    print("test_features:", test_features_np_array)

    predicted_results = []
    for position in range(0, num_future_days - 1):
        test_sub_set = []
        test_sub_set.append(test_features_np_array[position: position + num_previous_days])
        test_sub_set = np.array(test_sub_set)
        # print("subset: for ", position, " :", test_sub_set)
        test_sub_set = np.reshape(test_sub_set, (test_sub_set.shape[0], test_sub_set.shape[1], 1))
        predictions = model.predict(test_sub_set)
        scaled_predictions = scaler.inverse_transform(predictions)
        scaled_predictions = scaled_predictions.flatten()  # convert nd array output to 1 d array
        predicted_results.append(scaled_predictions[0])
        test_features_np_array = np.append(test_features_np_array, predictions[0])
        print("predicting ", position, " day ahead, value = ,", predictions[0], " scaled value = ",
              scaled_predictions[0])
    return predicted_results
