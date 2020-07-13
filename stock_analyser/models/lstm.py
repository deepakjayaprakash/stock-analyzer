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

    predict_using_lstm(company_data_modified[['close']])


def predict_using_lstm(total_data):
    predited_data = pd.DataFrame(columns=['predicted_price', 'date'])
    actual_test_data = get_actual_test_data(31)
    predited_data['date'] = actual_test_data['trade_date']

    print("first shape",[predited_data.shape])

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

    data_length = scaled_data.shape[0]
    predicted_results = []
    test_features = []
    test_features.append(scaled_data[data_length - 60: data_length, 0])

    for position in range(0, 30):
        test_sub_set = []
        test_features_np_array = np.array(features_set)
        test_sub_set.append(test_features_np_array[position: position + 60, 0])
        test_sub_set = np.array(test_sub_set)
        test_sub_set = np.reshape(test_sub_set, (test_sub_set.shape[0], test_sub_set.shape[1], 1))
        predictions = model.predict(test_sub_set)
        predictions = scaler.inverse_transform(predictions)
        predictions = predictions.flatten()  # convert nd array output to 1 d array
        predicted_results.append(predictions[0])
        test_features.append(predictions[0])
        print("predicting ", position, " day ahead, value = ", predictions[0])


    predited_data['predicted_price'] = predicted_results
    print("shape predicted", predited_data.shape)

    ax = predited_data.plot(x='date', y='predicted_price', style='b-', grid=True)
    ax.set_xlabel("date")
    ax.set_ylabel("price")
    ax.legend(['predicted'])
    plt.show()

    response = {}
    return JsonResponse(response)
