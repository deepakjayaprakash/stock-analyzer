from stock_analyser.models.lstm import lstm_actual_prediction
from stock_analyser.models.simple_model import simple_test, predict_from_model


def simple_api(request, id):
    return simple_test(id)

def predictor(request, id):
    return predict_from_model(id)

def main_predictor(request, id):
    return lstm_actual_prediction(id)
