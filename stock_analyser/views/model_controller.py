from stock_analyser.models.simple_model import simple_test, predict


def simple_api(request, id):
    return simple_test(id)

def predictor(request, id):
    return predict(id)
