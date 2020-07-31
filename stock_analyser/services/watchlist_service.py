import pandas as pd
from django.http import HttpResponse

from stock_analyser import settings
from stock_analyser.helper.JsonHelper import dump_json
from stock_analyser.models.lstm import run_model_and_return_stats


def analyze_watchlist_by_id(request, id):
    total_dict = {}
    company_id_list = get_company_ids_of_watchlist_id(id)
    for company_id in company_id_list:
        stats_company = run_model_and_return_stats(int(company_id))
        total_dict[stats_company['symbol']] = stats_company
    print(total_dict)
    json_response = dump_json(total_dict)
    return HttpResponse(json_response, content_type='application/json')


def get_company_ids_of_watchlist_id(id):
    sql = "select company_ids from watchlist where id = %d" % (id)
    company_ids = pd.read_sql(sql, settings.DATABASE_URL)
    company_id_list = str(company_ids['company_ids'].tolist()[0]).split(",")
    return company_id_list
