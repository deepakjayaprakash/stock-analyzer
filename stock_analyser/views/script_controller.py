from django.http import HttpResponse
import pandas as pd

from stock_analyser import settings


def testAPI(request):
    sql = "select * from companies"
    df = pd.read_sql(sql, settings.DATABASE_URL)
    return HttpResponse("This means its working")
