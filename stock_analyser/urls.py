"""stock_analyser URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/2.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from stock_analyser.views import script_controller, load_controller, model_controller

urlpatterns = [
    # path('admin/', admin.site.urls),
    path("companies-stats", script_controller.get_companies_stats),
    path("get-company-details/<str:name>", script_controller.get_company_details),
    path("get-company-data/<str:name>", script_controller.get_company_data),
    path("load-time-series/<int:id>", load_controller.load_company_data),
    path("update-time-series/<int:id>", load_controller.update_company_data),
    path("update-time-series-with-date/<int:id>", load_controller.update_company_data_with_date),
    path("model-test/<int:id>", model_controller.simple_api),
    path("predictor/<int:id>", model_controller.predictor),
    path("main-predictor/<int:id>", model_controller.main_predictor),
]
