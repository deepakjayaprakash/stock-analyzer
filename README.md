# Stock Analyzer


The project can be used for buiding personal trading stategy and to follow the trend of certain stocks/companies that you usually add in your watchlist in trading platforms. 
It is aimed to solve the following use cases:
- Load historical data incrementally so that you can analyze offline without the need to pull data every time you want to build a model
- Build models using algorithms like Regression, Neural Networks(LSTM), Ensemble of more than 1 type of models, etc to predict future values of a company. Analyze its trend and build statistics which helps you decide the time and price of when you can buy the stock if it's good enough
- Get the finanicals of a company in one go and display the summarised results of selected metrics which you usually use to check if a company is worth investing. This is to help you understand and pick stocks which have good balance sheets, cash flows, ratios, share holding pattern, etc.

---
### Tech stack used:
- Django(Python) server to build APIs through which data can be loaded into the db, build models, visualize results, etc.
- Mysql database to store historical data
- [Quandl](https://www.quandl.com/) as data source to extract incremental data

---
### Installation steps:
Run the following commands

```make venv```


### Steps to start the server

```make install```

```source .venv/bin/activate```

```make run env=local```

---
