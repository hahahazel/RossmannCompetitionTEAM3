import pandas as pd
import numpy as np

import pandas_profiling as pp
from pandas_profiling import ProfileReport


# reading two csv files
data1 = pd.read_csv('data/train.csv', dtype={'Store':'string'})
data2 = pd.read_csv('data/store.csv', dtype={'Store':'string', 'StoreType':'string', 'Assortment':'string'})

# using merge function by setting how='inner'
total = pd.merge(data1, data2,
                   on='Store',
                   how='inner')
print(data1.dtypes)

print(data2.dtypes)





data1.info()


from datetime import datetime 
from datetime import date 

## Create a new column that is a real date.
## For a time series we need real dates.
total['sales_date'] = pd.to_datetime(total['Date'])
total.info()





#Create a date out of the Promo2 variables so that we can create a time-series variable for Promo2
    
total['temp_date'] = total['Promo2SinceYear'] * 1000 + total['Promo2SinceWeek']  * 10 + 0
total['promo2_start_date'] = pd.to_datetime(total['temp_date'], format='get_ipython().run_line_magic("Y%W%w')", "")
total = total.drop(columns=['temp_date'])
total.info()


       
## Create an array to Group the data by group. and make dummy Store by Store
for Store, grouped in total.groupby('Store'):
   if [total['Promo2'] == 0]:
      total['Promo2_yes'] = 0
   elif  [total['Promo2_start_date'].isnull()]:
      total['Promo2_yes'] = 0
   elif [(total['sales_date'] >= total['Promo2_start_date'])]:
      total['Promo2_yes'] = 1
   else:
      total['Promo2_yes'] = 0

total.head()



print(total.groupby('Promo2').mean(total['Promo2']))


def create_a_timeseries_dummy_from_weeks(df, var_week, var_year, date_new, var_dummy):
    #Create a date out of the Promo2 variables so that we can create a time-series variable for Promo2
    df['temp_date'] = df[var_year] * 1000 + df[var_week]  * 10 + 0
    df[date_new] = pd.to_datetime(df['temp_date'], format='get_ipython().run_line_magic("Y%W%w')", "")
    ## Create an array to Group the data by group. and make dummy Store by Store
    for Store, grouped in df.groupby('Store'):
      if [df['sales_date'] >= df[date_new]]:
        df[var_dummy] = 1
      else:
        df[var_dummy] = 0
    df = df.drop(columns=['temp_date'])
    df.head()

    
def create_a_timeseries_dummy_from_months(df, var_month, var_year, date_new, var_dummy):
    #Create a date out of the Competition variables so that we can create a time-series variable for Competitioin
    df['temp_date'] = df[var_year] * 1000 + df[var_month]  * 10 + 0
    ## Create an array to Group the data by group. and make dummy Store by Store
    df[date_new] = pd.to_datetime(df['temp_date'], format='get_ipython().run_line_magic("Y%m%w')", "")
    for Store, grouped in df.groupby('Store'):
      if [df['sales_date'] >= df[date_new]]:
        df[var_dummy] = 1
      else:
        df[var_dummy] = 0
    df = df.drop(columns=['temp_date'])    
    df.head()


total.head()


create_a_timeseries_dummy_from_weeks(df=total,
                                     var_week='Promo2SinceWeek'  ,
                                     var_year='Promo2SinceYear'  ,
                          date_new='Promo2_start_date'          ,
                          var_dummy='Promo2_yes'
                                    )


create_a_timeseries_dummy_from_months(df=total   ,
                          var_month='CompetitionOpenSinceMonth'  ,
                          var_year='CompetitionOpenSinceYear'   ,
                          date_new='Competition_start_date'    ,
                          var_dummy='Competition_yes'
                                    )



total.head()


create_a_timeseries_dummy_from_months(df=total                  ,
                          var_month='CompetitionOpenSinceYear'  ,
                          var_year='CompetitionOpenSinceMonth'  ,
                          date_new='Competition_start_date',
                          var_dummy='Competition_yes')


  ## Create an array to Group the data by group. and make dummy Store by Store
    for Store, grouped in df.groupby('Store'):
      if temp_date.isnull():
        df.var_dummy = 0
      elif df.sales_date >= df.date_new:
        df.var_dummy = 1
      else:
        df.var_dummy = 0






total.Promo2.value_counts()
total.Promo2_date.info



total.info()




## Create an array to Group the data by group.
for Store, grouped in total.groupby('Store'):
  if Promo2_date.isnull():
    total.Promo = 0
  elif total.sales_date >= total.Promo2_date:
    total.Promo2 = 1
  else:
    total.Promo = 0




report = ProfileReport(data1, infer_dtypes=True) 
report.to_html.write_html('prof_report.html')


data1["date"].apply(lambda x: x if isinstance(x,datetime.datetime) else np.nan).fillna(method = "ffill")




