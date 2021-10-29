import numpy as np
import pandas as pd


# reading two csv files
data1 = pd.read_csv('data/train.csv',dtype={'Store':'string', 'StateHoliday':'string'})
data2 = pd.read_csv('data/store.csv',dtype={'Store':'string', 'StoreType':'string', 'Assortment':'string'})
  
# using merge function by setting how='inner'
total = pd.merge(data1, data2, 
                   on='Store', 
                   how='inner')


total


X = total.drop(columns='Sales')
y = total.loc[:,'Sales']


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


todo = [y_train, X_train]


df_train = pd.concat(todo, axis=1)


from pandas_profiling import ProfileReport


#prof1 = ProfileReport(df_train)


#prof1.to_file(output_file='prof1.html')


df_train.info()


df_train.astype


df_train = df_train.dropna(subset =['Sales'])


df_train['Sales'].isnull().value_counts()


df_train = df_train.drop(columns=['Customers','PromoInterval'])


df_train


df_train.isnull().sum()


#competitor distance

df_train['CompetitionDistance']=df_train.loc[:,'CompetitionDistance'].fillna(df_train['CompetitionDistance'].median())


df_train.isnull().sum()


df_train['Open'].isna().sum()


df_train.loc[df_train['Sales']> 0, 'Open']=1


df_train.loc[df_train['Sales']== 0, 'Open']=0


df_train['Open'].value_counts()


df_train[df_train['Sales'] > 0]


df_train['StateHoliday'] = df_train.loc[:,'StateHoliday'].fillna('0')


df_train


df_train


df_train['StateHoliday'] = df_train['StateHoliday'].replace(0.0, '0')


df_train



df_train.StateHoliday.unique()


df_train['SchoolHoliday'].isnull().value_counts()


df_train['SchoolHoliday'] = df_train['SchoolHoliday'].fillna(0)


df_train


df_train.info()


df_train['Promo'].isnull().value_counts()


df_train['DayOfWeek'].value_counts()


df_train.index


df_train['Date'] = pd.to_datetime(df_train['Date'])
df_train['Year'] = pd.DatetimeIndex(df_train['Date']).year


df_train['Month'] = pd.DatetimeIndex(df_train['Date']).month


df_train['Day'] = pd.DatetimeIndex(df_train['Date']).day


df_train


df_train.info()


df_train.info()


df_train


#prof2 = ProfileReport(df_train)


#prof2.to_file(output_file='prof2.html')


df_train['Promo'].isna().value_counts()


df_train['Promo'] = df_train.loc[:,'Promo'].fillna(0)


df_train


df_train['Promo2'].isna().value_counts()


from datetime import datetime 
from datetime import date 

## Create a new column that is a real date.
## For a time series we need real dates.
df_train['sales_date'] = pd.to_datetime(df_train['Date'])
total.info()


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


create_a_timeseries_dummy_from_weeks(df=df_train        ,
                          var_week='Promo2SinceWeek'  ,
                          var_year='Promo2SinceYear'  ,
                          date_new='Promo2_start_date',
                          var_dummy='Promo2_yes'
                                    )


create_a_timeseries_dummy_from_months(df=df_train                  ,
                          var_month='CompetitionOpenSinceMonth'  ,
                          var_year='CompetitionOpenSinceYear'   ,
                          date_new='Competition_start_date'    ,
                          var_dummy='Competition_yes'
                                    )



df_train


df_train = df_train.drop(columns=['Promo2_start_date','Competition_start_date','temp_date','Date', 'Promo2'])


df_train = df_train.drop(columns=['CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','CompetitionOpenSinceMonth'])


#df_train = df_train.drop(columns=['CompetitionOpenSinceMonth'])


df_train





import category_encoders as ce



# these are already dummies 'Open','Promo', 'SchoolHoliday',,'Promo2'
ce_one = ce.OneHotEncoder(cols=['DayOfWeek','StateHoliday','StoreType','Assortment'])
ce_one.fit_transform(df_train)


df_train.columns


df_train.head()


pip list




