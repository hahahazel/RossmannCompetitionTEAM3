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


#Train test split processing

X = total.drop(columns='Sales')
y = total.loc[:,'Sales']


from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


todo = [y_train, X_train]


#Merge train data (X and y) to process them together
df_train = pd.concat(todo, axis=1)


#Profilibg to check data
from pandas_profiling import ProfileReport


#prof1 = ProfileReport(df_train)


#prof1.to_file(output_file='prof1.html')


#Drop na values in sales

df_train = df_train.dropna(subset =['Sales'])

#Drop columns Customers and PromoInterval

df_train = df_train.drop(columns=['Customers','PromoInterval'])


df_train['Sales'].isnull().value_counts()


# Competitor distance : missing values

df_train['CompetitionDistance']=df_train.loc[:,'CompetitionDistance'].fillna(df_train['CompetitionDistance'].median())


df_train.isnull().sum()


df_train['Open'].isna().sum()


#Open: filling na using Sales

df_train.loc[df_train['Sales']> 0, 'Open']=1
df_train.loc[df_train['Sales']== 0, 'Open']=0


df_train['Open'].value_counts()


#SatateHoloiday: fillna with 0 and get a string
df_train['StateHoliday'] = df_train.loc[:,'StateHoliday'].fillna('0')
df_train['StateHoliday'] = df_train['StateHoliday'].replace(0.0, '0')


df_train.StateHoliday.unique()


df_train['SchoolHoliday'].isnull().value_counts()


#Schoolholiday fillna using 0
df_train['SchoolHoliday'] = df_train['SchoolHoliday'].fillna(0)


#Tranform Data into datatime pd to get the day, week and year
df_train['Date'] = pd.to_datetime(df_train['Date'])
df_train['Year'] = pd.DatetimeIndex(df_train['Date']).year
df_train['Month'] = pd.DatetimeIndex(df_train['Date']).month
df_train['Day'] = pd.DatetimeIndex(df_train['Date']).day


#prof2 = ProfileReport(df_train)


#prof2.to_file(output_file='prof2.html')


df_train['Promo'].isna().value_counts()


#Promo missing values (0)
df_train['Promo'] = df_train.loc[:,'Promo'].fillna(0)


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


#Drop columns for RandomForest Regressor
df_train = df_train.drop(columns=['Promo2_start_date','Competition_start_date','temp_date','Date', 'Promo2'])
df_train = df_train.drop(columns=['CompetitionOpenSinceYear','Promo2SinceWeek','Promo2SinceYear','CompetitionOpenSinceMonth'])


df_train


import category_encoders as ce


# these are already dummies 'Open','Promo', 'SchoolHoliday',,'Promo2'
ce_one = ce.OneHotEncoder(cols=['DayOfWeek','StateHoliday','StoreType','Assortment'])
df_train = ce_one.fit_transform(df_train)


df_train.columns


df_train.head()


#use df_randomf for onehotencoding
df_randomf = df_train.copy()


df_randomf


ce_one = ce.OneHotEncoder(cols=['Year','Month'])
df_randomf = ce_one.fit_transform(df_randomf)


df_randomf.info()


#Drop Day and Sales_date 

df_randomf = df_randomf.drop(columns=['Day','sales_date'])


#Remove for Sales 0 values (regarding metrics)
df_randomf_clean = df_randomf['Sales'].dropna(axis=0)
df_randomf_clean

df_randomf_clean1 = df_randomf.copy()
df_randomf_clean1.loc[:,'Sales'] = df_randomf['Sales'].dropna(axis=0)
df_randomf_clean1['Sales'].isnull().value_counts()
df_randomf_clean1 = df_randomf_clean1.drop(df_randomf_clean1[df_randomf_clean1['Sales']==0].index)


df_randomf_clean1.isna().sum()


#Split data
X_train = df_randomf_clean1.drop(columns=['Sales'])
y_train = np.asarray(df_randomf_clean1.loc[:,'Sales'])


from sklearn.preprocessing import power_transform
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error



# Scale 
scal = StandardScaler()
X_train_scaled = scal.fit_transform(X_train)


#Model Random Forest
from sklearn.ensemble import RandomForestRegressor




#Prediction on training Data
clf = RandomForestRegressor(n_estimators=300, n_jobs=2)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_train_scaled)



np.sqrt(mean_squared_error(y_train, y_pred))





type(y_train)


#Building required metric
def metric(preds, actuals):
    preds = preds.reshape(-1)
    actuals = actuals.reshape(-1)
    assert preds.shape == actuals.shape
    return 100 * np.linalg.norm((actuals - preds) / actuals) / np.sqrt(preds.shape[0])


metric(y_pred,y_train)


df_randomf_clean1.info()


y_pred.isnull().sum()



