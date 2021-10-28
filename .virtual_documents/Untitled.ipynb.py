import pandas as pd
import numpy as np

import pandas_profiling as pp
from pandas_profiling import ProfileReport


# reading two csv files
data1 = pd.read_csv('data/train.csv', parse_dates=True)
data2 = pd.read_csv('data/store.csv', parse_dates=True)

# using merge function by setting how='inner'
total = pd.merge(data1, data2,
                   on='Store',
                   how='inner')
data1.





data1.info()


from datetime import datetime 
from datetime import date
data1['date'] = pd.to_datetime(data1['date'])
pd.data1.info()


report = ProfileReport(data1, infer_dtypes=True) 
report.to_html()


data1["date"].apply(lambda x: x if isinstance(x,datetime.datetime) else np.nan).fillna(method = "ffill")




