# Class to print head and check missing values
import os
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller,kpss
from arch.unitroot import PhillipsPerron


class predictions(object):
    # Importing libraries
    import os
    import numpy as np 
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.stattools import adfuller,kpss
    from arch.unitroot import PhillipsPerron
    plt.style.use('fivethirtyeight') 
    # Above is a special style template for matplotlib, highly useful for visualizing time series data
    os.chdir(r"C:\Users\mehul\Time_Series_Analysis")
    
    def __init__(self,file):
        self.data = pd.read_csv(file,index_col='Date', parse_dates=['Date'])
        self.data.index = pd.DatetimeIndex(self.data.index.values,freq=self.data.index.inferred_freq)
    
    def description(self):
        print(self.data.head())
        print("---------------------------------------------------------------")
        print(self.data.info())
        print("---------------------------------------------------------------")
        
    def missing(self):
        print(self.data.isna().sum())
        print("---------------------------------------------------------------")
        
    def plot_daily(self):
        self.data.plot(subplots=True, figsize=(10,12))
        plt.title('Daily price of Cryptocurrency')
        plt.show()
        
    def plot_monthly(self):
        self.data["Mean"].plot(kind='line') 
        plt.title('Average Monthly price of Crytocurrency')
        plt.show()
        
    def Percent_change(self):
        self.data["Percent"] = self.data.Mean.div(self.data.Mean.shift())
        self.data["Percent"].plot(figsize=(20,8))
        plt.title("Percentage Change in Price")
        plt.show()
        
    def Returns(self):
        self.data["Returns"] = self.data.Percent.sub(1).mul(100)
        self.data["Returns"].plot(figsize=(20,8))
        plt.title("Returns Plot")
        plt.show()
        
    def stationarity_test(self):
        self.data.rolmean = self.data.Mean.rolling(30).mean()
        self.data.rolstd = self.data.Mean.rolling(30).std()
        plt.plot(self.data.Mean,color='blue',label='Original')
        plt.plot(self.data.rolmean,color='red',label='Rolling Mean')
        plt.plot(self.data.rolstd,color='black',label='Rolling Std')
        plt.legend(loc='best')
        plt.title("Rolling Mean & Standard Deviation")
        plt.show()
        
        # Perform Dickey Fuller Test 
        print("Results of Dickey-Fuller Test")
        dftest = adfuller(self.data.Mean)
        dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,Value in dftest[4].items():
            dfoutput['Critical Value(%s)'%key] = Value
        print(dfoutput)
        print("------------------------------------------------------------------------------")
        print ('Results of KPSS Test:')
        kpsstest = kpss(self.data.Mean, regression='c')
        kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
        for key,value in kpsstest[3].items():
            kpss_output['Critical Value (%s)'%key] = value
        print (kpss_output)
        print("------------------------------------------------------------------------------")
        print ('Results of Phillips-Perron Test:')
        pptest = PhillipsPerron(self.data.Mean)
        print(pptest)
        print("------------------------------------------------------------------------------")