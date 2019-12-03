from arch.unitroot import PhillipsPerron
import statsmodels.tsa.api as smt
import os
import numpy as np 
import pandas as pd
from statsmodels.tsa.stattools import adfuller,kpss
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
os.chdir(r"C:\Users\mehul\Time_Series_Analysis")
from statsmodels.tsa.arima_model import ARIMA


class model(object):
    
    def __init__(self,df):
        self.data = df
    
    def stationarity_test(self):
        rolmean = self.data.Returns.rolling(30).mean()
        rolstd = self.data.Returns.rolling(30).std()
        plt.plot(self.data.Returns,color='blue',label='Original')
        plt.plot(rolmean,color='red',label='Rolling Mean')
        plt.plot(rolstd,color='black',label='Rolling Std')
        plt.legend(loc='best')
        plt.title("Rolling Mean & Standard Deviation")
        plt.show()
        # Perform Dickey Fuller Test 
        print("Results of Dickey-Fuller Test")
        self.data.Returns.dropna(inplace=True)
        dftest = adfuller(self.data.Returns)
        dfoutput = pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,Value in dftest[4].items():
            dfoutput['Critical Value(%s)'%key] = Value
        print(dfoutput)
        print ('Results of KPSS Test:')
        print("------------------------------------------------------------------------------")
        kpsstest = kpss(self.data.Returns, regression='c')
        kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
        for key,value in kpsstest[3].items():
            kpss_output['Critical Value (%s)'%key] = value
        print (kpss_output)
        print("------------------------------------------------------------------------------")
        print ('Results of Phillips-Perron Test:')
        pptest = PhillipsPerron(self.data.Returns)
        print(pptest)
        print("------------------------------------------------------------------------------")
        
    def lags(self):
        from statsmodels.tsa.stattools import acf,pacf
        lag_acf = acf(self.data.Returns,nlags=40,fft=False)
        lag_pacf = pacf(self.data.Returns,nlags=40)
        # Plotting ACF and PACF plots
        plt.figure(figsize=(20,8))
        plt.subplot(121)
        plt.plot(lag_acf)
        plt.axhline(y=0,linestyle='--',color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(self.data.Returns)),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(self.data.Returns)),linestyle='--',color='gray')
        plt.title("Autocorrelation Plot")
        plt.subplot(122)
        plt.plot(lag_pacf)
        plt.axhline(y=0,linestyle='--',color='gray')
        plt.axhline(y=-1.96/np.sqrt(len(self.data.Returns)),linestyle='--',color='gray')
        plt.axhline(y=1.96/np.sqrt(len(self.data.Returns)),linestyle='--',color='gray')
        plt.title("Partial Autocorrelation Plot")
        plt.show()
    
    def auto_graphics(self):
        fig,ax = plt.subplots(figsize=(14,7))
        acf = smt.graphics.plot_acf(self.data.Returns, lags=40 , alpha=0.05,ax=ax)
        plt.title("Autocorrelation Plot")
        plt.show()
        
    def partial_graphics(self):
        fig,ax = plt.subplots(figsize=(14,7))
        acf = smt.graphics.plot_pacf(self.data.Returns, lags=40 , alpha=0.05,ax=ax)
        plt.title("Partial Autocorrelation Plot")
        plt.show()
    
    def ARIMA_model(self,p,q):
        model = ARIMA(self.data.Returns,order=(p,0,q))
        self.results = model.fit() 
        print(self.results.summary2())
    
    def plot_predict(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.data.Returns)
        plt.plot(self.results.fittedvalues,color='red')
        plt.show()
     