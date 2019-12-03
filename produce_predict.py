from statsmodels.tsa.statespace.sarimax import SARIMAX
from itertools import product
import pandas as pd
import matplotlib.pyplot as plt 

class produce_predict(object):
    def __init__(self,original):
        self.original = original
    
    def order(self):
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from itertools import product
        p = d = q = range(0, 2)
        pdq = list(product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 30) for x in list(product(p, d, q))]
        for param in pdq:
            for param_seasonal in seasonal_pdq:
                try:
                    mod = SARIMAX(self.original.Mean,order=param,seasonal_order=param_seasonal,
                                  enforce_stationarity=False,enforce_invertibility=False)
                    results = mod.fit()
                    print('ARIMA{}x{} - AIC:{}'.format(param, param_seasonal, results.aic))
                except:
                    continue
    def sarima_model(self,order,seasonal_order):
        model = SARIMAX(self.original.Mean,order = order,seasonal_order=seasonal_order, enforce_stationarity=False,               enforce_invertibility=False)                                                     
        self.results = model.fit()
            
    def diagnostics(self):
        self.results.plot_diagnostics(figsize=(16,9))
        plt.show()
    
    def pred_vs_actual(self):
        pred = self.results.get_prediction(start=pd.to_datetime('2017-01-01'), dynamic=False)    
        pred_ci = pred.conf_int()
        ax = self.original.Mean.plot(label='Observed')
        pred.predicted_mean.plot(ax=ax, label='Forecast', alpha=.7, figsize=(14, 7))
        ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.2)
        ax.set_xlabel('Date')
        ax.set_ylabel('Crypocurrency Returns')
        plt.legend()
        plt.show()
        
    def performance(self):
        mse = ((self.original.Mean-self.results.fittedvalues)**2).mean()
        RSS = sum((self.original.Mean-self.results.fittedvalues)**2)
        print(f' Residual Sum of Squares: {RSS}')
        print(f' Mean Squared Errot: {mse}')
        print(f' Root Mean Squared Errot: {round(np.sqrt(mse),2)}')
        
    def produce_forecast(self):
        pred_uc = self.results.get_forecast(steps=100)
        pred_ci = pred_uc.conf_int()
        ax = self.original.Mean.plot(label='observed', figsize=(14, 7))
        pred_uc.predicted_mean.plot(ax=ax, label='Forecast')
        ax.fill_between(pred_ci.index,pred_ci.iloc[:, 0],pred_ci.iloc[:, 1], color='k', alpha=.25)
        ax.set_xlabel('Date')
        ax.set_ylabel('Cryptocurrency Returns')
        plt.legend()
        plt.show()