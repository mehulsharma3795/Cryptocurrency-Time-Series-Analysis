import pandas as pd
from fbprophet import Prophet
import matplotlib.pyplot as plt

class prophet_model(object):
    def __init__(self,file,ind):
        self.data = pd.read_csv(file, parse_dates=['Date'])
        self.data["Percent"] = self.data.Mean.div(self.data.Mean.shift())
        self.data["Returns"] = self.data.Percent.sub(1).mul(100)
        self.data = self.data.reindex(index=self.data.index[::-1])
        self.data = self.data[ind:]
        self.data = self.data.rename(columns={'Date': 'ds', 'Mean': 'y'})
    
    def model(self):
        from fbprophet import Prophet
        self.model = Prophet(interval_width = 0.99 ,changepoint_prior_scale = 0.1,daily_seasonality=False,yearly_seasonality=True)
        self.model.fit(self.data)
        
    def forecast(self):
        forecast = self.model.make_future_dataframe(periods=50)
        self.future_forecast = self.model.predict(forecast)
        plt.figure(figsize=(18, 6))
        self.model.plot(self.future_forecast, xlabel = 'Date', ylabel = 'Cryptocurrency Returns')
        plt.title('Crytocurrency Returs Over Time')

    def Vizualization(self):
        self.model.plot_components(self.future_forecast)