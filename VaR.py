import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from scipy.stats import norm
import matplotlib.mlab as mlab
import seaborn as sns

class Var(object):
    def __init__(self,data):
        self.ret = data.Returns
    def var_cov(self,bin_no):
        self.mean = np.mean(self.ret)
        self.std = np.std(self.ret)
        sns.distplot(self.ret,bins=bin_no)
        plt.show()
        
        #VaR Computation 
        VaR_90 = norm.ppf(1-0.9, self.mean, self.std)
        VaR_95 = norm.ppf(1-0.95, self.mean, self.std)
        VaR_99 = norm.ppf(1-0.99, self.mean, self.std)
        print(tabulate([['90%',VaR_90],['95%',VaR_95],['99%',VaR_99]],headers=['Confidence Interval','Value At Risk']))
    
    def hist_sim(self,bin_no):
        plt.hist(self.ret,bins = bin_no)
        plt.xlabel("Returns")
        plt.ylabel("frequency")
        plt.grid(True)
        plt.show()
        sort_ret = self.ret.sort_values(ascending=True)
        VaR_90 = self.ret.quantile(0.1)
        VaR_95 = self.ret.quantile(0.05)
        VaR_99 = self.ret.quantile(0.01)
        print(tabulate([['90%',VaR_90],['95%',VaR_95],['99%',VaR_99]],headers=['Confidence Interval','Value At Risk']))
        
        
    def monte_carlo(self):
        self.mean = np.mean(self.ret)
        self.std = np.std(self.ret)
        np.random.seed(42)
        n_sims = 1000000
        sim_returns = np.random.normal(self.mean, self.std, n_sims)
        SimVAR = self.ret.iloc[-1]*np.percentile(sim_returns, 1)
        print('Simulated VAR is ', SimVAR)  
        