import numpy as np
import pandas as pd
import yfinance as yf
import textwrap
import statsmodels.api as sm

class Metrics:
    def __init__(self, returns: pd.DataFrame, interval = "d"):
        self.returns = returns
        self.interval = interval
        self.interval_factor = self.get_interval_factor()

    def get_interval_factor(self):
        if self.interval == "d":
            return 252
        elif self.interval == "m":
            return 12
        else:
            return 252

    def add(self, name, data):
        self.returns[name] = data

    def mean(self):
        """
        Assumes geometric mean return
        """
        N = len(self.returns)
        Vt = (1 + self.returns).prod()
        G = Vt ** (self.interval_factor / N) - 1
        return G

    def nav(self):
        return (1 + self.returns).cumprod()

    def t_stat_mean(self):
        N = len(self.returns)

    def regress(self, index = "SPY"):
        print("HI")
        if index in self.returns:
            res = []
            portfolios = []
            X = sm.add_constant(self.returns[index])
            for portfolio in self.returns.columns:
                if portfolio != index:
                    y = self.returns[portfolio]
                    model = sm.OLS(y, X).fit()
                    portfolios.append(portfolio)
                    res.append({
                        "Alpha": model.params['const'],
                        "Beta": model.params[index],
                        "t-stat Alpha": model.tvalues['const'],
                        "R2": model.rsquared,
                    })
            return pd.DataFrame(res, index = portfolios).T


    def std(self):
        vol = self.returns.std(ddof = 1)
        return vol * np.sqrt(self.interval_factor)

    def sharpe(self):
        """
        Assumes portfolio returns are already in excess returns
        """
        mu = self.mean()
        std = self.std()
        return mu / std

    def report(self, index = "SPY"):
        mu = self.mean()
        mu.name = "Mean"

        st = self.std()
        st.name = "Std"

        sr = self.sharpe()
        sr.name = "Sharpe"

        regression = self.regress(index)

        out = pd.concat([mu, st, sr], axis = 1).T
        out = pd.concat([out, regression])

        return out
