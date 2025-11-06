import numpy as np
import pandas as pd
import yfinance as yf
import textwrap

class Metrics:
    def __init__(self, returns: pd.DataFrame):
        self.returns = returns

    def add(self, name, data):
        self.returns[name] = data

    def mean(self):
        """
        Assumes geometric mean return
        """
        N = len(self.returns)
        Vt = (1 + self.returns).prod()
        G = Vt ** (252 / N) - 1
        return G

    def nav(self):
        return (1 + self.returns).cumprod()

    def t_stat_mean(self):
        N = len(self.returns)

    def regress(self, index = "SPY"):
        pass

    def std(self):
        vol = self.returns.std(ddof = 1)
        return vol * np.sqrt(252)

    def sharpe(self):
        """
        Assumes portfolio returns are already in excess returns
        """
        mu = self.mean()
        std = self.std()
        return mu / std

    def report(self):
        mu = self.mean()
        st = self.std()
        sr = self.sharpe()
        return textwrap.dedent(f"""
            Portfolio Report:
            Mu: {mu:.3f}
            Std: {st:.3f}
            Sharpe Ratio: {sr:.2f}
        """)
