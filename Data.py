import pandas as pd
import yfinance as yf
class DataCollector:
    def __init__(self, tickers, beg, end, interval):
        self.tickers = tickers
        self.beg = beg
        self.end = end
        self.interval = "1d"

    def get_prices(self):
        data = None
        if self.beg and self.end:
            data = yf.download(self.tickers, start = self.beg, end = self.end, auto_adjust=True)
        else:
            data = yf.download(self.tickers, period = "5y", interval = "1d", auto_adjust=True)
        data = data['Close'] if isinstance(data, pd.DataFrame) else None
        return data

    def get_returns(self, excess = False):
        data = None
        if self.beg and self.end:
            data = yf.download(self.tickers, start = self.beg, end = self.end, auto_adjust=True)
        else:
            data = yf.download(self.tickers, period = "5y", interval = "1d", auto_adjust=True)
        data = data['Close'] if isinstance(data, pd.DataFrame) else None
        data = data.pct_change().dropna() if isinstance(data, pd.DataFrame) else None

        if excess:
            ff = pd.read_csv("data/ff_daily.csv")
            ff['Date'] = pd.to_datetime(ff['Date'], format = "%Y%m%d")
            ff.set_index('Date', inplace = True)
            common = data.index.intersection(ff.index)
            ff = ff.loc[common]
            rf = ff["RF"] / 100
            data = data.sub(rf, axis = 0)

        return data

    def get_shares(self):
        """
        Note: this is not exact since I do not have access to time-specific shares out. data
        """
        data = {}
        for ticker in self.tickers:
            try:
                data[ticker] = yf.Ticker(ticker).info['sharesOutstanding']
            except:
                try:
                    data[ticker] = yf.Ticker(ticker).info['totalAssets']
                except:
                    print(f"No data found for {ticker}")
        return pd.Series(data)
