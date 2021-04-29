from datetime import date
import yfinance as yf
import pandas as pd


class DataLoader:
    def __init__(self):
        self.START = '2015-01-01'
        self.TODAY = date.today().strftime('%Y-%m-%d')
        self.STOCK = None

    def LoadData(self, STOCK):
        self.STOCK = STOCK
        data = yf.download(STOCK, self.START, self.TODAY)
        return data

    def LoadLocal(self):
        df = pd.read_csv('FileName.csv', index_col='Date')
        df.index = df.index.astype('datetime64[ns]')
        return df

    def SaveData(self, data):
        if data is not None:
            UNIQUE_NAME = self.STOCK + self.TODAY + '.csv'
            data.to_csv(UNIQUE_NAME)
            return True
        return False
