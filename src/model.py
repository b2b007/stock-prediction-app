import numpy as np
import pandas as pd
import torch
import torch.nn as nn
#import torch.functional as F
from torch.autograd import Variable
from datetime import date, timedelta


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear1 = nn.Linear(hidden_size, 32)
        self.linear2 = nn.Linear(32, 1)

        self.relu = nn.ReLU()

    def forward(self, X):
        h_0 = Variable(torch.zeros(self.num_layers, X.size(0), self.hidden_size))
        c_0 = Variable(torch.zeros(self.num_layers, X.size(0), self.hidden_size))

        output, (hn, cn) = self.lstm(X, (h_0, c_0))
        hn = hn.view(-1, self.hidden_size)
        out = self.relu(hn)
        out = self.linear1(out)
        out = self.relu(out)
        out = self.linear2(out)
        return out

    def preprocessing(self, X, y):
        X = Variable(torch.Tensor(X))
        y = Variable(torch.Tensor(y))
        return X, y

    def train(self, X, y, lr, n):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        
        X, y = self.preprocessing(X, y)

        for epoch in range(n):
            out = self.forward(X)
            optimizer.zero_grad()

            loss = criterion(out, y)
            loss.backward()

            optimizer.step()
            if epoch % (n/10) == 0:
                print('Epoch: %d, loss: %1.5f' % (epoch, loss.item()))
        return True

    def generate_future_dates(self, period):
        t = []
        for i in range(period * 2):
            if len(t) >= period:
                break
            temp = date.today() + timedelta(days=i)
            if temp.weekday() not in [5, 6]:
                t.append(temp.strftime('%Y-%m-%d'))
        return t

    def predict(self, period, dates, scaler, data):
        previous_dates = [i.strftime('%Y-%m-%d') for i in dates]
        future_dates = self.generate_future_dates(period)
        for i in range(period):
            temp = data[-self.input_size:].astype('float32').reshape(-1, 1, self.input_size)
            temp = torch.from_numpy(temp)
            pred_temp = self.forward(temp).item()
            data = np.append(data, pred_temp)
        data = scaler.inverse_transform(data)
        df = pd.DataFrame(data, index=previous_dates + future_dates, columns=['Close'])
        df.index = df.index.astype('datetime64[ns]')
        return df
