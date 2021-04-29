from src import dataloader
from src import model

import numpy as np
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import streamlit as st

# load data
loader = dataloader.DataLoader()

@st.cache(allow_output_mutation=True)
def getData(stock):    
    df1 = loader.LoadData(stock)
    # df1 = loader.LoadLocal()
    dates = df1.index
    return df1, dates

def plotRawData(df1):
    fig = px.line(df1, x=df1.index, y='Close')
    fig.update_xaxes(rangeslider_visible=True)
        
    st.plotly_chart(fig, use_container_width=True)

@st.cache
def trainModel(df1, input_size, hidden_size, num_layers, lr, n):
    # transform data
    df1 = df1.Close.values.reshape(-1, 1)
    scaler = StandardScaler()
    df2 = scaler.fit_transform(df1)
    
    X = np.array([df2[i-input_size:i] for i in range(input_size, df2.shape[0])]).reshape(-1, 1, input_size)
    y = df2[input_size:]
    
    # work with model
    m = model.LSTM(input_size, hidden_size, num_layers)
    m.train(X, y, lr, n)
    return df2, scaler, m

def plotPredictData(pred):
    fig = px.line(pred, x=pred.index, y='Close')
    fig.update_xaxes(rangeslider_visible=True)
        
    st.plotly_chart(fig, use_container_width=True)


### main code ###
# ------------------------------- #

# Get stock data
stock = st.sidebar.text_input('Enter stock symbol', value='TSLA')
df1, dates = getData(stock)
if not df1.empty:
    plotRawData(df1)

# Train model
df2, scaler, m = trainModel(df1, 100, 2, 1, 0.001, 1000)

# Predict Model
days = st.sidebar.slider('Select future days to predict', min_value=0, max_value=180, value=1, step=1)
if days > 0:
    pred = m.predict(days, dates, scaler, df2)
    plotPredictData(pred)
    

