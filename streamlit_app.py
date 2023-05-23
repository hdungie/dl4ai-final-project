import streamlit as st
import datetime
from datetime import datetime
import pandas as pd
import tensorflow 
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import plotly.express as px
import math
import h5py
import requests
import io
import time
# from predictions_graph import graph

tab1, tab2, tab3 = st.tabs(["📈 Stock price prediction","🪙 Trading point Identification"," 📓 Portfolio Management"])
with tab1:
  col1, col2 = st.columns([1,4])
  with col1: region = st.selectbox('Select a region', ('Nasdaq', 'Vietnam'), index=0)

  if region == "Nasdaq":
    df = pd.read_csv('./search_engine_nasdaq.csv')
    df = df.fillna('')
    reg = 'nasdaq'
  else: 
    df = pd.read_csv('./search_engine_vn.csv')
    df = df.fillna('')
    reg = 'vn'

  df_search = df['company']
  with col2:
    company = st.selectbox("Search companies by name or symbol", df_search, index=0)

  col1, col2 = st.columns(2)
  with col1: 
    if region == 'Nasdaq':
      start_date = st.date_input( "Start date: ", min_value=datetime(2022,12,13), max_value=datetime(2023,12,13))
    if region == 'Vietnam':
      start_date = st.date_input( "Start date: ", min_value=datetime(2023,2,28), max_value=datetime(2023,8,28))
  with col2: 
    if region == 'Nasdaq':
      end_date = st.date_input( "End date: ", min_value=datetime(2022,12,13), max_value=datetime(2023,12,13))
    if region == 'Vietnam':
      end_date = st.date_input( "End date: ", min_value=datetime(2023,2,28), max_value=datetime(2023,8,28))

  comp = company.split(' ')
  ticker = comp[0]
  if region == "Nasdaq":
    filepath = f'./data-nasdaq/{ticker}.csv'
  if region == "Vietnam":
    filepath = f'./data-vn/history/{ticker}.csv'

  if end_date <= start_date:
    interval = (end_date - start_date).days
    st.error("End date must be after start date.", icon = "❌")
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    with col1:
      pass
    with col2:
      pass
    with col4:
      pass
    with col5:
      pass
    with col3 :
        predict_button = st.button('Predict', disabled = True)
  else: 
    interval = (end_date - start_date).days
    st.success(f"Interval: {interval} days")
    col1, col2, col3, col4, col5 = st.beta_columns(5)
    with col1:
      pass
    with col2:
      pass
    with col4:
      pass
    with col5:
      pass
    with col3 :
      predict_button = st.button("Predict")

  if predict_button:
    data = pd.read_csv(filepath)
    if region == "Nasdaq":
      new_df = data[['Date', 'Close']]
    if region == "Vietnam":
      new_df = data

    latest = new_df.loc[len(new_df)-1,'Date']
    if region == "Nasdaq":
      latest = datetime.strptime(latest, '%d-%m-%Y').date()
    else: latest = datetime.strptime(latest, '%Y-%m-%d').date()
    gap_end = (end_date - latest).days
    gap_start = (start_date - latest).days

    if gap_end <= 7:
        future = 7
        window_size = 30
        model = load_model(f'./prediction-models/model-{ticker}--7d-ws30.h5')
    elif gap_end > 7 and gap_end <=30:
        future = 30
        window_size = 30
        model = load_model(f'./prediction-models/model-{ticker}--30d-ws30.h5')
    elif gap_end > 30 and gap_end <=180:
        future = 180
        window_size = 60
        model = load_model(f'./prediction-models/model-{ticker}--180d-ws60.h5')
    elif gap_end >180 and gap_end <=365:
        future = 365
        window_size = 180
        model = load_model(f'./prediction-models/model-{ticker}--365d-ws180.h5')

    new_data = []
    for i in range(1, len(new_df) - window_size - future):
        data_predict = []
        # Get a window_size time frame for data feature
        for j in range(window_size):
          if region == "Nasdaq":
            case = 1
            data_predict.append(new_df.loc[i + j, 'Close'])
          if region == "Vietnam":
            if ticker in {'BID','CTG','TCB','VCB','VPB'}:
              case = 2
              data_predict.append(new_df.loc[i+j, ['Close','roe','roa','earningPerShare', 'payableOnEquity', 'assetOnEquity','bookValuePerShare']])
            else:
              case = 3
              data_predict.append(new_df.loc[i+j, ['Close','roe','roa','earningPerShare', 'payableOnEquity', 'assetOnEquity','debtOnEquity','grossProfitMargin','bookValuePerShare','operatingProfitMargin']])
        if case == 1:
          new_data.append(np.array(data_predict).reshape(window_size, 1))
        if case == 2:
          new_data.append(np.array(data_predict).reshape(window_size, 7))
        if case == 3:
          new_data.append(np.array(data_predict).reshape(window_size, 10))

    new_data = np.array(new_data)
    if case == 1:
      new_data = new_data.reshape(new_data.shape[0], window_size, 1)
    if case == 2:
      new_data = new_data.reshape(new_data.shape[0], window_size, 7)
    if case == 3:
      new_data = new_data.reshape(new_data.shape[0], window_size, 10)

    new_data_norm = new_data.copy()
    for i in range(0, len(new_data_norm)):
        min_feature = np.min(new_data[i])
        max_feature = np.max(new_data[i])
        new_data_norm[i] = (new_data[i] - min_feature) / (max_feature - min_feature)
     
    new_data_norm = tensorflow.convert_to_tensor(np.array(new_data_norm), dtype= tensorflow.float32)
    
    # Get prediction on the test data
    tensorflow.config.run_functions_eagerly(True)
    y_pred_norm = model.predict(new_data_norm)
    tensorflow.config.run_functions_eagerly(True)

    # Convert the result back to stock price (i.e., de-normalization) for visualization purpose
    y_pred_denorm = y_pred_norm
    for i in range(0, len(y_pred_denorm)): # denorm_x = norm_x * (max(x) - min(x)) + min(x)
        y_pred_denorm[i] = y_pred_norm[i] * (max_feature - min_feature) + min_feature

    from datetime import datetime, timedelta
    df = pd.DataFrame(y_pred_denorm[-1], columns = ['Close price'])

    dates = []
    current_date = latest
    for i in range(future):
        next_date = current_date + timedelta(days=1)
        dates.append(next_date)
        current_date = next_date

    for i in range(len(dates)):
        dates[i] = dates[i].strftime('%d %b %Y')

    df['Dates'] = pd.DataFrame(dates, columns = ['Dates'])
    df['Dates'] = df['Dates'].astype(str)
    close_prices = df['Close price'].apply("{:.2f}".format).tolist()

    # Create the line graph
    st.info("You can click on the data points for more details on close price and its corresponding date.")
    fig = px.line(df[gap_start:gap_end], x='Dates', y='Close price', markers = False, title = f'Predicted close price of {ticker} from {start_date} to {end_date}')
  #   fig.add_trace(px.scatter(df[gap_start:gap_end], x='Dates', y='Close price',
  #                           color_continuous_scale='oranges').data[0])
  #   fig.update_traces(textposition="top center")
  #   fig.update_traces(line_color=colors)
  #   fig.update_traces(marker_color='#ffa500')

    # Show the graph
    st.plotly_chart(fig)

with tab2:
  df = pd.read_csv('./search_engine_vn.csv')
  df_search = df['company']
  company = st.selectbox("Choose a company", df_search, index=0)
  col1, col2= st.columns([1,2])
  with col1: 
    price = st.number_input("Close Price", step = 0.1)
    eps = st.number_input("Earning per Share", step = 0.1)
    opm = st.number_input("Operating Profit Margin", step = 0.1)
    roe = st.number_input("ROE", step = 0.2)
    doe = st.number_input("Debt on Equity", step = 0.1)
    aoe = st.number_input("Asset on Equity", step = 0.1)
    roa = st.number_input("ROA", step = 0.1)
    gpm = st.number_input("Gross Profit Margin", step = 0.1)
    poe = st.number_input("Payable on Equity", step = 0.1)
  with col2: 
    scores = [0.3, 0.6, 0.1]
    action = ['Sell','Hold','Buy']
    fin = pd.DataFrame(columns = ['scores','action'])
    fin['scores'] = scores
    fin['action']=action
    fig = px.pie(fin, values='scores', names='action')
    st.plotly_chart(fig)
  
