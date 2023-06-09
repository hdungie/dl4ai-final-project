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
    col1, col2, col3, col4, col5 = st.columns(5)
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
    fig = px.line(df[gap_start:gap_end], x='Dates', y='Close price', markers = False, title = f':red[Predicted close price of {ticker} from {start_date} to {end_date}]')
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
  
  col1, col2, col3= st.columns([1,1,2.5])
  comp = company.split(' ')
  ticker = comp[0]
  finance = pd.read_csv(f'./data-vn/history/{ticker}.csv')
  
  with col1: 
    price = st.number_input("Close Price", step = 0.1)
    price = (price - np.min(finance['Close'])) / (np.max(finance['Close']) - np.min(finance['Close']))
    
    eps = st.number_input("Earning per Share", step = 0.1)
    eps = (eps - np.min(finance['earningPerShare'])) / (np.max(finance['earningPerShare']) - np.min(finance['earningPerShare']))
    
    opm = st.number_input("Operating Profit Margin", step = 0.1)
    
    roe = st.number_input("ROE", step = 0.2)
    roe = (roe - np.min(finance['roe'])) / (np.max(finance['roe']) - np.min(finance['roe']))
    
    doe = st.number_input("Debt on Equity", step = 0.1)
  with col2:
    aoe = st.number_input("Asset on Equity", step = 0.1)
    aoe = (aoe - np.min(finance['assetOnEquity'])) / (np.max(finance['assetOnEquity']) - np.min(finance['assetOnEquity']))
    
    roa = st.number_input("ROA", step = 0.1)
    roa = (roa - np.min(finance['roa'])) / (np.max(finance['roa']) - np.min(finance['roa']))
    
    gpm = st.number_input("Gross Profit Margin", step = 0.1)
    
    poe = st.number_input("Payable on Equity", step = 0.1)
    poe = (poe - np.min(finance['payableOnEquity'])) / (np.max(finance['payableOnEquity']) - np.min(finance['payableOnEquity']))
    
    st.text("")
    st.text("")
    if (price is not None) and (eps is not None) and (opm is not None) and (roe is not None) and (doe is not None) and (aoe is not None) and (gpm is not None) and (poe is not None):
      generate_button = st.button("Generate")
    else: generate_button = st.button("Generate", disabled = True)
  
  with col3: 
    if generate_button:
        model = load_model(f"./tradingpoint-models/model-{ticker}--tradingpoint.h5")
        new_data = []
        if ticker in {'BID','CTG','TCB','VCB','VPB'}:
          new_data.append([price, roe, roa, eps, aoe, poe])
          new_data = np.array(new_data)
          new_data = new_data.reshape(new_data.shape[0],1,6)
          
          new_data_norm = tensorflow.convert_to_tensor(np.array(new_data), dtype= tensorflow.float32)
          y_pred_norm = model.predict(new_data_norm)

          scores = y_pred_norm[0][0]
          action = ['Buy','Sell','Hold']
          fin = pd.DataFrame(columns = ['scores','action'])
          fin['scores'] = scores
          fin['action']=action
          color_mapping = {'Buy': 'green', 'Sell': 'red', 'Hold':'yellow'}
          fig = px.pie(fin, values='scores', names='action', color_discrete_sequence = list(color_mapping.values()))
          st.plotly_chart(fig)
          
        else:
          opm = (opm - np.min(finance['operatingProfitMargin'])) / (np.max(finance['operatingProfitMargin']) - np.min(finance['operatingProfitMargin']))
          doe = (doe - np.min(finance['debtOnEquity'])) / (np.max(finance['debtOnEquity']) - np.min(finance['debtOnEquity']))
          gpm = (gpm - np.min(finance['grossProfitMargin'])) / (np.max(finance['grossProfitMargin']) - np.min(finance['grossProfitMargin']))
          
          new_data.append([price, roe, roa, eps, doe, gpm, opm, aoe, poe])
          new_data = np.array(new_data)
          new_data = new_data.reshape(new_data.shape[0],1,9)

          new_data_norm = tensorflow.convert_to_tensor(np.array(new_data), dtype= tensorflow.float32)
          y_pred_norm = model.predict(new_data_norm)

          scores = y_pred_norm[0][0]
          action = ['Buy','Sell','Hold']
          fin = pd.DataFrame(columns = ['scores','action'])
          fin['scores'] = scores
          fin['action']=action
          color_mapping = {'Buy': 'green', 'Sell': 'red', 'Hold':'yellow'}
          fig = px.pie(fin, values='scores', names='action', color_discrete_sequence = list(color_mapping.values()))
          st.plotly_chart(fig)

with tab3:
  df = pd.read_csv('./search_engine_vn.csv')
  df_search = df['company']
  company = st.selectbox("Select a company", df_search, index=0)
  comp = company.split(' ')
  ticker = comp[0]
  col1, col2, col3, col4 = st.columns([1,1,1,1])
  with col1:
      last_quarter = st.number_input("Choose the start quarter", min_value = 1, max_value =4, step =1)
  with col2:
    if ticker in {'TCB','VPB','VHM'}:
      last_year = st.number_input("Choose the start year", min_value = 2018, max_value = 2022, step = 1)
    else:
      last_year = st.number_input("Choose the start year", min_value = 2015, max_value = 2022, step = 1)
  with col3:
    quarter = st.number_input("Choose the end quarter", min_value = 1, max_value =4, step =1)
  with col4:
    if ticker == "BID" and quarter == 4:
      year = st.number_input("Choose the end year", min_value = 2016, max_value = 2021, step = 1)
    else:
      year = st.number_input("Choose the end year", min_value = 2016, max_value = 2022, step = 1)
    
  col1, col2, col3 = st.columns([1,1,1])
  with col1:
    pass
  with col3:
    pass
  with col2 :
      generate_button = st.button('Generate', key = 0)
  
  if generate_button:
    new_data = []
    new_df = pd.read_csv('./data-portfolio-management.csv')

    for i in range(len(new_df)):
      if quarter == new_df['quarter'][i] and year == new_df['year'][i] and ticker == new_df['ticker'][i]:
        current_quarter = quarter
        current_year = year

        pte = round(new_df['pte'][i],3)
        pte_delta = round(pte - new_df['pte'][i-1],3)

        ptb = round(new_df['ptb'][i],3)
        ptb_delta = round(ptb - new_df['ptb'][i-1],3)

        roe = round(new_df['roe'][i],3)
        roe_delta = round(roe - new_df['roe'][i-1],3)

        roa = round(new_df['roa'][i],3)
        roa_delta = round(roa - new_df['roa'][i-1],3)

        epsC = round(new_df['epsChange'][i],3)
        epsC_delta = round(epsC - new_df['epsChange'][i-1],3)

        bvpsC = round(new_df['bookValuePerShareChange'][i],3)
        bvpsC_delta = round(bvpsC - new_df['bookValuePerShareChange'][i-1],3)

        poe = round(new_df['payableOnEquity'][i],3)
        poe_delta = round(poe - new_df['payableOnEquity'][i-1],3)

        eoa = round(new_df['equityOnAsset'][i],3)
        eoa_delta = round(eoa - new_df['equityOnAsset'][i-1],3)

        new_data.append([pte,ptb,roe,roa,epsC, bvpsC,poe,eoa])
        new_data = np.array(new_data)
        new_data = new_data.reshape(new_data.shape[0], 1,8)
        new_data = tensorflow.convert_to_tensor(np.array(new_data), dtype=tensorflow.float32)

        model = load_model("portfolio-management.h5")
        y_pred = model.predict(new_data)

        scores = y_pred[0][0]
        action = ['Potential','Risk']
        fin = pd.DataFrame(columns = ['scores','action'])
        fin['scores'] = scores
        fin['action']=action

    st.info("Scroll down to see the metrics")

    column1, column2 =st.columns([3,2])
    with column1:
      if fin['scores'][0] > fin['scores'][1]:
        st.subheader(f'In quarter {quarter} of {year}, the company {ticker} is:')
        st.title("✔️ :green[Potential]")
      else: 
        st.subheader(f'In quarter {quarter} of {year}, the company {ticker} is:')
        st.title("❌ :red[Risk]")

      color_mapping = {'Potential': 'green', 'Risk': 'red'}
      fig = px.bar(fin, x="scores", y="action", orientation='h', color = "action", color_discrete_map = color_mapping, width = 400, height = 200)
      fig.update_layout(showlegend=False)
      st.plotly_chart(fig)

    with column2:
      history = pd.read_csv(f'./data-vn/history/{ticker}.csv')
      current = str('quarter') + "/" + str('year')

      if quarter == 1:
        date = f'1/{year}'
      elif quarter == 2:
        date = f'4/{year}'
      elif quarter == 3:
        date = f'7/{year}'
      else:
        date = f'10/{year}'
      date = datetime.strptime(date, '%m/%Y')

      if last_quarter == 1:
        last_date = f'1/{last_year}'
      elif last_quarter == 2:
        last_date = f'4/{last_year}'
      elif last_quarter == 3:
        last_date = f'7/{last_year}'
      else:
        last_date = f'10/{last_year}'
      last_date = datetime.strptime(last_date, '%m/%Y')

      current_price = []
      for i in range(len(history)):
        his_date = datetime.strptime(history['Date'][i], '%Y-%m-%d')
        if date.month == his_date.month and date.year == his_date.year:
          current_price.append(history['Close'][i])

      last_price = []
      for i in range(len(history)):
        his_date = datetime.strptime(history['Date'][i], '%Y-%m-%d')
        if last_date.month == his_date.month and last_date.year == his_date.year:
          last_price.append(history['Close'][i])

      now_price = np.max(current_price)
      l_price = np.max(last_price)

      col1, col2, col3 = st.columns([1,1,1])
      with col1:
        pass
      with col2:
        st.title("Profit")
      with col3:
        pass

      profit = (now_price - l_price)
      if profit >= 0:
        st.header(f'⬆️ :green[{profit} VND]')
      else:
        st.header(f'⬇️ :red[{profit} VND]')

    st.subheader(f"In comparison with quarter {last_quarter} of {last_year}, the metrics of the quarter {quarter} of {year} is: ")  
    col1, col2, col3, col4 = st.columns([1,1,1,1])
    with col1:
      st.metric("Price To Earning", pte, pte_delta)
      st.metric("EPS Change", epsC, epsC_delta)
    with col2:
      st.metric("Price To Book",  ptb, ptb_delta)
      st.metric("Book Value Per Share Change",  bvpsC, bvpsC_delta)
    with col3:
      st.metric("ROE", roe, roe_delta)
      st.metric("Payable On Equity", poe, poe_delta)
    with col4:
      st.metric("ROA", roa, "0.5")
      st.metric("Equity On Asset", eoa, eoa_delta)
  
