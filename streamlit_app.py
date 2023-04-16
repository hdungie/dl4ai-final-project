pip install pip install -r requirements.txt
import streamlit as st
import datetime
import pandas as pd
import tensorflow 
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
import plotly.express as px
import math
import requests
import io
# from predictions_graph import graph

col1, col2 = st.columns([1,4])
with col1: region = st.selectbox('Select a region', ('--','Nasdaq', 'Vietnam'), index=0)
  
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
with col1: start_date = st.date_input( "Start date: ")
with col2: end_date = st.date_input("End date: ")

comp = company.split(' ')
ticker = comp[0]
filepath = f'./filtered-data-nasdaq/csv/{ticker}.csv'

if end_date <= start_date:
  interval = (end_date - start_date).days
  st.warning("End date must be after start date.", icon = "❌")
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
  st.warning(f"Interval: {interval} days")
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
  if interval <= 7:
      window_size = 30
      model = load_model(f'./{reg}-model-7d.h5')
  elif: interval > 7 and interval <=30:
      window_size = 150
      model = load_model(f'./{reg}-model-30d.h5')
  elif: interval > 30 and interval <=365:
      window_size = 500
      # set the shareable link for the .h5 file
      link = "https://drive.google.com/uc?id=13Jiyg6IrvYob8qFy4tJdh2-Rr2io2RNm"

      # extract the file ID from the shareable link
      file_id = link.split("=")[1]

      # set the download link for the file
      download_link = f"https://drive.google.com/uc?id={file_id}"

      # use requests to download the file
      response = requests.get(download_link)
      content = response.content

      # read the file using h5py and load the model
      with io.BytesIO(content) as f:
          with h5py.File(f, 'r') as h5_file:
              model = load_model(h5_file)
      
  future = interval

  data = pd.read_csv(filepath)
  new_df = data[['Date', 'Close']]

  new_data = []
  for i in range(1, len(new_df) - window_size - 1):
      data_predict = []
      # Get a window_size time frame for data feature
      for j in range(window_size):
          data_predict.append(new_df.loc[i + j, 'Close'])
      new_data.append(np.array(data_predict).reshape(window_size, 1))

  new_data = np.array(new_data)
  # Reshape the array to have shape (number of sequences, window_size, 1)
  new_data = new_data.reshape(new_data.shape[0], window_size, 1)

  new_data_norm = new_data.copy()
  for i in range(0, len(new_data_norm)):
      min_feature = np.min(new_data[i])
      max_feature = np.max(new_data[i])
      new_data_norm[i] = (new_data[i] - min_feature) / (max_feature - min_feature)

  # Get prediction on the test data
  y_pred_norm = model.predict(new_data_norm)

  # Convert the result back to stock price (i.e., de-normalization) for visualization purpose
  y_pred_denorm = y_pred_norm
  for i in range(0, len(y_pred_denorm)): # denorm_x = norm_x * (max(x) - min(x)) + min(x)
      y_pred_denorm[i] = y_pred_norm[i] * (max_feature - min_feature) + min_feature

  from datetime import datetime, timedelta
  df = pd.DataFrame(y_pred_denorm[-1], columns = ['Close price'])

  latest = new_df.loc[len(new_df)-1,'Date']
  latest = datetime.strptime(latest, '%d-%m-%Y').date()
  gap_end = (end_date - latest).days
  gap_start = (start_date - latest).days

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
  fig = px.line(df[gap_start:gap_end], x='Dates', y='Close price', markers = True, title = f'Predicted close price of {ticker} from {start_date} to {end_date}', text = close_prices[gap_start:gap_end])
  fig.add_trace(px.scatter(df[gap_start:gap_end], x='Dates', y='Close price',
                          color_continuous_scale='oranges').data[0])
  fig.update_traces(textposition="top center")
  fig.update_traces(line_color='#f63366')
  fig.update_traces(marker_color='#ffa500')

  # Show the graph
  st.plotly_chart(fig)

