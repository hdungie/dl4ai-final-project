import streamlit as st
import datetime
import pandas as pd
import predictions

col1, col2 = st.columns([1,4])
with col1: region = st.selectbox('Select a region', ('--','Nasdaq', 'Vietnam'), index=0)
  
if region == "Nasdaq":
  df = pd.read_csv('./search_engine_nasdaq.csv')
  df = df.fillna('')
else: 
  df = pd.read_csv('./search_engine_vn.csv')
  df = df.fillna('')
  
df_search = df['company']
with col2:
  company = st.selectbox("Search companies by name or symbol", df_search, index=0)

col1, col2 = st.columns(2)
with col1: start_date = st.date_input( "Start date: ")
with col2: end_date = st.date_input("End date: ")
  
if end_date > start_date:
  interval = (end_date - start_date).days
  st.warning(f"Interval: {interval} days")
if end_date < start_date:
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
    predict_button = st.button('Predict')
    
##
import os
if region == "Nasdaq":
  comp = company.split('-')
  ticker = comp[0]
  base_dir = os.path.abspath(os.path.dirname(__file__))
  file_path = os.path.join(base_dir, 'filtered-data-nasdaq', 'csv', f'{ticker}.csv')
  predictions.get_predictions(file_path, interval)

if predict_button:
  st_write(predictions.graph)
