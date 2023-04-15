import streamlit as st
import datetime
import pandas as pd

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

if end_date < start_date:
  st.warning("Invalid end date. Please choose another date.", icon = "❌")
  
col1, col2, col3 , col4, col5 = st.beta_columns(5)
with col1:
    pass
with col2:
    pass
with col4:
    pass
with col5:
    pass
with col3 :
    center_button = st.button('Predict')
st.markdown(""" div.stButton > button:first-child {
background-color: #f63366;color:white;font-size:20px;height:3em;width:30em;border-radius:10px 10px 10px 10px;
}
“”", unsafe_allow_html=True)
