import streamlit as st
import datetime
import pandas as pd

col1, col2 = st.columns([1,4])
with col1: region = st.selectbox('Choose a region?', ('Nasdaq', 'Vietnam'))
  
if region == "Nasdaq":
  df = pd.read_csv('./search_engine_nasdaq.csv')
else: 
  df = pd.read_csv('./search_engine_vn.csv')
  
df_search = df['company']
if st.session_state.get('user_input') is None:
    with col2: user_input = st.text_input("Enter your search query:")
    st.session_state['user_input'] = user_input
else:
    with col2:
        company = st.selectbox("Search companies by name or symbol", df_search)

col1, col2 = st.columns(2)
with col1: start_date = st.date_input( "Start date: ")
with col2: end_date = st.date_input("End date: ")

if end_date < start_date:
  st.warning("Invalid end date. Please choose another date.", icon = "❌")
