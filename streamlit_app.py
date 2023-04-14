import streamlit as st
import datetime

region = st.selectbox('Choose a region?', ('Nasdaq', 'Vietnam'))

company = st.text_input("Search companies by name or ticker", value="")

interval = st.slider('Choose an interval', min_value = 1, max_value = 365, step = 1)

col1, col2 = st.columns(2)
col1.header = st.date_input( "Start date: ")
col2.header = st.date_input( "End date: ")
