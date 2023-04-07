import streamlit as st

region = st.selectbox('Choose a region?', ('Nasdaq', 'Vietnam'))

company = st.text_input("Search companies by name or ticker", value="")

interval = st.slider('Choose an interval', min_value = 1, max_value = 365, step = 1)
