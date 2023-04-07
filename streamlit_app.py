import streamlit as st

region = st.selectbox(
     'Choose a region?',
     ('Nasdaq', 'Vietnam'))

company = st.text_input("Search companies by name or ticker", value="")
