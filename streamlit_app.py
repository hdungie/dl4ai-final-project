import streamlit as st
import datetime

# col1, col2 = st.columns([1,3])
# with col1: region = st.selectbox('Choose a region?', ('Nasdaq', 'Vietnam'))
  
sheet_id = "1eCdfzIPQRilLy_twlJqvtgjXGtC5cevSb4xgRRwFnUg"
sheet_name = "Nasdaq"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
df = pd.read_csv(url, dtype=str)
st.write(df)
# with col2: company = st.text_input("Search companies by name or symbol", value="")

col1, col2 = st.columns(2)
with col1: start_date = st.date_input( "Start date: ")
with col2: end_date = st.date_input("End date: ")

if end_date < start_date:
  st.warning("Invalid end date. Please choose another date.", icon = "❌")
