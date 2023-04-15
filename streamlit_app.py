import streamlit as st
import datetime
import pandas as pd
import tensorflow 
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt, mpld3
import mpld3
import streamlit.components.v1 as components
from mpld3 import plugins
# from predictions_graph import graph

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
    st.warning("End date must be after start date.", icon = "❌")
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
    predict_button = st.button('Predict')

comp = company.split(' ')
ticker = comp[0]
filepath = f'./filtered-data-nasdaq/csv/{ticker}.csv'

if interval < 30:
    window_size = 150
future = interval
    
data = pd.read_csv(filepath)
new_df = data[['Date', 'Close']]

new_data = []
for i in range(1, len(df) - window_size - 1):
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

model = load_model('./nasdaq-model-30d.h5')
# Get prediction on the test data
y_pred_norm = model.predict(new_data_norm)

# Convert the result back to stock price (i.e., de-normalization) for visualization purpose
y_pred_denorm = y_pred_norm
for i in range(0, len(y_pred_denorm)): # denorm_x = norm_x * (max(x) - min(x)) + min(x)
    y_pred_denorm[i] = y_pred_norm[i] * (max_feature - min_feature) + min_feature

from datetime import datetime, timedelta

dates = []
start_date = new_df.loc[len(new_df)-1,'Date']
start_date = datetime.strptime(start_date, '%d-%m-%Y').date()

current_date = start_date
for i in range(future):
    next_date = current_date + timedelta(days=1)
    dates.append(next_date)
    current_date = next_date
for i in range(len(dates)):
    dates[i] = dates[i].strftime('%d %b %Y')

df = pd.DataFrame(y_pred_denorm[-1], columns = ['Close price'])
df['Dates'] = pd.DataFrame(dates, columns = ['Dates'])
df['Dates'] = df['Dates'].astype(str)

# graph = plt.figure(figsize=(16, 8), dpi=300)
# plt.plot(df['Dates'][:future-1], df['Close price'][:future-1], label='Predicted price', marker = '.')
# plt.title(f'Close price prediction of {comp[0]} in {future} days')
# plt.ylabel('Close price in $')
# plt.xlabel('Dates')
# plt.legend()

# if predict_button:
css = """
table
{
  border-collapse: collapse;
}
th
{
  color: #ffffff;
  background-color: #000000;
}
td
{
  background-color: #cccccc;
}
table, th, td
{
  font-family:Arial, Helvetica, sans-serif;
  border: 1px solid black;
  text-align: right;
}
"""

# Create a figure and plot the data
fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
line, = ax.plot(df['Dates'][:future-1], df['Close price'][:future-1], label='Predicted price', marker='.')

# Create the labels for each point with the x and y coords
xy_data = line.get_xydata()
labels = []
for x, y in xy_data:
    html_label = f'<table border="1" class="dataframe"> <thead> <tr style="text-align: right;"> </thead> <tbody> <tr> <th>x</th> <td>{x}</td> </tr> <tr> <th>y</th> <td>{y}</td> </tr> </tbody> </table>'
    labels.append(html_label)

# Create the tooltip with the labels (x and y coords) and attach it to the line with the CSS specified
tooltip = plugins.PointHTMLTooltip(line, labels, css=css)
plugins.connect(fig, tooltip)

# Set the plot title, labels, and legend
ax.set_title(f'Close price prediction of {comp[0]} in {future} days')
ax.set_ylabel('Close price in $')
ax.set_xlabel('Dates')
ax.legend()

# Convert the plot to HTML and display it
mpld3.display(fig)

#   fig_html = mpld3.fig_to_html(fig)
#   components.html(fig_html, height=600)
#   st.pyplot(fig)
