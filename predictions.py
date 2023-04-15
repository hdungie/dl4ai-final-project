import streamlit_app
import pandas as pd
import tensorflow
from tensorflow.keras.models import load_model

# if streamlit_app.interval < 30:
window_size = 30

def get_predictions(file_path):
    data = pd.read_csv(file_path)
    df = data[['Date', 'Close']]

    new_data = []
    for i in range(1, len(df) - window_size - 1):
        data_predict = []
        # Get a window_size time frame for data feature
        for j in range(window_size):
            data_predict.append(df.loc[i + j, 'Close'])
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
    # start_date = start_date.strftime('%d %b %Y')
    dates.append(start_date)

    current_date = start_date
    for i in range(future):
        next_date = current_date + timedelta(days=1)
        dates.append(next_date)
        current_date = next_date
    for i in range(len(dates)):
        dates[i] = dates[i].strftime('%d %b %Y')

    df = pd.DataFrame(y_pred_denorm[-1], columns = ['Close price'])
    df['Dates'] = pd.DataFrame(dates, columns = ['Dates'])

    graph = plt.figure(figsize=(16, 8), dpi=300)
    plt.plot(df['Dates'], df['Close price'], label='Predicted price')
    plt.ylabel('Close price in $')
    plt.xlabel('Dates')
    plt.legend()
