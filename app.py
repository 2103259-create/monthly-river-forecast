
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
import pmdarima as pm

# === Load Data ===
@st.cache_data
def load_data(river):
    file_map = {
        "Sungai Kelantan": "data/Sungai Kelantan.txt",
        "Sungai Sokor": "data/Sungai Sokor.txt"
    }
    with open(file_map[river], 'r') as f:
        lines = f.readlines()

    data = []
    current_year = None
    for line in lines:
        line = line.strip()
        if line.startswith('Daily means') and 'Year' in line:
            current_year = int(line.split('Year')[1].split()[0])
        elif line and line[0].isdigit():
            parts = line.split()
            if len(parts) >= 13 and current_year:
                day = int(parts[0])
                for i, value in enumerate(parts[1:13]):
                    try:
                        val = float(value)
                        date = pd.Timestamp(year=current_year, month=i+1, day=day)
                        data.append((date, val))
                    except:
                        continue
    df = pd.DataFrame(data, columns=['Date', 'Flow'])
    df.set_index('Date', inplace=True)
    df = df.resample('M').mean()
    df['Flow_log'] = np.log(df['Flow'] + 1)
    return df

# === Forecast Functions ===
def forecast_arima(df, river_name, steps=30):
    model = joblib.load(f"models/arima_{river_name.lower().replace(' ', '_')}.pkl")
    forecast = model.predict(n_periods=steps)
    last_30 = df['Flow'].values[-30:]
    return last_30, forecast

def forecast_lstm(df, river_name, steps=30):
    scaler = joblib.load(f"models/scaler_{river_name.lower().replace(' ', '_')}.pkl")
    model = load_model(f"models/model_{river_name.lower().replace(' ', '_')}_lstm.keras")

    input_seq = df['Flow_log'].values[-30:]
    input_scaled = scaler.transform(input_seq.reshape(-1,1)).flatten().tolist()
    forecast_scaled = []

    for _ in range(steps):
        x_input = np.array(input_scaled[-30:]).reshape(1, 30, 1)
        yhat = model.predict(x_input, verbose=0)[0][0]
        forecast_scaled.append(yhat)
        input_scaled.append(yhat)

    forecast_unscaled = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1,1)).flatten()
    forecast = np.exp(forecast_unscaled) - 1
    last_30 = np.exp(input_seq) - 1
    return last_30, forecast

# === Streamlit UI ===
st.title("🌊 Kelantan River Forecasting System")
st.markdown("This app forecasts river flow using ARIMA or LSTM models for **Sungai Kelantan** and **Sungai Sokor**.")

river = st.selectbox("Select River:", ["Sungai Kelantan", "Sungai Sokor"])
model_type = st.radio("Select Model:", ["ARIMA", "LSTM"])

if st.button("Run Forecast"):
    df = load_data(river)
    if model_type == "ARIMA":
        last_30, forecast = forecast_arima(df, river)
    else:
        last_30, forecast = forecast_lstm(df, river)

    # Plot
    fig, ax = plt.subplots(figsize=(14,6))
    ax.plot(np.arange(30), last_30, label="Last 30 Actual", color="blue")
    ax.plot(np.arange(30, 60), forecast, label="Forecast", color="orange")
    ax.set_title(f"30-Day Forecast - {river}")
    ax.set_xlabel("Days")
    ax.set_ylabel("Flow (m³/s)")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
