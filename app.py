import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
import joblib
import matplotlib.pyplot as plt
from io import BytesIO

# Set the title and favicon
st.set_page_config(page_title="StockAI", page_icon="favicon.png")

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Predict", "Train Model"])

# Global variables
model = None
scaler = None

def load_trained_model(uploaded_model_file):
    global model
    model = joblib.load(uploaded_model_file)

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        if np.isnan(dataset[i:(i + time_step + 1)]).any():
            continue
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

def predict_future(model, last_data, steps, scaler):
    future_predictions = []
    current_data = last_data.copy()
    
    for _ in range(steps):
        prediction = model.predict(current_data.reshape(1, -1))
        future_predictions.append(prediction[0])
        current_data = np.append(current_data[1:], prediction)
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Page: Predict
if page == "Predict":
    st.title("StockAI - Prediction")

    # File upload for model and CSV
    model_file = st.file_uploader("Upload your .pkl model file", type='pkl')
    csv_file = st.file_uploader("Upload your CSV data file", type='csv')

    if model_file is not None:
        load_trained_model(model_file)

    if csv_file is not None and model is not None:
        try:
            # Read the CSV file
            data = pd.read_csv(csv_file, date_parser=True)
            data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
            data.set_index('Date', inplace=True)
            data = data[['Close']]

            missing_values_count = data.isnull().sum().sum()
            if missing_values_count > 0:
                st.warning(f"The data contains missing values. These will be ignored during processing.")

            # Scale data
            data_values = data.values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data_values)

            # Set default time step
            time_step = 60
            
            # Create dataset for predictions while ignoring NaNs
            X, y = create_dataset(scaled_data, time_step)

            if len(X) == 0 or len(y) == 0:
                st.error("No valid data available for model training after ignoring missing values.")
                st.stop()

            # Fit the model on past data
            model.fit(X.reshape(X.shape[0], -1), y)

            # Make predictions for the last available dates
            predictions = model.predict(X.reshape(X.shape[0], -1))
            predictions = scaler.inverse_transform(predictions.reshape(-1, 1))

            # Plotting
            plt.figure(figsize=(14, 5))
            plt.plot(data.index, data_values, label='True Price', color='blue')
            predicted_dates = data.index[time_step:len(predictions) + time_step]
            plt.plot(predicted_dates, predictions, label='Predicted Price', color='red')
            plt.xlabel('Date')
            plt.ylabel('Stock Price (USD)')
            plt.title('Stock Price Prediction')
            plt.legend()
            plt.grid(True)

            # Show plot in Streamlit
            st.pyplot(plt)

            # Display predictions
            prediction_df = pd.DataFrame({'Date': predicted_dates, 'Predicted Price': predictions.flatten()})
            st.write(prediction_df)

            # Automatically predict for February 1, 2029
            future_date = pd.Timestamp("2029-02-01")
            days_to_predict = (future_date - data.index[-1]).days
            
            if days_to_predict > 0:
                last_data = scaled_data[-time_step:]  # Get the last available data for prediction
                future_predictions = predict_future(model, last_data, days_to_predict, scaler)
                st.write(f"Predicted Price on {future_date.date()}: ${future_predictions[-1][0]:.2f}")
            else:
                st.warning("The selected future date must be after the last date in the data.")

            # Download predictions
            csv = prediction_df.to_csv(index=False)
            st.download_button("Download Predictions", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"An error occurred: {e}")

# Page: Train Model
elif page == "Train Model":
    st.title("StockAI - Train Model")

    # File upload for CSV
    csv_file = st.file_uploader("Upload your CSV data file for training", type='csv')

    if csv_file is not None:
        try:
            # Read CSV file
            data = pd.read_csv(csv_file, date_parser=True)
            data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
            data.set_index('Date', inplace=True)
            data = data[['Close']]

            # Scale data
            data_values = data.values
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(data_values)

            # Set default parameters for model training
            time_step = 60
            n_estimators = 100
            max_depth = 10

            # Create dataset for training
            X, y = create_dataset(scaled_data, time_step)

            if len(X) == 0 or len(y) == 0:
                st.error("No valid data available for model training after ignoring missing values.")
                st.stop()

            # Train the model
            model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)
            model.fit(X, y)
            st.success("Model trained successfully!")

            # Save the model
            model_file = BytesIO()
            joblib.dump(model, model_file)
            model_file.seek(0)

            st.download_button("Download Trained Model", model_file, "trained_model.pkl", "application/octet-stream")

        except Exception as e:
            st.error(f"An error occurred: {e}")
