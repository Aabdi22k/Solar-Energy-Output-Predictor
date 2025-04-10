import streamlit as st
import os
import datetime
import joblib
from get_data import get_data
from train_model import train_and_evaluate_model
from predict import predict
from calculation import (
    calculate_solar_output_MAE,
    calculate_solar_output_1std,
    calculate_solar_output_2std,
    calculate_solar_output_3std
)
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib as plt

# Directory to save models
MODEL_DIR = "saved_models"
FORECAST_DIR = "saved_forecasts"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(FORECAST_DIR, exist_ok=True)

# Cache trained models
@st.cache_resource
def load_or_train_model(lat, lon):
    model_path = os.path.join(MODEL_DIR, f"model_{lat}_{lon}.pkl")
    scaler_path = os.path.join(MODEL_DIR, f"scaler_{lat}_{lon}.pkl")
    metadata_path = os.path.join(MODEL_DIR, f"meta_{lat}_{lon}.pkl")

    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(metadata_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        meta = joblib.load(metadata_path)
        mae = meta["mae"]
        std = meta["std"]
        accuracies = meta["accuracies"]
    else:
        merged_fe_df, _ = get_data(lat, lon)
        model, scaler, mae, std, accuracies = train_and_evaluate_model(merged_fe_df)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)
        joblib.dump({"mae": mae, "std": std, "accuracies": accuracies}, metadata_path)
    return model, scaler, mae, std, accuracies

# Cache Forecasts
@st.cache_resource
def load_or_fetch_forecast(lat, lon):
    # Get today's date
    today = datetime.date.today() + datetime.timedelta(days=1)

    # Forecast is for next 5 days
    start_date = today
    end_date = today + datetime.timedelta(days=4)  # 5 days total

    # Format dates nicely
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")

    # Build the filename
    forecast_filename = f"forecast_{lat}_{lon}_{start_date_str}_to_{end_date_str}.csv"

    forecast_path = os.path.join(FORECAST_DIR, forecast_filename)

    if os.path.exists(forecast_path):
        forecast_df = pd.read_csv(forecast_path)
    else:
        _, forecast_df = get_data(lat, lon)
        forecast_df.to_csv(forecast_path, index=False)
    return forecast_df

# UI Layout
st.set_page_config(page_title="Solar Output Predictor", layout="wide")

left_col, right_col = st.columns([1, 1])

with left_col:
    st.title("☀️ Solar Power Output Predictor")

    # --- Input Form ---
    with st.form(key="input_form"):
        st.subheader("Enter Your Solar System Info")
        
        # Disabled Location Inputs
        lat = st.number_input(
            "Latitude", value=33.448376, disabled=True,
            help="ℹ️ Only Phoenix, Arizona is available right now. More locations coming soon!"
        )
        lon = st.number_input(
            "Longitude", value=-112.074036, disabled=True,
            help="ℹ️ Only Phoenix, Arizona is available right now. More locations coming soon!"
        )

        array_size = st.number_input("Solar Array Size (m^2)", min_value=1.0, value=10.0, step=0.5)
        panel_efficiency = st.number_input("Panel Efficiency (Eg. 0.15 = 15%)", min_value=0.01, max_value=1.0, value=0.15, step=0.01)
        submit_button = st.form_submit_button(label="Predict Solar Output")

    if submit_button:
        processing_message = st.empty()
        processing_message = st.success("Processing your request...")

        # Load or train model
        model, scaler, mae, std, accuracies = load_or_train_model(lat, lon)
        forecast_df = load_or_fetch_forecast(lat, lon)

        # Predict GHI
        ghi_values = predict(model, scaler, forecast_df)

        # Calculate Solar Output
        output_mae = calculate_solar_output_MAE(array_size, panel_efficiency, ghi_values, mae)
        output_1std = calculate_solar_output_1std(array_size, panel_efficiency, ghi_values, std)
        output_2std = calculate_solar_output_2std(array_size, panel_efficiency, ghi_values, std)
        output_3std = calculate_solar_output_3std(array_size, panel_efficiency, ghi_values, std)

        

        today = datetime.date.today() + datetime.timedelta(days=1)

        # --- Results ---
        st.subheader("🔎 Solar Output Predictions for Next 5 Days")
        st.write("Predictions are given as ranges based on model uncertainty:")

        # --- Model Accuracies ---
        st.subheader("📈 Model Accuracy Metrics")

        # Make the accuracies show up nicely in one row
        col1, col2, col3, col4 = st.columns(4)

        col1.metric("\u00B1MAE Accuracy", f"{accuracies['\u00B1MAE']:.2f}%")
        col2.metric("\u00B11 STD Accuracy", f"{accuracies['\u00B11std']:.2f}%")
        col3.metric("\u00B12 STD Accuracy", f"{accuracies['\u00B12std']:.2f}%")
        col4.metric("\u00B13 STD Accuracy", f"{accuracies['\u00B13std']:.2f}%")
        st.markdown("---")
        # --- Forecast Outputs ---

        # Now make a 3-column layout for your forecast outputs
        cols = st.columns(3)  # Creates 3 columns across the page

        for i in range(len(output_mae)):
            # Cycle through the columns: 0, 1, 2, 0, 1, 2, etc.
            col = cols[i % 3]

            with col:
                forecast_date = today + datetime.timedelta(days=i)
                st.markdown(f"### {forecast_date.strftime('%B %d, %Y')}")
                st.write(f"**GHI Prediction:** {(ghi_values[i]/1000):.2f} KW/m²")

                st.markdown("*Solar Power Output*")
                st.write(f"- Within MAE: {output_mae[i][0]:.2f} kWh to {output_mae[i][1]:.2f} kWh")
                st.write(f"- Within 1 STD: {output_1std[i][0]:.2f} kWh to {output_1std[i][1]:.2f} kWh")
                st.write(f"- Within 2 STD: {output_2std[i][0]:.2f} kWh to {output_2std[i][1]:.2f} kWh")
                st.write(f"- Within 3 STD: {output_3std[i][0]:.2f} kWh to {output_3std[i][1]:.2f} kWh")
                st.markdown("---")
        
        processing_message.empty()

        

        
# --- About Section on the Right ---
with right_col:
    st.title("About ")
    st.markdown("""
    **Solar Output Predictor** uses machine learning models trained on historical **Global Horizontal Irradiance (GHI)** and weather data to predict solar power output for the next 5 days. This application is designed to help homeowners, businesses, and solar energy enthusiasts predict how much energy their solar panels will generate based on future weather forecasts. The app currently supports Phoenix, Arizona, but is expected to expand to other global locations soon.

    ## How it Works:
    The Solar Output Predictor relies on the following steps:
    1. **Data Collection**: 
        - Historical GHI (solar radiation data) and weather forecast data for the next 5 days.
        - The application uses **NREL's NSRDB** for historical **GHI** data and **OpenMeteo** for both **historical weather data** and **forecast data**.
    2. **Model Training**: 
        - A machine learning model, specifically a **Random Forest Regressor**, is trained on the collected data. The model learns the relationship between weather patterns and solar energy output.
        - It evaluates accuracy using **Mean Absolute Error (MAE)** and **Standard Deviation (STD)** metrics, providing users with a measure of confidence in the predictions.
    3. **Prediction**:
        - Using the trained model, the application predicts the **Global Horizontal Irradiance (GHI)** for the next 5 days based on the forecasted weather data.
        - The output is then scaled to provide estimated **solar power output**, accounting for panel efficiency and size of the solar array.
        - The predicted output is provided within **confidence intervals**: \u00B1MAE, \u00B11 STD, \u00B12 STD, and \u00B13 STD to give users an understanding of the possible range of solar energy production.

    ## Key Features:
    - **Solar Output Predictions**: Displays solar energy output predictions for the next 5 days with uncertainty bands (\u00B1MAE, \u00B11 STD, \u00B12 STD, \u00B13 STD).
    - **Machine Learning Model**: Uses a trained Random Forest model for predicting solar power output.
    - **Forecast Data**: Real-time weather forecast data and GHI predictions.
    - **Accurate Results**: Provides accuracies for the model’s performance (within MAE and STD bands).
    - **Solar Array Size & Efficiency Input**: Users can input their own solar array size (in kW) and panel efficiency, allowing for customized results.

    ## Supported Locations:
    - **Current Location**: Phoenix, Arizona (Lat: 33.448376, Lon: -112.074036)
    - **Future Expansion**: The app will soon support more global locations with solar data from the **NREL NSRDB** database. Future versions will include features like panel tilt and azimuth angle for more precise predictions.

    ## Future Improvements:
    - **Global Location Support**: The ability to predict solar output for any location around the world using the corresponding GHI and weather data.
    - **Panel Settings**: Incorporate more detailed solar panel settings such as tilt and azimuth to improve the accuracy of predictions.
    - **Live API Updates**: Integration with live APIs to pull real-time weather data for continuous updates.
    - **Enhanced User Interface**: More intuitive interface with additional features like interactive maps and user-defined regions.
    - **Integration with Solar Equipment**: Potential for integration with solar inverter systems to fetch real-time data and continuously improve predictions.
    - **Mobile App Version**: A future version of this app will be available as a mobile app for easy access on-the-go.

    ## Technologies Used:
    - **Python**: The primary language for developing the application and machine learning model.
    - **Streamlit**: A powerful and easy-to-use library to create interactive web apps. It handles the UI for user inputs and visualizations.
    - **Scikit-learn**: For implementing the machine learning model (Random Forest Regressor).
    - **Pandas**: For handling data processing and manipulation.
    - **Numpy**: For numerical operations.
    - **NREL NSRDB**: For historical **Global Horizontal Irradiance (GHI)** data.
    - **OpenMeteo API**: For both **historical weather data** and **weather forecasts**.

    ## Contact:
    - For any questions, suggestions, or collaboration opportunities, feel free to reach out to [Your Email] or visit [Your GitHub].

    """)
