# ☀️ Solar Energy Output Predictor

**Solar Energy Output Predictor** is a machine learning-powered web application that predicts daily solar panel energy output based on real-world solar radiation and weather data. It helps users estimate how much energy their solar panels will generate over the next 5 days.

Built for renewable energy enthusiasts, data scientists, and solar engineers who want quick, accurate insights into solar energy production.

## 🔥 Demo

[![Streamlit App](https://img.shields.io/badge/Streamlit-Live--Demo-orange?logo=streamlit)](http://3.23.124.179:8501/)

## 🚀 Features

- **📈 5-Day Solar Output Forecast**  
  Predicts solar panel output using real-world weather and solar radiation data.

- **🛠️ Custom Panel Configuration**  
  Input your **solar array size** (m^2) and **panel efficiency** (%) for personalized predictions.

- **🌎 Location Input (Work in Progress)**  
  Currently available: **Phoenix, Arizona**.  
  (Working on adding more locations globally!)

- **🎯 Error Analysis**  
  Shows predictions within **Mean Absolute Error (MAE)** and **Standard Deviation (STD)** bands.

- **⚡ Cached Models**  
  Saves and reloads trained models for faster predictions.

- **🌐 Web App**  
  Built with **Streamlit** for a fast and beautiful user experience.

## 🛠️ Tech Stack

| Tool/Library | Purpose |
|--------------|---------|
| **Python** | Core programming language |
| **Streamlit** | Web application frontend |
| **Scikit-learn** | Machine learning (Random Forest Regressor) |
| **Matplotlib & Seaborn** | Visualization and plotting |
| **Joblib** | Model serialization (saving and loading) |
| **Open-Meteo API** | Historical and forecast weather data |
| **NREL NSRDB API** | Historical solar radiation (GHI) data |

## 📥 Data Sources

- **Solar Radiation (GHI)**: [NREL NSRDB](https://nsrdb.nrel.gov/)
- **Historical & Forecast Weather**: [Open-Meteo Historical API](https://open-meteo.com/)

## 📋 How It Works

1. **Data Collection**  
   - Fetch GHI and weather data from APIs for a given latitude and longitude.

2. **Model Training**  
   - Train a **Random Forest Regressor** on historical GHI + weather features.
   - Cache/save the model for future use.

3. **Prediction**  
   - Use forecasted weather to predict GHI for the next 5 days.

4. **Solar Output Calculation**  
   - Convert GHI predictions into **solar energy output** based on user panel settings.

5. **Error Analysis**  
   - Show output bands based on model MAE and standard deviation.

## 🧠 Future Improvements

- 🌍 **Global Location Support**  
  Allow users to input any latitude/longitude.

- 🧮 **Advanced Panel Settings**  
  Add panel tilt, azimuth angle, and degradation factors.

- 🛰️ **Real-Time API Updates**  
  Continuously refresh forecasts for live predictions.

- 📈 **More Machine Learning Models**  
  Explore XGBoost, LightGBM, and Deep Neural Networks for even better accuracy.

- 📡 **API Service**  
  Provide predictions via an API endpoint.

## 🧑‍💻 Developer Info

Developed by Farah Abdi.

If you like this project, consider giving it a ⭐ and following me for more!
