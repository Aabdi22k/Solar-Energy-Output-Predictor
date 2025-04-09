import os
import dotenv
import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from get_NREL_data import fetch_nrel_data
from get_openmeteo_data import fetch_openmeteo_forecast_weather, fetch_openmeteo_historical_weather
from clean_process_data import agg_nrel_data, feature_engineering, merge_datasets, remove_outliers, handle_outliers

dotenv.load_dotenv()

# Perform Exploratory Data Analysis (EDA) on the merged dataset.
def perform_eda(merged_df):
    """
    The merged dataset is assumed to have the following columns:
      - date
      - Solar data: GHI
      - Meteostat weather data: tavg, tmin, tmax, prcp, snow, wdir, wspd, wpgt, pres, tsun
    
    This function creates:
      - A correlation matrix heatmap (excluding the date column)
      - Scatter plots for selected weather features vs. GHI
      - Histograms for GHI and average temperature (tavg)
    """
    
    # Set seaborn style
    sns.set(style="whitegrid", context="talk")
    
    # --- 1. Correlation Matrix ---
    plt.figure(figsize=(15, 10))
    # Drop non-numeric columns like 'date' for correlation analysis.
    corr_matrix = merged_df.drop(columns=["date"]).corr()
    sns.heatmap(corr_matrix, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix: Solar Output & Weather Features")
    plt.tight_layout()
    plt.show()
    
    # --- 2. Scatter Plots ---
    
    # Scatter Plot: Meteostat Average Temperature (tavg) vs. GHI
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="tavg", y="GHI", data=merged_df, color="b")
    plt.title("Scatter Plot: Average Temperature (tavg) vs. GHI")
    plt.xlabel("Average Temperature (°C)")
    plt.ylabel("Global Horizontal Irradiance (GHI)")
    plt.tight_layout()
    plt.show()
    
    # Scatter Plot: Precipitation (prcp) vs. GHI
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="prcp", y="GHI", data=merged_df, color="g")
    plt.title("Scatter Plot: Precipitation (prcp) vs. GHI")
    plt.xlabel("Precipitation (mm)")
    plt.ylabel("Global Horizontal Irradiance (GHI)")
    plt.tight_layout()
    plt.show()
    
    # --- 3. Histograms ---
    
    # Histogram for GHI
    plt.figure(figsize=(10, 6))
    sns.histplot(merged_df["GHI"], bins=30, kde=True, color="purple")
    plt.title("Histogram of Global Horizontal Irradiance (GHI)")
    plt.xlabel("GHI")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
    # Histogram for Meteostat Average Temperature (tavg)
    plt.figure(figsize=(10, 6))
    sns.histplot(merged_df["tavg"], bins=30, kde=True, color="orange")
    plt.title("Histogram of Average Temperature (tavg)")
    plt.xlabel("Average Temperature (°C)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

def perform_eda_2(merged_df):
    """
    The merged dataset is assumed to have the following columns:
      - date
      - Solar data: GHI
      - Meteostat weather data: tavg, tmin, tmax, prcp, snow, wdir, wspd, wpgt, pres, tsun
    
    This function creates:
      - A correlation matrix heatmap (excluding the date column)
      - Scatter plots for selected weather features vs. GHI
      - Histograms for GHI and average temperature (tavg)
    """
    
    # Set seaborn style
    sns.set(style="whitegrid", context="talk")
    
    # --- 1. Correlation Matrix ---
    plt.figure(figsize=(12, 8))
    # Drop non-numeric columns like 'date' for correlation analysis.
    corr_matrix = merged_df.drop(columns=["date"]).corr()
    sns.heatmap(corr_matrix, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix: Solar Output & Weather Features")
    plt.tight_layout()
    plt.show()
    
    # --- 2. Scatter Plots ---
    
    # Scatter Plot: Meteostat Average Temperature (tavg) vs. GHI
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="temp", y="GHI", data=merged_df, color="b")
    plt.title("Scatter Plot: Average Temperature (tavg) vs. GHI")
    plt.xlabel("Average Temperature (°C)")
    plt.ylabel("Global Horizontal Irradiance (GHI)")
    plt.tight_layout()
    plt.show()
    
    # Scatter Plot: Precipitation (prcp) vs. GHI
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="prcp", y="GHI", data=merged_df, color="g")
    plt.title("Scatter Plot: Precipitation (prcp) vs. GHI")
    plt.xlabel("Precipitation (mm)")
    plt.ylabel("Global Horizontal Irradiance (GHI)")
    plt.tight_layout()
    plt.show()
    
    # --- 3. Histograms ---
    
    # Histogram for GHI
    plt.figure(figsize=(10, 6))
    sns.histplot(merged_df["GHI"], bins=30, kde=True, color="purple")
    plt.title("Histogram of Global Horizontal Irradiance (GHI)")
    plt.xlabel("GHI")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()
    
    # Histogram for Meteostat Average Temperature (tavg)
    plt.figure(figsize=(10, 6))
    sns.histplot(merged_df["temp"], bins=30, kde=True, color="orange")
    plt.title("Histogram of Average Temperature (tavg)")
    plt.xlabel("Average Temperature (°C)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

# --- Main Pipeline ---
def get_data(lat, lon, NREL_DATA=None):

    # Define the period for historical weather data and years for NSRDB Data
    start_date = datetime.datetime(1998, 1, 1)
    end_date = datetime.datetime(2023, 12, 31)
    years = [1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    forecast_days = 5
    # API Keys for NREL and OpenWeatherMap
    NREL_API_KEY = os.getenv("NREL_API_KEY")

    # filename for NREL NSRDB Data
    filename = f"NREL_Data_{lat}_{lon}.csv"

    # 1. Fetch solar data from NREL NSRDB using the GOES Aggregated endpoint
    agg_nrel_df = None

    if os.path.exists(filename):
        print("NREL NSRDB data already exists. Skipping download.")
        agg_nrel_df = pd.read_csv(filename)
    else:
        agg_nrel_df = fetch_nrel_data(lat, lon, years, NREL_API_KEY)
        agg_nrel_df.to_csv(f"NREL_Data_{lat}_{lon}.csv", index=False)
    
    agg_nrel_df = agg_nrel_data(agg_nrel_df)

    if agg_nrel_df is not None:
        print("First few rows of the aggregated NREL data:")
        print(agg_nrel_df.head())
    else:
        print("Failed to load NREL data.")
    
    #2 Fetch historical weather data from Open Meteo
    historical_df = fetch_openmeteo_historical_weather(lat, lon, start_date, end_date)
    if historical_df is not None:
        print("First few rows of Historical Weather Data (Open Meteo) :")
        print(historical_df.head())
    # 3. Merge datasets and perform feature engineering
    merged_df = merge_datasets(agg_nrel_df, historical_df)
    merged_fe_df = feature_engineering(merged_df)

    # Handle Outliers
    # merged_fe_df = handle_outliers(merged_fe_df)
    
    print("First few rows of the merged & feature engineered dataset:")
    print(merged_fe_df.head())

    # 4. Perform EDA to analyze the merged dataset
    perform_eda(merged_fe_df)

    # 5. save the merged dataset to a CSV file for model training
    # merged_fe_df.to_csv('merged_dataset_features.csv', index=False)

    # 6. Fetch forecast weather data from Open Meteo
    today = datetime.datetime.today()
    # Forecast starts from tomorrow.
    tomorrow = (today + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
    
    forecast_df = fetch_openmeteo_forecast_weather(lat, lon, forecast_days)

    # perform feature engineering and save to csv for prediction
    forecast_df = feature_engineering(forecast_df)
    if forecast_df is not None:
        print("Sample of Open Meteo forecast data (first few rows):")
        print(forecast_df.head())
    
    forecast_df.to_csv(f'forecast_{lat}_{lon}_{tomorrow}.csv', index=False)

    return merged_fe_df, forecast_df



