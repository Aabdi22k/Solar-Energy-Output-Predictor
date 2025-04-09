import requests
import pandas as pd
from datetime import datetime, timedelta
def fetch_openmeteo_historical_weather(lat, lon, start_date, end_date):
    """
    Fetches daily historical weather data (last 5 years up to yesterday) from the Open-Meteo Archive API.
    Aggregates hourly data into daily values and includes sunshine_duration and daylight_duration from daily data.
    
    Returns:
        pd.DataFrame: Daily weather data with columns:
                      date, tavg, tmin, tmax, prcp, humidity, wspd, wdir, pres, cloud_cover,
                      sunshine_duration, daylight_duration.
    """
    # Format dates
    end_date_str = end_date.strftime("%Y-%m-%d")
    start_date_str = start_date.strftime("%Y-%m-%d")
    
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        
        "longitude": lon,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m,pressure_msl,cloudcover,precipitation",
        "daily": "sunshine_duration,daylight_duration",
        "timezone": "auto"
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        hourly_data = data.get("hourly", {})
        if not hourly_data:
            print("No hourly data found.")
            return None
        
        # Process hourly data and aggregate to daily values.
        df_hourly = pd.DataFrame(hourly_data)
        df_hourly["time"] = pd.to_datetime(df_hourly["time"])
        df_hourly["date"] = df_hourly["time"].dt.date
        
        df_daily = df_hourly.groupby("date").agg(
            tavg=("temperature_2m", "mean"),
            tmin=("temperature_2m", "min"),
            tmax=("temperature_2m", "max"),
            prcp=("precipitation", "sum"),
            humidity=("relativehumidity_2m", "mean"),
            wspd=("windspeed_10m", "mean"),
            wdir=("winddirection_10m", "mean"),
            pres=("pressure_msl", "mean"),
            cloud_cover=("cloudcover", "mean")
        ).reset_index()
        
        # Create a string version of the date for merging.
        df_daily["date_str"] = df_daily["date"].astype(str)
        
        # Merge in daily fields if available.
        daily_data = data.get("daily", {})
        if daily_data:
            df_daily_extra = pd.DataFrame(daily_data)
            df_daily_extra["time"] = pd.to_datetime(df_daily_extra["time"]).dt.date
            df_daily_extra["time_str"] = df_daily_extra["time"].astype(str)
            df_daily = df_daily.merge(
                df_daily_extra[["time_str", "sunshine_duration", "daylight_duration"]],
                left_on="date_str", right_on="time_str", how="left"
            )
            df_daily.drop(columns=["date_str", "time_str"], inplace=True)
        else:
            df_daily["sunshine_duration"] = None
            df_daily["daylight_duration"] = None
        
        # Convert date to string if desired.
        df_daily["date"] = df_daily["date"].astype(str)
        return df_daily
    else:
        print("Error fetching Open-Meteo historical data:", response.status_code, response.text)
        return None


def fetch_openmeteo_forecast_weather(lat, lon, forecast_days):
    """
    Fetches daily forecast weather data for the next N days (starting tomorrow) from the Open-Meteo API.
    Aggregates hourly data into daily values and adds daily fields for sunshine_duration and daylight_duration.
    
    Returns:
        pd.DataFrame: Daily forecast data with columns:
                      date, tavg, tmin, tmax, prcp, humidity, wspd, wdir, pres, cloud_cover,
                      sunshine_duration, daylight_duration.
    """
    from datetime import datetime, timedelta  # Ensure these are imported
    today = datetime.today()
    start_date = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = (today + timedelta(days=forecast_days)).strftime("%Y-%m-%d")
    
    base_url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,relativehumidity_2m,windspeed_10m,winddirection_10m,pressure_msl,cloudcover,precipitation",
        "daily": "sunshine_duration,daylight_duration",
        "timezone": "auto"
    }
    
    response = requests.get(base_url, params=params)
    if response.status_code == 200:
        data = response.json()
        hourly_data = data.get("hourly", {})
        if not hourly_data:
            print("No hourly forecast data found.")
            return None
        
        # Process hourly data.
        df_hourly = pd.DataFrame(hourly_data)
        df_hourly["time"] = pd.to_datetime(df_hourly["time"])
        df_hourly["date"] = df_hourly["time"].dt.date
        
        df_daily = df_hourly.groupby("date").agg(
            tavg=("temperature_2m", "mean"),
            tmin=("temperature_2m", "min"),
            tmax=("temperature_2m", "max"),
            prcp=("precipitation", "sum"),
            humidity=("relativehumidity_2m", "mean"),
            wspd=("windspeed_10m", "mean"),
            wdir=("winddirection_10m", "mean"),
            pres=("pressure_msl", "mean"),
            cloud_cover=("cloudcover", "mean")
        ).reset_index()
        
        # Prepare key for merging.
        df_daily["date_str"] = df_daily["date"].astype(str)
        
        # Merge in daily fields from forecast data.
        daily_data = data.get("daily", {})
        if daily_data:
            df_daily_extra = pd.DataFrame(daily_data)
            df_daily_extra["time"] = pd.to_datetime(df_daily_extra["time"]).dt.date
            df_daily_extra["time_str"] = df_daily_extra["time"].astype(str)
            df_daily = df_daily.merge(
                df_daily_extra[["time_str", "sunshine_duration", "daylight_duration"]],
                left_on="date_str", right_on="time_str", how="left"
            )
            df_daily.drop(columns=["date_str", "time_str"], inplace=True)
        else:
            df_daily["sunshine_duration"] = None
            df_daily["daylight_duration"] = None
        
        # Convert date back to string.
        df_daily["date"] = df_daily["date"].astype(str)
        return df_daily
    else:
        print("Error fetching Open-Meteo forecast data:", response.status_code, response.text)
        return None
        # Function returns None if there was an error with the API request
        # Otherwise returns a pandas DataFrame with the following columns:
        # - date: Date in YYYY-MM-DD format
        # - tavg: Average daily temperature in Celsius
        # - tmin: Minimum daily temperature in Celsius  
        # - tmax: Maximum daily temperature in Celsius
        # - prcp: Total daily precipitation in mm
        # - humidity: Average daily relative humidity in %
        # - wspd: Average daily wind speed at 10m height in km/h
        # - wdir: Average daily wind direction at 10m height in degrees
        # - pres: Average daily mean sea level pressure in hPa
        # - cloud_cover: Average daily cloud cover in %
        # - sunshine_duration: Total daily sunshine duration in seconds (if available)
        # - daylight_duration: Total daily daylight duration in seconds (if available)
