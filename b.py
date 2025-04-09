import requests
import pandas as pd
from datetime import datetime
from typing import Optional

def get_daily_ghi(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches hourly GHI data from Open-Meteo API and aggregates it to daily values.

    Parameters:
        lat (float): Latitude of the location (-90 to 90).
        lon (float): Longitude of the location (-180 to 180).
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: DataFrame with daily aggregated GHI values.

    Raises:
        ValueError: If input parameters are invalid
        requests.RequestException: If API request fails
    """
    # Input validation
    if not -90 <= lat <= 90:
        raise ValueError("Latitude must be between -90 and 90 degrees")
    if not -180 <= lon <= 180:
        raise ValueError("Longitude must be between -180 and 180 degrees")
    
    try:
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')
    except ValueError:
        raise ValueError("Dates must be in 'YYYY-MM-DD' format")

    # Define API URL
    url = "https://archive-api.open-meteo.com/v1/archive"

    # API request parameters
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "shortwave_radiation",
        "timezone": "auto",
    }

    try:
        # Make the API request
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
    except requests.RequestException as e:
        raise requests.RequestException(f"Failed to fetch data from API: {str(e)}")

    # Extract hourly timestamps and GHI values
    timestamps = data["hourly"]["time"]
    ghi_values = data["hourly"]["shortwave_radiation"]

    # Create a DataFrame
    df = pd.DataFrame({"timestamp": timestamps, "GHI": ghi_values})

    # Convert timestamp to datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    # Extract date from timestamp and aggregate GHI values
    df["date"] = df["timestamp"].dt.date
    df_daily = df.groupby("date")["GHI"].sum().reset_index()

    return df_daily

def save_ghi_data(df: pd.DataFrame, filename: str) -> None:
    """
    Save GHI data to a CSV file.

    Parameters:
        df (pd.DataFrame): DataFrame containing GHI data
        filename (str): Name of the file to save (without extension)
    """
    df.to_csv(f"{filename}.csv", index=False)
    print(f"Data saved to {filename}.csv")

# Example Usage
if __name__ == "__main__":
    try:
        # Example coordinates for Columbus, Ohio
        df_ghi = get_daily_ghi(39.9612, -82.9988, "2023-01-01", "2023-12-31")
        print("First few rows of GHI data:")
        print(df_ghi.head())
        
        # Save the data
        save_ghi_data(df_ghi, "columbus_ghi_2023")
    except Exception as e:
        print(f"Error: {str(e)}")