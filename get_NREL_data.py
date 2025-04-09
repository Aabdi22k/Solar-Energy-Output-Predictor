import requests
import pandas as pd
from meteostat import Point, Daily 
import time
import io

# --- Request NREL NSRDB GHI Data --- 
def fetch_nrel_data(lat, lon, years, api_key, sleep_time=5):
    """
    Fetch solar resource data from NREL NSRDB for multiple years using the GOES Aggregated PSM v4 endpoint.
    This function makes a separate request for each year, processes the CSV response (skipping the first two header rows),
    and then concatenates the data into a single pandas DataFrame.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        years (list): List of years (as integers or strings) to fetch data for (each between 1998 and 2023).
        api_key (str): Your NREL API key.
        sleep_time (int, optional): Seconds to wait between requests to avoid rate limiting. Default is 1 second.

    Returns:
        pandas.DataFrame: A combined DataFrame containing data for all the requested years, or None if no data was fetched.
    """
    base_url = "https://developer.nrel.gov/api/nsrdb/v2/solar/nsrdb-GOES-aggregated-v4-0-0-download.csv"
    combined_dfs = []
    
    for year in years:
        params = {
            "api_key": api_key,
            "wkt": f"POINT({lon} {lat})",
            "names": str(year),  # Send one year at a time as a string
            "leap_day": "false",
            "interval": "30",
            "utc": "true",
            "email": "farahaabdi22@gmail.com",  # Replace with your email
            "attributes": "ghi,dhi,dni,air_temperature,wind_speed"
        }
        
        print(f"Requesting NSRDB data for year {year}...")
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            try:
                # Process CSV directly from the response text while skipping header rows.
                df_year = pd.read_csv(io.StringIO(response.text), skiprows=2)
            except Exception as e:
                print(f"Error processing CSV data for year {year}: {e}")
                continue
            combined_dfs.append(df_year)
            print(f"Data for year {year} fetched successfully.")
        else:
            print(f"Error fetching data for year {year}: {response.status_code} {response.text}")
        
        # Pause between requests to help avoid rate limiting.
        time.sleep(sleep_time)
    
    if combined_dfs:
        combined_df = pd.concat(combined_dfs, ignore_index=True)
        combined_df.to_csv("NREL_Data.csv", index=False)
        print("Combined NSRDB data saved to NREL_Data.csv")
        return combined_df
    else:
        print("No data was fetched.")
        return None
    

