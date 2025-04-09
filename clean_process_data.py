import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler


# --- Aggregate NREL from 30 min intervals to daily ---
def agg_nrel_data(df):
    """
    Loads the NREL CSV file into a DataFrame, aggregates the half-hourly data to daily values,
    and returns the daily aggregated DataFrame.
    
    Assumes the CSV has two header rows before the column names:
    Year,Month,Day,Hour,Minute,GHI,DHI,DNI,Temperature,Wind Speed
    Aggregation:
      - GHI: daily sum (e.g., total irradiance)
    """
    try:        
        # Create a datetime column from Year, Month, and Day columns
        df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
        
        # Define aggregation rules:
        # Sum the irradiance values and average temperature and wind speed
        agg_dict = {
            'GHI': 'sum',
        }
        
        # Group by the new date column and aggregate accordingly
        df_daily = df.groupby('date').agg(agg_dict).reset_index()
        return df_daily
    
    except Exception as e:
        print("Error loading and aggregating CSV file:", e)
        return None

# --- Adds many new features to the dataframe ---
def feature_engineering(dataframe):
    """
    Adds new features to the DataFrame based on the top predictors from the correlation matrix,
    and drops features that were found to be useless.

    Parameters:
      dataframe (DataFrame): Input dataset containing weather data. It is assumed to include columns 
                             such as 'date', 'tmax', 'tmin', 'tavg', 'prcp', 'wspd', 'wdir', 'pres',
                             'cloud_cover', 'sunshine_duration', and 'daylight_duration'.
    
    Returns:
      DataFrame: The augmented DataFrame with newly engineered features.
    """
    import pandas as pd
    import numpy as np

    df = dataframe.copy()
    
    # Ensure the date column is in datetime format
    df['date'] = pd.to_datetime(df['date'])
    
    # --- Top Features & New Interactions ---
    # Effective sunshine: Adjust sunshine by the cloud cover
    df['effective_sunshine'] = df['sunshine_duration'] * (1 - df['cloud_cover'])
    
    # Sunshine ratio: Actual sunshine relative to the total available daylight duration
    df['sunshine_ratio'] = df['sunshine_duration'] / (df['daylight_duration'] + 1e-5)
    
    # Temperature Range (already a top derived feature)
    df['temp_range'] = df['tmax'] - df['tmin']
    
    # Cloud-adjusted temperature range (interaction: temp range reduced by cloud cover)
    df['cloud_adjusted_temp_range'] = df['temp_range'] * (1 - df['cloud_cover'])
    
    # --- Cyclical & Date-based Features ---
    # Day of year and its sine/cosine transformation capture seasonal cycles
    df['day_of_year'] = df['date'].dt.dayofyear
    df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # --- Additional Useful Features ---
    # Difference and exponential moving average for tavg as found useful
    df['tavg_diff'] = df['tavg'].diff().fillna(0)
    df['tavg_ewm'] = df['tavg'].ewm(span=7, adjust=False).mean()
    
    # (Optional) Include other interactions that might be useful
    # e.g., interaction between wind direction and wind speed is sometimes informative.
    df['wdir_wspd'] = df['wdir'] * df['wspd']
    
    # --- Features Deemed Less Useful (to be dropped) ---
    # The following features have extremely low importance based on your correlation matrix:
    df['season'] = (df['date'].dt.month % 12 + 3) // 3
    df['day_of_week'] = df['date'].dt.weekday
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Drop the less useful features
    df.drop(['season', 'day_of_week', 'is_weekend'], axis=1, inplace=True)
    
    return df

# --- Merges the GHI and Historical Weather Data ---
def merge_datasets(nrel_df, meteostat_df):
    """
    Merges the daily NREL solar data with Meteostat historical weather data on the date.
    
    Parameters:
      nrel_df (DataFrame): NREL daily aggregated solar data (with 'date' column).
      meteostat_df (DataFrame): Meteostat daily weather data (with 'time' column).
    
    Returns:
      merged_df (DataFrame): Combined DataFrame with data from both sources.
    """
    # Rename Meteostat 'time' column to 'date' for merging
    print(meteostat_df)
    meteostat_df = meteostat_df.copy()
    meteostat_df.rename(columns={'time': 'date'}, inplace=True)
    
    # Ensure the date columns are in datetime format
    nrel_df['date'] = pd.to_datetime(nrel_df['date'])
    meteostat_df['date'] = pd.to_datetime(meteostat_df['date'])
    
    # Merge on the 'date' column (using inner join to keep matching dates)
    merged_df = pd.merge(nrel_df, meteostat_df, on='date', how='inner')
    
    
    return merged_df

# --- Remove outliers from the dataframe for more consistent data ---
def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    # Select only numerical columns (excluding 'date' if present)
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Create a copy of the dataset to avoid modifying the original
    df_clean = df.copy()

    # Iterate over each numerical column to remove outliers
    for col in numerical_cols:
        # Calculate the IQR
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define the acceptable range
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out the outliers
        df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]

    return df_clean

# --- Different function to handle outliers in the dataframe ---
def handle_outliers(df, z_score_threshold=3, iqr_threshold=1.5, replace=False):
    """
    Function to detect and handle outliers in the dataset using Z-Score or IQR.
    Outliers are either clipped or replaced with median values.

    Parameters:
    - df (DataFrame): The input DataFrame to process.
    - z_score_threshold (float): Threshold for Z-score to detect outliers.
    - iqr_threshold (float): Threshold for IQR to detect outliers.
    - replace (bool): Whether to replace outliers with median (True) or clip them (False).

    Returns:
    - DataFrame: The cleaned DataFrame.
    """
    
    # Z-Score Outlier Detection
    def z_score_outlier_detection(df, threshold):
        z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
        return (z_scores > threshold)
    
    # IQR Outlier Detection
    def iqr_outlier_detection(df, threshold):
        Q1 = df.quantile(0.25)
        Q3 = df.quantile(0.75)
        IQR = Q3 - Q1
        return ((df < (Q1 - threshold * IQR)) | (df > (Q3 + threshold * IQR)))

    # Detect outliers
    z_outliers = z_score_outlier_detection(df, z_score_threshold)
    iqr_outliers = iqr_outlier_detection(df, iqr_threshold)

    # Combine both outlier conditions
    outliers = z_outliers | iqr_outliers
    
    if replace:
        # Replace outliers with the median (you can change to mean if preferred)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = np.where(outliers[col], df[col].median(), df[col])
    else:
        # Clip outliers to the nearest valid value within the range (based on IQR)
        for col in df.select_dtypes(include=[np.number]).columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_limit = Q1 - iqr_threshold * IQR
            upper_limit = Q3 + iqr_threshold * IQR
            df[col] = np.clip(df[col], lower_limit, upper_limit)
    
    return df

# --- scale Dataframe features using Standard Scaler ---
def scale_features(df):
    """
    Function to scale numerical features using StandardScaler.
    
    Parameters:
    - df (DataFrame): The input DataFrame to process.
    
    Returns:
    - DataFrame: The scaled DataFrame.
    """
    scaler = StandardScaler()
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df
