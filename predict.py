import pandas as pd
from clean_process_data import scale_features

def predict(model, scaler, forecast_df):
    X = forecast_df.drop(columns=['date'])
    
    # Use transform, not fit_transform
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)   

    return prediction
