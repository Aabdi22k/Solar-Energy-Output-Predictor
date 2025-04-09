import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from clean_process_data import scale_features

def predict_one_by_one(model, X, scaler):
    predictions = []
    
    # Use the scaler fit on the entire batch, not fit per row
    X_scaled = scaler.transform(X)
    
    for _, row in X.iterrows():
        # Reshape each row and predict
        row_scaled = X_scaled[_].reshape(1, -1)
        prediction = model.predict(row_scaled)
        predictions.append(prediction[0])  # Store single prediction
    
    return predictions

def main():
    # Load the model
    model = joblib.load("solar_model.pkl")

    # Load forecast data
    forecast_df = pd.read_csv("forecast_33.448376_-112.074036_2025-03-11.csv")

    # Batch prediction
    X_batch = forecast_df.drop(columns=['date'])
    scaler = joblib.load("solar_scaler.pkl")  
    X_batch_scaled = scaler.transform(X_batch)  # Use transform (not fit_transform) here
    batch_predictions = model.predict(X_batch_scaled)

    # One-by-one prediction
    single_predictions = predict_one_by_one(model, X_batch, scaler)

    # Print results
    print("Batch Predictions:", batch_predictions)
    print("One-by-One Predictions:", single_predictions)

if __name__ == "__main__":
    main()
