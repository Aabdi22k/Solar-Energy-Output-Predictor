#!/usr/bin/env python3
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from clean_process_data import scale_features
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor

def calculate_accuracy_within_mae(y_true, y_pred, mae):
    """
    Compares predicted and actual values.
    Counts a prediction as "correct" if the actual value is within the predicted value ± MAE.
    
    Parameters:
        y_true (array-like): Actual target values.
        y_pred (array-like): Predicted target values.
        mae (float): Mean Absolute Error (threshold).
        
    Returns:
        accuracy_percentage (float): Percentage of correct predictions.
        correct (int): Number of correct predictions.
        incorrect (int): Number of incorrect predictions.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    # A prediction is considered "correct" if the actual value falls within the predicted ± MAE.
    correct = np.sum((y_true >= (y_pred - mae)) & (y_true <= (y_pred + mae)))
    total = len(y_true)
    incorrect = total - correct
    accuracy_percentage = (correct / total) * 100
    return accuracy_percentage, correct, incorrect

def calculate_accuracy_within_std_devs(y_true, y_pred, mae, std_error):
    """
    Calculates accuracy within multiple standard deviation ranges: ±MAE, ±1, ±2, ±3 Std Deviations.
    
    Parameters:
        y_true (array-like): Actual target values.
        y_pred (array-like): Predicted target values.
        mae (float): Mean Absolute Error (threshold).
        std_error (float): Standard deviation of the errors.
        
    Returns:
        accuracy_percentage (dict): Dictionary of accuracy percentages within each range.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Create ranges for MAE, ±1, ±2, and ±3 standard deviations
    accuracy = {}
    accuracy["MAE"] = np.sum((y_true >= (y_pred - mae)) & (y_true <= (y_pred + mae))) / len(y_true) * 100
    accuracy["1std"] = np.sum((y_true >= (y_pred - std_error)) & (y_true <= (y_pred + std_error))) / len(y_true) * 100
    accuracy["2std"] = np.sum((y_true >= (y_pred - 2*std_error)) & (y_true <= (y_pred + 2*std_error))) / len(y_true) * 100
    accuracy["3std"] = np.sum((y_true >= (y_pred - 3*std_error)) & (y_true <= (y_pred + 3*std_error))) / len(y_true) * 100
    
    return accuracy

def calculate_continuous_accuracy(y_true, y_pred):
    """
    Calculates a continuous accuracy metric based on the distribution of absolute errors.
    
    For each prediction:
      - If the absolute error is <= MAE, it scores 1.
      - If the absolute error is between MAE and MAE + std_error,
        it gets a linearly decreasing score from 1 down to 0.
      - If the error is above (MAE + std_error), it scores 0.
    
    The overall accuracy is the mean of these scores (converted to a percentage).
    
    Parameters:
        y_true (array-like): Actual target values.
        y_pred (array-like): Predicted target values.
        
    Returns:
        continuous_accuracy (float): The overall continuous accuracy (in percentage).
        mae (float): The mean absolute error of the predictions.
        std_error (float): The standard deviation of the absolute errors.
    """
    errors = np.abs(np.array(y_true) - np.array(y_pred))
    mae = np.mean(errors)
    std_error = np.std(errors)
    
    scores = np.where(
        errors <= mae,
        1.0,
        np.where(errors < (mae + std_error), 1 - (errors - mae) / std_error, 0.0)
    )
    continuous_accuracy = np.mean(scores) * 100
    
    return continuous_accuracy, mae, std_error

def train_and_evaluate_model(df):
    """
    Trains a Random Forest regressor to predict GHI (solar irradiance) using
    the engineered weather features and evaluates its performance.
    
    Uses the following features for training (all columns except 'date' and 'GHI'):
      - tavg, tmin, tmax, prcp, snow, wdir, wspd, pres, day_of_year, day_of_week
    
    Prints model evaluation metrics, including a custom accuracy metric that checks
    if actual values are within the predicted value ± MAE, as well as a continuous 
    accuracy metric that provides partial credit for near-miss predictions.
    Also prints feature importances and plots diagnostic graphs.
    """
    # Define target (GHI) and features (all columns except 'date' and 'GHI')
    # Split data into features and target
    X = df.drop(columns=['date', 'GHI'])
    y = df['GHI']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize the scaler
    scaler = StandardScaler()

    # Fit the scaler on the training data and transform it
    X_train_scaled = scaler.fit_transform(X_train)

    # Transform the test data using the same scaler (do not fit again)
    X_test_scaled = scaler.transform(X_test)

    # Define and train the model
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predict on test set
    y_pred = model.predict(X_test_scaled)
    # Evaluate model performance using common metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    scores = cross_val_score(model, X, y, cv=5, scoring='r2')

    print("Model Performance:")
    print(f"  Mean Squared Error (MSE): {mse:.2f}")
    print(f"  Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"  Mean Absolute Error (MAE): {mae:.2f}")
    print(f"  R² Score: {r2:.2f}")
    print("  Cross-Validation R² Scores:", scores)
    
    # Calculate and display the custom binary accuracy metric based on MAE tolerance
    accuracy, correct, incorrect = calculate_accuracy_within_mae(y_test, y_pred, mae)
    print(f"\nCustom Accuracy (within ±MAE): {accuracy:.2f}% (Correct: {correct}, Incorrect: {incorrect})")
    
    # Calculate and display the continuous accuracy metric
    cont_accuracy, cont_mae, cont_std = calculate_continuous_accuracy(y_test, y_pred)
    print(f"Continuous Accuracy: {cont_accuracy:.2f}% (MAE: {cont_mae:.2f}, Std Error: {cont_std:.2f})")
    
    # Calculate and display accuracy within different standard deviation ranges
    accuracy_std_devs = calculate_accuracy_within_std_devs(y_test, y_pred, cont_mae, cont_std)
    print(f"\nAccuracy within ±MAE: {accuracy_std_devs['MAE']:.2f}%")
    print(f"Accuracy within ±1std: {accuracy_std_devs['1std']:.2f}%")
    print(f"Accuracy within ±2std: {accuracy_std_devs['2std']:.2f}%")
    print(f"Accuracy within ±3std: {accuracy_std_devs['3std']:.2f}%")
    
    # Calculate and display feature importances
    feature_importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    print("\nFeature Importances:")
    print(importance_df)
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    sns.barplot(data=importance_df, x='Importance', y='Feature')
    plt.title("Feature Importances")
    plt.tight_layout()
    plt.show()

    # Plot histogram of residuals
    sns.histplot(y_test - y_pred, kde=True)
    plt.xlabel("Residuals")
    plt.show()

    # Calculate and plot residuals vs predicted values
    residuals = y_test - y_pred
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted GHI')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted GHI')
    plt.show()

    mean = 0
    std = cont_std
    mae = cont_mae

    # Define standard deviation ranges
    x = np.linspace(mean - 4*std, mean + 4*std, 1000)
    y = np.exp(-((x - mean) ** 2) / (2 * std ** 2))  # Gaussian curve

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Normal Distribution", color='black')

    # Fill areas for MAE and standard deviations
    plt.fill_between(x, y, where=(x >= mean - mae) & (x <= mean + mae), color='green', alpha=0.3, label=f'±MAE ({accuracy_std_devs['MAE']:.2f}%)')
    plt.fill_between(x, y, where=(x >= mean - std) & (x <= mean + std), color='red', alpha=0.2, label=f'±1 Std Dev ({accuracy_std_devs['1std']:.2f}%)')
    plt.fill_between(x, y, where=(x >= mean - 2*std) & (x <= mean + 2*std), color='orange', alpha=0.2, label=f'±2 Std Dev ({accuracy_std_devs['2std']:.2f}%)')
    plt.fill_between(x, y, where=(x >= mean - 3*std) & (x <= mean + 3*std), color='purple', alpha=0.2, label=f'±3 Std Dev ({accuracy_std_devs['3std']:.2f}%)')

    # Add vertical lines
    plt.axvline(mean - mae, color='green', linestyle='dashed', linewidth=1)
    plt.axvline(mean + mae, color='green', linestyle='dashed', linewidth=1)
    plt.axvline(mean - std, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(mean + std, color='red', linestyle='dashed', linewidth=1)
    plt.axvline(mean - 2*std, color='orange', linestyle='dashed', linewidth=1)
    plt.axvline(mean + 2*std, color='orange', linestyle='dashed', linewidth=1)
    plt.axvline(mean - 3*std, color='purple', linestyle='dashed', linewidth=1)
    plt.axvline(mean + 3*std, color='purple', linestyle='dashed', linewidth=1)

    # Labels and legend
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title("Distribution with ±MAE and Standard Deviations")
    plt.legend()
    plt.show()

    joblib.dump(model, 'solar_model.pkl')
    joblib.dump(scaler, 'solar_scaler.pkl')
    return model, scaler, cont_mae, cont_std, accuracy_std_devs
