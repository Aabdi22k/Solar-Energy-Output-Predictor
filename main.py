from get_data import get_data
from train_model import train_and_evaluate_model
from predict import predict
from calculation import calculate_solar_output_MAE, calculate_solar_output_1std, calculate_solar_output_2std, calculate_solar_output_3std

# Get Historical GHI & Weather Data along with Forecast
# Returns Merged, Feature Engineered, and Cleaned Dataframe
# Return Forecast Dataframe
# Phoeinx, Ariozna for testing lat: 33.448376 lon: -112.074036
merged_fe_df, forecast_df = get_data(33.448376, -112.074036)

# Train & Evaluate RFR Model
# Return the trained model
# Return Mean Absolute Error & 1st Standard Deviation
# Return Accuracy Within MAE, 1st, 2nd, & 3rd Standard Deviations
model, scaler, mae, std, accuracies = train_and_evaluate_model(merged_fe_df)

# Predict GHI for a certain number of days (default = 5)
# Returns value of GHI for next 5 days
ghi_values = predict(model, scaler, forecast_df)

# Calculate Solar Output
solar_panel_power_output_range_mae = calculate_solar_output_MAE(10, 0.15, ghi_values, mae)
solar_panel_power_output_range_1std = calculate_solar_output_1std(10, 0.15, ghi_values, std)
solar_panel_power_output_range_2std = calculate_solar_output_2std(10, 0.15, ghi_values, std)
solar_panel_power_output_range_3std = calculate_solar_output_3std(10, 0.15, ghi_values, std)

print(f"Predicted GHI values: {ghi_values}")
print(f'If solar array size = 100m^2 and panel efficiency = 15%')
print(f"Predicted Solar Output values within MAE - {accuracies['±MAE']}% acc")
print(solar_panel_power_output_range_mae)
print(f"Predicted Solar Output values within 1std - {accuracies['±1std']}% acc")
print(solar_panel_power_output_range_1std)
print(f"Predicted Solar Output values within 2std - {accuracies['±2std']}% acc")
print(solar_panel_power_output_range_2std)
print(f"Predicted Solar Output values within 3std - {accuracies['±3std']}% acc")
print(solar_panel_power_output_range_3std)



