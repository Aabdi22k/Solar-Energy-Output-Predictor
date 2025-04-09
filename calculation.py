def calculate_solar_output_MAE(solar_array_size, solar_array_efficiency, ghi_values, mae):
    """
    Calculates the solar panel power output for each GHI value after adjusting for MAE.
    
    For each GHI value, the function computes two outputs:
      - One with MAE subtracted from the GHI.
      - One with MAE added to the GHI.
    
    Parameters:
    solar_array_size (float): The size of the solar array.
    solar_array_efficiency (float): The efficiency of the solar array.
    ghi_values (list of float): Array of Global Horizontal Irradiance (GHI) values.
    mae (float): The Mean Absolute Error to adjust the GHI values.
    
    Returns:
    list of tuple: Each tuple contains:
      (solar output with (GHI - MAE), solar output with (GHI + MAE))
    """
    results = []
    for ghi in ghi_values:
        output_minus = solar_array_size * solar_array_efficiency * (ghi - mae) / 1000
        output_plus = solar_array_size * solar_array_efficiency * (ghi + mae) / 1000
        results.append((output_minus, output_plus))
    return results

def calculate_solar_output_1std(solar_array_size, solar_array_efficiency, ghi_values, std):
    """
    Calculates the solar panel power output for each GHI value after adjusting for 1std.
    
    For each GHI value, the function computes two outputs:
      - One with 1std subtracted from the GHI.
      - One with 1std added to the GHI.
    
    Parameters:
    solar_array_size (float): The size of the solar array.
    solar_array_efficiency (float): The efficiency of the solar array.
    ghi_values (list of float): Array of Global Horizontal Irradiance (GHI) values.
    std (float): The Standard Deviation to adjust the GHI values.
    
    Returns:
    list of tuple: Each tuple contains:
      (solar output with (GHI - 1std), solar output with (GHI + 1std))
    """
    results = []
    for ghi in ghi_values:
        output_minus = solar_array_size * solar_array_efficiency * (ghi - std) / 1000
        output_plus = solar_array_size * solar_array_efficiency * (ghi + std) / 1000
        results.append((output_minus, output_plus))
    return results

def calculate_solar_output_2std(solar_array_size, solar_array_efficiency, ghi_values, std):
    """
    Calculates the solar panel power output for each GHI value after adjusting for 2std.
    
    For each GHI value, the function computes two outputs:
      - One with 2std subtracted from the GHI.
      - One with 2std added to the GHI.
    
    Parameters:
    solar_array_size (float): The size of the solar array.
    solar_array_efficiency (float): The efficiency of the solar array.
    ghi_values (list of float): Array of Global Horizontal Irradiance (GHI) values.
    std (float): The Standard Deviation to adjust the GHI values.
    
    Returns:
    list of tuple: Each tuple contains:
      (solar output with (GHI - 2std), solar output with (GHI + 2std))
    """
    results = []
    for ghi in ghi_values:
        output_minus = solar_array_size * solar_array_efficiency * (ghi - (std * 2)) / 1000
        output_plus = solar_array_size * solar_array_efficiency * (ghi + (std * 2)) / 1000
        results.append((output_minus, output_plus))
    return results

def calculate_solar_output_3std(solar_array_size, solar_array_efficiency, ghi_values, std):
    """
    Calculates the solar panel power output for each GHI value after adjusting for 3std.
    
    For each GHI value, the function computes two outputs:
      - One with 3std subtracted from the GHI.
      - One with 3std added to the GHI.
    
    Parameters:
    solar_array_size (float): The size of the solar array.
    solar_array_efficiency (float): The efficiency of the solar array.
    ghi_values (list of float): Array of Global Horizontal Irradiance (GHI) values.
    std (float): The Standard Deviation to adjust the GHI values.
    
    Returns:
    list of tuple: Each tuple contains:
      (solar output with (GHI - 3std), solar output with (GHI + 3std))
    """
    results = []
    for ghi in ghi_values:
        output_minus = solar_array_size * solar_array_efficiency * (ghi - (std * 3)) / 1000
        output_plus = solar_array_size * solar_array_efficiency * (ghi + (std * 3)) / 1000
        results.append((output_minus, output_plus))
    return results
