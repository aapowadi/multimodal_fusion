"""
We provide a set of time points as a numpy array and a set of values as a numpy array using np.polyfit
The function `curve_fitting` will fit a polynomial curve to the data points
Use the curve fitting to predict the missing values in the time series data.
"""

import numpy as np


def curve_fitting(time_points, values, degree=3):
    """
    Fit a polynomial curve to the given time points and values.

    Parameters:
    - time_points: np.ndarray, 1D array of time points
    - values: np.ndarray, 1D array of values corresponding to the time points
    - degree: int, degree of the polynomial to fit

    Returns:
    - coefficients: np.ndarray, coefficients of the fitted polynomial
    """
    # Fit a polynomial of the specified degree to the data
    coefficients = np.polyfit(time_points, values, degree)

    return coefficients


def predict_missing_values(time_points, coefficients):
    """
    Predict missing values using the fitted polynomial coefficients.
    Parameters:
    - time_points: np.ndarray, 1D array of time points for prediction
    - coefficients: np.ndarray, coefficients of the fitted polynomial
    Returns:
    - predicted_values: np.ndarray, predicted values at the given time points
    """
    # Create a polynomial function from the coefficients
    polynomial = np.poly1d(coefficients)

    # Predict values at the given time points
    predicted_values = polynomial(time_points)

    return predicted_values


def impute_missing_values(time_series_data, time_points, degree=3):
    """
    Impute missing values in a time series using polynomial curve fitting.

    Parameters:
    - time_series_data: np.ndarray, 1D array of time series data with NaN for missing values
    - time_points: np.ndarray, 1D array of time points corresponding to the data
    - degree: int, degree of the polynomial to fit

    Returns:
    - imputed_data: np.ndarray, time series data with missing values imputed
    """
    # Identify indices of missing values
    missing_indices = np.isnan(time_series_data)

    # Get valid data points and their corresponding time points
    valid_time_points = time_points[~missing_indices]
    valid_values = time_series_data[~missing_indices]

    # Fit a polynomial curve to the valid data points
    coefficients = curve_fitting(valid_time_points, valid_values, degree)

    # Predict missing values using the fitted polynomial
    imputed_values = predict_missing_values(
        time_points[missing_indices], coefficients)

    # Create a copy of the original data to fill in missing values
    imputed_data = np.copy(time_series_data)

    # Fill in the imputed values at the missing indices
    imputed_data[missing_indices] = imputed_values

    return imputed_data


if __name__ == "__main__":
    # Example usage
    time_points = np.array([1, 2, 3, 4, 5])
    values = np.array([1.0, np.nan, 3.0, np.nan, 5.0])

    imputed_data = impute_missing_values(values, time_points)
    print("Original data:", values)
    print("Imputed data:", imputed_data)
