import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.interpolate import interp1d
import warnings

def load_data(file_path):
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format.")
    
    data['y'] = data['y'].apply(lambda v: np.nan if pd.isna(v) or v == '' or (isinstance(v, str) and v.lower() == 'none') else v)
    
    return data

def divided_difference(x, y):
    n = len(x)
    coeff = np.copy(y)
    
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            if np.isnan(coeff[i]) or np.isnan(coeff[i-1]) or np.isnan(x[i]) or np.isnan(x[i-j]):
                continue
            coeff[i] = (coeff[i] - coeff[i-1]) / (x[i] - x[i-j])
    
    return coeff

def interpolate_missing_values(x, y):
    known_x = [x[i] for i in range(len(y)) if not np.isnan(y[i])]
    known_y = [y[i] for i in range(len(y)) if not np.isnan(y[i])]

    coeff = divided_difference(known_x, known_y)

    def interpolate(x_val):
        result = coeff[-1]
        for i in range(len(coeff)-2, -1, -1):
            result = result * (x_val - known_x[i]) + coeff[i]
        return result

    for i in range(len(y)):
        if np.isnan(y[i]):
            y[i] = interpolate(x[i])

    return y

def simple_linear_regression(x, y, k):
    known_x = np.array([x[i] for i in range(len(y)) if not np.isnan(y[i])]).reshape(-1, 1)
    known_y = np.array([y[i] for i in range(len(y)) if not np.isnan(y[i])])

    model = LinearRegression()
    model.fit(known_x, known_y)
    forecasted_x_2d = np.linspace(min(x), min(x) + len(x) + k - 1, len(x) + k)
    forecasted_x = forecasted_x_2d.reshape(-1, 1)
    forecasted_y = model.predict(forecasted_x)
    
    return forecasted_x.flatten(), forecasted_y.flatten()

def main():
    while True:
        print("Choose an option:")
        print("1 - Newton's Divided-Difference Interpolation")
        print("2 - Linear Regression Forecasting")
        print("0 - Exit")
        
        choice = int(input("Enter your choice: "))

        if choice == 1:
            file_path = input("Enter the file path (CSV or Excel): ").strip()
            data = load_data(file_path)
            x = data['x'].values
            y = data['y'].values
            
            
            warnings.filterwarnings("ignore", message="linestyle is redundantly defined")

            #interpolated data
            y_interpolated = interpolate_missing_values(x, y.copy())
            spline = interp1d(x, y_interpolated, kind='cubic', fill_value="extrapolate")
            dense_x = np.linspace(min(x), max(x), 500)
            dense_y = spline(dense_x)
            plt.plot(dense_x, dense_y, 'g-', label="Interpolated Data", linewidth=2)  # Green curve
            plt.plot(x, y_interpolated, 'go', label="Interpolated Data Points", markersize=6) #dots
            #original data
            x_known = x[~np.isnan(y)]
            y_known = y[~np.isnan(y)]
            spline_original = interp1d(x_known, y_known, kind='cubic', fill_value="extrapolate")
            dense_x = np.linspace(min(x_known), max(x_known), 500)
            dense_y = spline_original(dense_x)
            plt.plot(dense_x, dense_y, 'b-', label="SOriginal Data", linewidth=2)  # Blue curve
            plt.plot(x_known, y_known, 'bo', label="Original Data Points", markersize=6) # dots
            
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.title("Newton's Divided-Difference Interpolation")
            plt.grid(True)
            plt.show()

            print("\nOriginal Data and Interpolated Data:")
            print(f"{'x':<10}{'Original y':<20}{'Interpolated y'}")
            for xi, yi, yi_interpolated in zip(x, y, y_interpolated):
                print(f"{xi:<10}{yi:<20}{yi_interpolated}")

        elif choice == 2:
            file_path = input("Enter the file path (CSV or Excel): ").strip()
            data = load_data(file_path)
            x = data['x'].values
            y = data['y'].values
            k = int(input("Enter number of future points to forecast: "))
            warnings.filterwarnings("ignore", message="linestyle is redundantly defined")

            forecasted_x, forecasted_y = simple_linear_regression(x, y, k)
            spline_forecast = interp1d(forecasted_x, forecasted_y, kind='cubic', fill_value="extrapolate")
            dense_forecasted_x = np.linspace(min(x), max(forecasted_x), 500)
            dense_forecasted_y = spline_forecast(dense_forecasted_x)
            plt.plot(dense_forecasted_x, dense_forecasted_y, 'r-', label="Forecasted Data", linewidth=2) #red line
            plt.plot(forecasted_x, forecasted_y, 'ro', label="Forecasted Data Points", markersize=6) # dots

            #original data
            x_known = x[~np.isnan(y)]
            y_known = y[~np.isnan(y)]
            spline_original = interp1d(x_known, y_known, kind='cubic', fill_value="extrapolate")
            dense_x = np.linspace(min(x_known), max(x_known), 500)
            dense_y = spline_original(dense_x)
            plt.plot(dense_x, dense_y, 'b-', label="Original Data", linewidth=2)  # Blue curve
            plt.plot(x_known, y_known, 'bo', label="Original Data Points", markersize=6) # dots
            
            plt.xlabel("x")
            plt.ylabel("y")
            plt.legend()
            plt.title("Simple Linear Regression Forecast")
            plt.grid(True)
            plt.show()

            print("\nOriginal Data and Forecasted Data:")
            print(f"{'x':<10}{'Original y':<20}{'Forecasted y'}")
            
            for xi, yi, yi_forecasted in zip(x, y, forecasted_y[:len(x)]):
                print(f"{xi:<10}{yi if not np.isnan(yi) else 'None':<20}{yi_forecasted}")
            for xi, yi_forecasted in zip(forecasted_x[len(x):], forecasted_y[len(x):]):
                print(f"{xi:<10}{'':<20}{yi_forecasted}")

        elif choice == 0:
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 0.")
        
        repeat = input("Do you want to solve another problem? (y/n): ").lower()
        if repeat != 'y':
            print("Exiting...")
            break


if __name__ == "__main__":
    main()