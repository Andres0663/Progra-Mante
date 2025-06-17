# -*- coding: utf-8 -*-
"""
This script provides a framework for predicting the Remaining Useful Life (RUL)
of turbofan engines based on the methodology described in the document
'Calculo de Vida Util Remanente'.

The approach uses Gaussian Process Regression (GPR), a technique highlighted
for its ability to provide uncertainty estimates alongside predictions.
The methodology is inspired by the PHM08 data challenge and a reference
study by Ayen and Heyns.

To use this script:
1.  Download the C-MAPSS dataset (e.g., 'train_FD001.txt', 'test_FD001.txt', 'RUL_FD001.txt').
2.  Place the data files in the same directory as this script.
3.  Run the script. It will perform data preparation, model training,
    evaluation, and visualization.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# --- 1. Data Loading and Preparation ---
def load_and_prepare_data(train_file, test_file, rul_file):
    """
    Loads, prepares, and computes RUL for the training and testing datasets.
    The C-MAPSS dataset contains multivariate time series data from 21 sensors.

    Args:
        train_file (str): Filename for the training data.
        test_file (str): Filename for the test data.
        rul_file (str): Filename for the ground truth RUL values.

    Returns:
        tuple: A tuple containing (X_train, y_train, X_test, y_test).
    """
    # Define column names based on the dataset's description
    columns = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]
              
    # Load training data
    train_df = pd.read_csv(train_file, sep=' ', header=None)
    train_df.drop(columns=[26, 27], inplace=True) # Drop extra empty columns
    train_df.columns = columns
    
    # Calculate RUL for training data
    # The training data contains full run-to-failure trajectories.
    max_cycles = train_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycles']
    train_df = train_df.merge(max_cycles, on='unit_number')
    train_df['RUL'] = train_df['max_cycles'] - train_df['time_in_cycles']
    train_df.drop(columns=['max_cycles'], inplace=True)

    # Load test data and ground truth RUL
    test_df = pd.read_csv(test_file, sep=' ', header=None)
    test_df.drop(columns=[26, 27], inplace=True)
    test_df.columns = columns
    y_test_truth = pd.read_csv(rul_file, sep=' ', header=None)
    y_test_truth.drop(columns=[1], inplace=True)
    
    # Get the last cycle for each engine in the test set to predict RUL 
    X_test = test_df.groupby('unit_number').last().reset_index()
    y_test = pd.DataFrame(y_test_truth.values, columns=['RUL'])

    # Feature Selection: Drop constant or non-informative sensors
    # This is a common practice for the C-MAPSS dataset.
    constant_sensors = [col for col in train_df.columns if train_df[col].std() < 1e-6]
    features = [col for col in train_df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL'] + constant_sensors]
    
    X_train = train_df[features]
    y_train = train_df['RUL']
    X_test = X_test[features]

    # Normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, y_train, X_test, y_test['RUL']

# --- 2. Model Selection and Training ---
def train_gpr_model(X_train, y_train):
    """
    Selects the best GPR kernel and trains the model.
    The methodology involves selecting the best combination of mean and covariance
    functions, evaluated by RMSE and MAPE. This function automates
    the kernel selection part.

    Args:
        X_train (np.ndarray): Training feature data.
        y_train (pd.Series): Training target data (RUL).

    Returns:
        GaussianProcessRegressor: The trained GPR model.
    """
    print("Starting GPR model training and selection...")
    
    # Define candidate kernels based on common choices like Squared Exponential and Matérn 
    kernels = {
        "Matern": Matern(nu=1.5) + WhiteKernel(),
        "RBF": RBF() + WhiteKernel(),
    }
    
    best_score = float('inf')
    best_kernel_name = None
    
    # Use a small subset for rapid kernel evaluation to mimic the paper's selection process
    X_sample, _, y_sample, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=42)

    for name, kernel in kernels.items():
        gpr = GaussianProcessRegressor(kernel=kernel, random_state=42, n_restarts_optimizer=5, alpha=1e-2)
        print(f"Testing kernel: {name}...")
        gpr.fit(X_sample, y_sample)
        y_pred = gpr.predict(X_sample)
        rmse = np.sqrt(mean_squared_error(y_sample, y_pred))
        
        print(f"  - RMSE for {name}: {rmse:.4f}")
        if rmse < best_score:
            best_score = rmse
            best_kernel_name = name

    print(f"\nSelected best kernel: {best_kernel_name} with RMSE: {best_score:.4f}")
    
    # Train the final model on the full training data with the best kernel
    final_kernel = kernels[best_kernel_name]
    final_gpr = GaussianProcessRegressor(kernel=final_kernel, random_state=42, n_restarts_optimizer=10, alpha=1e-2)
    print("Training final GPR model on all data... (This may take a moment)")
    final_gpr.fit(X_train, y_train)
    
    print("Model training complete.")
    return final_gpr

# --- 3. Evaluation ---
def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model using RMSE and MAPE metrics.

    Args:
        model (GaussianProcessRegressor): The trained model.
        X_test (np.ndarray): Test feature data.
        y_test (pd.Series): Ground truth RUL for the test data.

    Returns:
        tuple: A tuple containing (y_pred, y_std, rmse, mape).
    """
    # GPR provides a prediction and its uncertainty (standard deviation) 
    y_pred, y_std = model.predict(X_test, return_std=True)
    
    # Calculate performance metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mape = mean_absolute_percentage_error(y_test, y_pred)
    
    print("\n--- Model Evaluation Results ---")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.4%}")
    
    return y_pred, y_std, rmse, mape

# --- 4. Visualization ---
def plot_results(y_test, y_pred, y_std):
    """
    Visualizes the prediction results with 95% confidence intervals,
    similar to the reference paper's analysis.

    Args:
        y_test (pd.Series): Ground truth RUL.
        y_pred (np.ndarray): Predicted RUL.
        y_std (np.ndarray): Standard deviation of predictions.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Plot true RUL vs. predicted RUL
    ax.plot(y_test.values, y_test.values, 'r-', label='Perfect Prediction Line', lw=2)
    ax.errorbar(y_test, y_pred, yerr=1.96 * y_std, fmt='o', color='blue', ecolor='lightblue',
                elinewidth=3, capsize=0, label='GPR Predictions with 95% CI')

    ax.set_title('GPR Prediction of Remaining Useful Life (RUL)', fontsize=16)
    ax.set_xlabel('Actual RUL (cycles)', fontsize=12)
    ax.set_ylabel('Predicted RUL (cycles)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True)
    
    # Set axis limits for better visualization
    max_val = max(y_test.max(), y_pred.max())
    ax.set_xlim(0, max_val + 10)
    ax.set_ylim(0, max_val + 10)
    
    plt.tight_layout()
    plt.show()

# --- Main Execution ---
if __name__ == '__main__':
    # Define file paths for the first dataset FD001
    TRAIN_FILE = 'train_FD001.txt'
    TEST_FILE = 'test_FD001.txt'
    RUL_FILE = 'RUL_FD001.txt'

    try:
        # 1. Load data
        X_train, y_train, X_test, y_test = load_and_prepare_data(TRAIN_FILE, TEST_FILE, RUL_FILE)
        
        # 2. Train the model
        gpr_model = train_gpr_model(X_train, y_train)
        
        # 3. Evaluate the model
        y_pred, y_std, rmse, mape = evaluate_model(gpr_model, X_test, y_test)
        
        # 4. Visualize the results
        plot_results(y_test, y_pred, y_std)

    except FileNotFoundError:
        print("-" * 50)
        print("ERROR: Data files not found.")
        print(f"Please download the C-MAPSS dataset and place '{TRAIN_FILE}',")
        print(f"'{TEST_FILE}', and '{RUL_FILE}' in the same directory as this script.")
        print("-" * 50)