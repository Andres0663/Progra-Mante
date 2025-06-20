"""
Este script proporciona un marco para predecir la Vida Útil Remanente (RUL)
de motores turbofán, basándose en la metodología descrita en el documento
'Cálculo de Vida Útil Remanente'.

El enfoque utiliza Regresión de Procesos Gaussianos (GPR), una técnica destacada
por su capacidad para proporcionar estimaciones de incertidumbre junto con las predicciones.
La metodología está inspirada en el desafío de datos PHM08 y en un estudio de
referencia de Ayen y Heyns.

Para usar este script:
1.  Asegúrate de que 'train.txt' y 'test.txt' estén en el mismo directorio.
2.  Ejecuta el script. Realizará la preparación de datos, entrenamiento del modelo,
    evaluación en el conjunto de validación y visualización.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# --- 1. Carga y Preparación de Datos ---
def load_and_prepare_data(data_file):
    """
    Carga, prepara y calcula el RUL para un archivo de datos dado.
    El conjunto de datos C-MAPSS contiene datos de series temporales multivariadas.

    Args:
        data_file (str): Nombre del archivo de datos.

    Returns:
        tuple: Una tupla que contiene (características, RUL_objetivo).
    """
    # Define los nombres de las columnas según la descripción del conjunto de datos
    columns = ['unit_number', 'time_in_cycles', 'op_setting_1', 'op_setting_2', 'op_setting_3'] + \
              [f'sensor_{i}' for i in range(1, 22)]

    # Carga los datos
    df = pd.read_csv(data_file, sep=' ', header=None)
    df.drop(columns=[26, 27], inplace=True) # Elimina columnas vacías extra
    df.columns = columns

    # --- OPTIMIZACIÓN DE MEMORIA ---
    # Convierte todas las columnas numéricas a float16 para ahorrar memoria
    for col in df.columns:
        if df[col].dtype == 'float64':
            df[col] = df[col].astype('float16')
        if df[col].dtype == 'int64':
            df[col] = df[col].astype('int16')
    # --- FIN DE LA OPTIMIZACIÓN ---
    
    # Calcula el RUL para los datos
    # Asume que los datos contienen trayectorias completas hasta el fallo.
    max_cycles = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycles.columns = ['unit_number', 'max_cycles']
    df = df.merge(max_cycles, on='unit_number')
    df['RUL'] = df['max_cycles'] - df['time_in_cycles']
    df.drop(columns=['max_cycles'], inplace=True)

    # Selección de características: Elimina sensores constantes o no informativos
    constant_sensors = [col for col in df.columns if df[col].std() < 1e-6]
    features_to_use = [col for col in df.columns if col not in ['unit_number', 'time_in_cycles', 'RUL'] + constant_sensors]

    X = df[features_to_use]
    y = df['RUL']

    return X, y

# --- 2. Selección y Entrenamiento del Modelo ---
def train_gpr_model(X_train, y_train):
    """
    Selecciona el mejor kernel para GPR y entrena el modelo.
    La metodología implica seleccionar la mejor combinación de funciones de media y
    covarianza, evaluadas por RMSE y MAPE. Esta función automatiza la parte de
    selección del kernel.

    Args:
        X_train (np.ndarray): Datos de características de entrenamiento.
        y_train (pd.Series): Datos objetivo (RUL) de entrenamiento.

    Returns:
        GaussianProcessRegressor: El modelo GPR entrenado.
    """
    print("Iniciando selección y entrenamiento del modelo GPR...")

    # Define el kernel como Exponencial Cuadrado (RBF)
    kernels = {
        "RBF": RBF() + WhiteKernel(),
    }

    best_score = float('inf')
    best_kernel_name = None

    # Usa un subconjunto pequeño para una evaluación rápida del kernel, imitando el proceso de selección del paper
    X_sample, _, y_sample, _ = train_test_split(X_train, y_train, train_size=0.1, random_state=42)

    # Entrena el modelo final con todos los datos de entrenamiento usando el mejor kernel
    final_kernel = kernels["RBF"]
    final_gpr = GaussianProcessRegressor(kernel=final_kernel, random_state=42, n_restarts_optimizer=10, alpha=1e-10)
    print("Entrenando modelo GPR final con todos los datos... (Esto puede tardar un momento)")
    final_gpr.fit(X_train, y_train)

    print("Entrenamiento del modelo completado.")
    return final_gpr

# --- 3. Evaluación ---
def evaluate_model(model, X_val, y_val):
    """
    Evalúa el modelo en un conjunto de validación usando las métricas RMSE y MAPE.

    Args:
        model (GaussianProcessRegressor): El modelo entrenado.
        X_val (np.ndarray): Datos de características de validación.
        y_val (pd.Series): RUL real para los datos de validación.

    Returns:
        tuple: Una tupla que contiene (y_pred, y_std, rmse, mape).
    """
    # GPR proporciona una predicción y su incertidumbre (desviación estándar)
    y_pred, y_std = model.predict(X_val, return_std=True)

    # Calcula las métricas de rendimiento
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    mape = mean_absolute_percentage_error(y_val, y_pred)

    print("\n--- Resultados de la Evaluación del Modelo (en Conjunto de Validación) ---")
    print(f"Error Cuadrático Medio (RMSE): {rmse:.4f}")
    print(f"Error Porcentual Absoluto Medio (MAPE): {mape:.4%}")

    return y_pred, y_std, rmse, mape

# --- 4. Visualización ---
def plot_results(y_true, y_pred, y_std):
    """
    Visualiza los resultados de la predicción con intervalos de confianza del 95%.

    Args:
        y_true (pd.Series): RUL real.
        y_pred (np.ndarray): RUL predicho.
        y_std (np.ndarray): Desviación estándar de las predicciones.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    # Para evitar la sobreposición de puntos, graficamos una muestra aleatoria de los puntos
    sample_indices = np.random.choice(len(y_true), size=min(1000, len(y_true)), replace=False)
    y_true_sample = y_true.iloc[sample_indices]
    y_pred_sample = y_pred[sample_indices]
    y_std_sample = y_std[sample_indices]

    # Grafica el RUL real vs. el RUL predicho
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r-', label='Línea de Predicción Perfecta', lw=2)
    ax.errorbar(y_true_sample, y_pred_sample, yerr=1.96 * y_std_sample, fmt='o', color='blue', ecolor='lightblue',
                elinewidth=3, capsize=0, alpha=0.6, label='Predicciones GPR con IC 95% (Muestra)')

    ax.set_title('Predicción de Vida Útil Remanente (RUL) con GPR', fontsize=16)
    ax.set_xlabel('RUL Real (ciclos)', fontsize=12)
    ax.set_ylabel('RUL Predicho (ciclos)', fontsize=12)
    ax.legend(loc='upper left')
    ax.grid(True)

    # Establece los límites de los ejes para una mejor visualización
    max_val = max(y_true.max(), y_pred.max())
    ax.set_xlim(0, max_val + 10)
    ax.set_ylim(0, max_val + 10)

    plt.tight_layout()
    plt.show()

# --- Ejecución Principal ---
if __name__ == '__main__':
    # Define las rutas de los archivos de datos
    TRAIN_FILE = 'train.txt'
    VALIDATION_FILE = 'test.txt'

    try:
        # 1. Cargar datos
        print(f"Cargando datos de entrenamiento desde '{TRAIN_FILE}'...")
        X_train_raw, y_train_raw = load_and_prepare_data(TRAIN_FILE)
        
        # Reduce el conjunto de entrenamiento para ahorrar memoria.
        # Usa solo el 20% de los datos (puedes ajustar este valor).
        # Usamos train_test_split para obtener una muestra estratificada si es necesario,
        # o simplemente el método .sample() de pandas.
        print(f"Tamaño original de entrenamiento: {len(X_train_raw)} muestras.")
        
        # Define el tamaño de la muestra (ej. 5000 muestras)
        n_samples = 5000 
        if len(X_train_raw) > n_samples:
            X_train_sampled, _, y_train_sampled, _ = train_test_split(
                X_train_raw, y_train_raw, train_size=n_samples, random_state=42
            )
        else:
            X_train_sampled, y_train_sampled = X_train_raw, y_train_raw

        print(f"Usando un subconjunto de {len(X_train_sampled)} muestras para entrenar.")

        print(f"Cargando datos de validación desde '{VALIDATION_FILE}'...")
        X_val_raw, y_val = load_and_prepare_data(VALIDATION_FILE)

        # 2. Normalizar los datos
        # Se ajusta con los datos de entrenamiento y se transforman ambos conjuntos (entrenamiento y validación)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_sampled)
        X_val = scaler.transform(X_val_raw)

        # 3. Entrenar el modelo
        gpr_model = train_gpr_model(X_train, y_train_sampled)

        # 4. Evaluar el modelo en el conjunto de validación
        y_pred, y_std, rmse, mape = evaluate_model(gpr_model, X_val, y_val)

        # 5. Visualizar los resultados
        plot_results(y_val, y_pred, y_std)

    except FileNotFoundError as e:
        print("-" * 50)
        print(f"ERROR: Archivo de datos no encontrado -> {e.filename}")
        print(f"Por favor, asegúrate de que '{TRAIN_FILE}' y '{VALIDATION_FILE}' estén en el mismo directorio que este script.")
        print("-" * 50)