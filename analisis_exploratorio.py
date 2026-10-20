from pathlib import Path
import pandas as pd
import numpy as np

def configurar_entorno(semilla=42):
    np.random.seed(semilla)
    ruta_raiz = Path(__file__).resolve().parent
    # Crear carpetas si no existen
    (ruta_raiz / "resultados" / "tablas").mkdir(parents=True, exist_ok=True)
    (ruta_raiz / "resultados" / "figuras").mkdir(parents=True, exist_ok=True)
    return ruta_raiz

def cargar_y_validar_datos(ruta_archivo, umbral_filas=50000):
    # Carga con UTF-8
    df = pd.read_csv(ruta_archivo, encoding="utf-8")
    
    filas, columnas = df.shape
    print(f"Filas leídas: {filas} | Columnas: {columnas}")
    
    # Chequeo automático de mínimo de registros
    if filas < umbral_filas:
        raise ValueError(f"Error: El dataset tiene {filas} filas, se exigen {umbral_filas}.")
        
    print("Dataset validado con éxito.")
    return df