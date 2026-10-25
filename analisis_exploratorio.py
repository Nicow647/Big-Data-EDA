
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from diagnostico_calidad import diagnostico_calidad


# Rutas relativas del proyecto
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datos"
TABLAS_DIR = BASE_DIR / "resultados" / "tablas"
FIGURAS_DIR = BASE_DIR / "resultados" / "figuras"

ARCHIVO_CSV = DATA_DIR / "Chile Earthquake Dataset (1976-2021).csv"


def configurar_entorno(semilla=42):
    """Configura la semilla y crea las carpetas de resultados."""
    np.random.seed(semilla)

    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)

    return BASE_DIR


def cargar_y_validar_datos(ruta_archivo, umbral_filas=50000):
    """Carga los datos y realiza las validaciones iniciales."""
    if not ruta_archivo.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo del dataset: {ruta_archivo}"
        )

    df = pd.read_csv(ruta_archivo, encoding="utf-8")

    filas, columnas = df.shape
    print(f"Filas leídas: {filas} | Columnas: {columnas}")

    if filas < umbral_filas:
        raise ValueError(
            f"Error: el dataset tiene {filas} filas; "
            f"se exigen al menos {umbral_filas}."
        )

    # Combinar fecha y hora antes de retirar las columnas originales.
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%Y-%m-%d %H:%M:%S.%f",
        errors="coerce",
    )

    fechas_invalidas = df["datetime"].isna().sum()
    print(f"Fechas y horas que no pudieron convertirse: {fechas_invalidas}")

    df = df.drop(columns=["date", "time"])

    # Se conserva la lógica original del proyecto.
    # El diagnóstico previo permite comprobar si hay identificadores repetidos.
    df = df.drop_duplicates(subset=["event name"], keep="first")

    print(df.head(12))
    df.info()

    print("Dataset validado con éxito.")
    return df


def generar_graficos(df):
    """Genera los gráficos exploratorios principales."""
    print("Generando gráficos...")

    # Gráfico 1: distribución de magnitudes
    print("Gráfico 1: distribución de magnitudes")

    plt.figure(figsize=(10, 6))
    df["magnitude"].hist()
    plt.title("Distribución de Magnitudes Sísmicas (1976-2021)")
    plt.xlabel("Magnitud (Mw)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.show()

    # Gráfico 2: profundidad y momento sísmico
    print("Gráfico 2: profundidad vs. momento sísmico")

    plt.figure(figsize=(10, 6))
    plt.scatter(df["depth"], df["moment"], alpha=0.6)
    plt.title("Relación entre profundidad y momento sísmico")
    plt.xlabel("Profundidad (km)")
    plt.ylabel("Momento sísmico (dyn·cm)")
    plt.tight_layout()
    plt.show()


def main():
    configurar_entorno()
    print("Entorno configurado correctamente. Directorios listos.")

    # 1. Diagnóstico de calidad sobre los datos originales.
    print("\n--- Diagnóstico inicial de calidad ---")
    diagnostico_calidad(ARCHIVO_CSV, TABLAS_DIR)

    # 2. Carga y validación para el análisis exploratorio.
    print("\n--- Carga y validación de datos ---")
    df = cargar_y_validar_datos(ARCHIVO_CSV)

    # 3. Generación de gráficos.
    print("\n--- Análisis exploratorio ---")
    generar_graficos(df)


if __name__ == "__main__":
    main()
