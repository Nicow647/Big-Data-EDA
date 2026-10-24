from pathlib import Path
import pandas as pd
import numpy as np

# Rutas Relativas (configurar)
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "datos"
TABLAS_DIR = BASE_DIR / "resultados" / "tablas"
FIGURAS_DIR = BASE_DIR / "resultados" / "figuras"

# 2. Creación de Directorios
DATA_DIR.mkdir(parents=True, exist_ok=True)
TABLAS_DIR.mkdir(parents=True, exist_ok=True)
FIGURAS_DIR.mkdir(parents=True, exist_ok=True)


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
        raise ValueError(
            f"Error: El dataset tiene {filas} filas, se exigen {umbral_filas}."
        )

    # Limpieza
    df["datetime"] = pd.to_datetime(
        df["date"] + " " + df["time"], format="%Y-%m-%d %H:%M:%S.%f", errors="coerce"
    )
    df = df.drop(columns=["date", "time"])
    df = df.drop_duplicates(subset=["event name"], keep="first")

    print(df.head(12))
    df.info()

    print("Dataset validado con éxito.")
    return df


def main():
    configurar_entorno()
    print("Entorno configurado correctamente. Directorios listos.")

    df = cargar_y_validar_datos(DATA_DIR / "Chile Earthquake Dataset (1976-2021).csv")
    # Integrar las funciones aquí


if __name__ == "__main__":
    main()
