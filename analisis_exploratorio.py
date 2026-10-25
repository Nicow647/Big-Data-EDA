from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


def estadistica_descriptiva(df):
    num_cols = ["magnitude", "depth", "moment", "half duration"]
    cols_evaluar = [c for c in num_cols if c in df.columns]

    # Estadísticas descriptivas univariadas
    desc = df[cols_evaluar].describe(percentiles=[0.25, 0.5, 0.75]).T
    desc = desc.rename(
        columns={"50%": "mediana", "25%": "Q1", "75%": "Q3", "std": "desv_std"}
    )
    desc["IQR"] = desc["Q3"] - desc["Q1"]

    columnas_orden = [
        "count",
        "mean",
        "mediana",
        "desv_std",
        "min",
        "Q1",
        "Q3",
        "max",
        "IQR",
    ]
    desc = desc[[c for c in columnas_orden if c in desc.columns]]

    # Derivar depth_category si no viene del pipeline de limpieza
    if "depth_category" not in df.columns and "depth" in df.columns:
        bins = [-np.inf, 70, 300, np.inf]
        labels = ["Superficial", "Intermedio", "Profundo"]
        df["depth_category"] = pd.cut(df["depth"], bins=bins, labels=labels)

    # Comparación por grupos
    if "depth_category" in df.columns:
        grupo_stats = df.groupby("depth_category", observed=False)[cols_evaluar].agg(
            ["count", "mean", "median", "std"]
        )
        grupo_stats.to_csv(TABLAS_DIR / "descriptiva_grupos.csv")

    # Matrices de correlación (Pearson y Spearman)
    corr_pearson = df[cols_evaluar].corr(method="pearson")
    corr_spearman = df[cols_evaluar].corr(method="spearman")
    corr_export = pd.concat(
        [corr_pearson, corr_spearman], keys=["Pearson", "Spearman"]
    )

    # Exportación
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    desc.to_csv(TABLAS_DIR / "descriptiva_general.csv")
    corr_export.to_csv(TABLAS_DIR / "correlaciones.csv")

    print("Estadísticas y correlaciones exportadas con éxito.")
    return df


def generar_graficos(df):
    print("Generando graficos...")

    print("Grafico 1 (Distribuición de Magnitudes)\n")
    df["magnitude"].hist()
    plt.title("Distribuición de Magnitudes Sísmicas (1976-2021)")
    plt.xlabel("Magnitud (Mw)")
    plt.ylabel("Frecuencia")
    plt.show()

    print("Grafico 2 (Relación de profundidad y energía liberada (momento sismico))\n")
    plt.scatter(df["depth"], df["moment"])
    plt.title("Relación profundidad vs momento sísmico")
    plt.xlabel("Profunidad (km)")
    plt.ylabel("Momento Sísmico (dyn*cm)")
    plt.show()


def main():
    configurar_entorno()
    print("Entorno configurado correctamente. Directorios listos.")

    df = cargar_y_validar_datos(DATA_DIR / "Chile Earthquake Dataset (1976-2021).csv")
    # Integrar las funciones aquí
    df = estadistica_descriptiva(df)
    generar_graficos(df)


if __name__ == "__main__":
    main()