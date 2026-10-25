
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from diagnostico_calidad import diagnostico_calidad


# Rutas del proyecto
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
    print(
        f"Fechas y horas que no pudieron convertirse: {fechas_invalidas}"
    )

    df = df.drop(columns=["date", "time"])

    # Conservar la lógica original del proyecto.
    df = df.drop_duplicates(subset=["event name"], keep="first")

    print(df.head(12))
    df.info()

    print("Dataset validado con éxito.")
    return df


def estadistica_descriptiva(df):
    """Genera estadísticas descriptivas y matrices de correlación."""
    num_cols = ["magnitude", "depth", "moment", "half duration"]
    cols_evaluar = [c for c in num_cols if c in df.columns]

    if not cols_evaluar:
        raise ValueError(
            "No se encontraron las columnas numéricas esperadas."
        )

    # Estadísticas descriptivas generales.
    desc = df[cols_evaluar].describe(
        percentiles=[0.25, 0.5, 0.75]
    ).T

    desc = desc.rename(
        columns={
            "50%": "mediana",
            "25%": "Q1",
            "75%": "Q3",
            "std": "desv_std",
        }
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

    # Clasificación de profundidad.
    if "depth_category" not in df.columns and "depth" in df.columns:
        bins = [-np.inf, 70, 300, np.inf]
        labels = ["Superficial", "Intermedio", "Profundo"]

        df["depth_category"] = pd.cut(
            df["depth"],
            bins=bins,
            labels=labels,
        )

    # Estadísticas agrupadas por categoría de profundidad.
    if "depth_category" in df.columns:
        grupo_stats = df.groupby(
            "depth_category",
            observed=False,
        )[cols_evaluar].agg(["count", "mean", "median", "std"])

        grupo_stats.to_csv(TABLAS_DIR / "descriptiva_grupos.csv")

    # Correlaciones de Pearson y Spearman.
    corr_pearson = df[cols_evaluar].corr(method="pearson")
    corr_spearman = df[cols_evaluar].corr(method="spearman")

    corr_export = pd.concat(
        [corr_pearson, corr_spearman],
        keys=["Pearson", "Spearman"],
    )

    # Exportación de resultados.
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)

    desc.to_csv(TABLAS_DIR / "descriptiva_general.csv")
    corr_export.to_csv(TABLAS_DIR / "correlaciones.csv")

    print("Estadísticas y correlaciones exportadas con éxito.")
    return df


def generar_graficos(df):
    """Genera los gráficos exploratorios principales."""
    print("Generando gráficos...")

    # Gráfico 1: distribución de magnitudes.
    print("Gráfico 1: distribución de magnitudes")

    plt.figure(figsize=(10, 6))
    df["magnitude"].hist()
    plt.title("Distribución de Magnitudes Sísmicas (1976-2021)")
    plt.xlabel("Magnitud (Mw)")
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "distribucion_magnitudes.png", dpi=300)
    plt.show()
    plt.close()

    # Gráfico 2: profundidad y momento sísmico.
    print("Gráfico 2: profundidad vs. momento sísmico")

    plt.figure(figsize=(10, 6))
    plt.scatter(df["depth"], df["moment"], alpha=0.6)
    plt.title("Relación entre profundidad y momento sísmico")
    plt.xlabel("Profundidad (km)")
    plt.ylabel("Momento sísmico (dyn·cm)")
    plt.tight_layout()
    plt.savefig(FIGURAS_DIR / "profundidad_vs_momento.png", dpi=300)
    plt.show()
    plt.close()

    print("Gráficos generados correctamente.")


def main():
    configurar_entorno()
    print("Entorno configurado correctamente. Directorios listos.")

    # 1. Diagnóstico de calidad sobre los datos originales.
    print("\n--- Diagnóstico inicial de calidad ---")
    diagnostico_calidad(ARCHIVO_CSV, TABLAS_DIR)

    # 2. Carga y validación de datos.
    print("\n--- Carga y validación de datos ---")
    df = cargar_y_validar_datos(ARCHIVO_CSV)

    # 3. Estadísticas descriptivas y correlaciones.
    print("\n--- Estadística descriptiva ---")
    df = estadistica_descriptiva(df)

    # 4. Generación de gráficos.
    print("\n--- Análisis exploratorio ---")
    generar_graficos(df)


if __name__ == "__main__":
    main()
