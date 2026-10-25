

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


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


def diagnostico_calidad(archivo_csv: Path, carpeta_salida: Path) -> pd.DataFrame:
    """
    Revisa la calidad del dataset y exporta un CSV con indicadores
    por columna, duplicados, fechas, horas, eventos y valores infinitos.
    """
    df = pd.read_csv(archivo_csv)
    resultados = []

    # Diagnóstico individual de cada columna.
    for columna in df.columns:
        serie = df[columna]

        resultados.append({
            "tipo_registro": "columna",
            "columna": columna,
            "tipo_dato": str(serie.dtype),
            "total_registros": len(df),
            "valores_nulos": int(serie.isna().sum()),
            "porcentaje_nulos": round(float(serie.isna().mean() * 100), 4),
            "valores_unicos": int(serie.nunique(dropna=True)),
            "duplicados_exactos_fila": pd.NA,
            "observacion": ""
        })

    # Resumen global.
    duplicados_exactos = int(df.duplicated().sum())
    total_celdas = len(df) * len(df.columns)
    total_nulos = int(df.isna().sum().sum())

    porcentaje_nulos_global = (
        round(float(total_nulos / total_celdas * 100), 4)
        if total_celdas > 0 else 0.0
    )

    resultados.append({
        "tipo_registro": "resumen",
        "columna": "__dataset__",
        "tipo_dato": "",
        "total_registros": len(df),
        "valores_nulos": total_nulos,
        "porcentaje_nulos": porcentaje_nulos_global,
        "valores_unicos": pd.NA,
        "duplicados_exactos_fila": duplicados_exactos,
        "observacion": "Duplicados exactos de filas en el dataset"
    })

    # Validaciones de fechas, horas e identificadores.
    for columna in ("date", "time", "event name"):
        if columna not in df.columns:
            continue

        valores = df[columna]

        if columna == "date":
            fechas = pd.to_datetime(
                valores, format="%Y-%m-%d", errors="coerce"
            )
            invalidos = int((fechas.isna() & valores.notna()).sum())
            detalle = "Fechas inválidas (formato YYYY-MM-DD)"

        elif columna == "time":
            patron_hora = (
                r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?"
            )
            horas_validas = valores.astype("string").str.fullmatch(
                patron_hora, na=False
            )
            invalidos = int((~horas_validas & valores.notna()).sum())
            detalle = "Horas inválidas (formato HH:MM:SS)"

        else:
            invalidos = int(valores.duplicated(keep=False).sum())
            detalle = "Filas cuyo identificador de evento está repetido"

        resultados.append({
            "tipo_registro": "validacion",
            "columna": columna,
            "tipo_dato": str(valores.dtype),
            "total_registros": len(df),
            "valores_nulos": int(valores.isna().sum()),
            "porcentaje_nulos": round(float(valores.isna().mean() * 100), 4),
            "valores_unicos": int(valores.nunique(dropna=True)),
            "duplicados_exactos_fila": pd.NA,
            "observacion": f"{detalle}: {invalidos}"
        })

    # Detectar valores infinitos en columnas numéricas.
    columnas_numericas = df.select_dtypes(include=[np.number]).columns

    if len(columnas_numericas) > 0:
        infinitos = int(
            np.isinf(df[columnas_numericas]).sum().sum()
        )
    else:
        infinitos = 0

    resultados.append({
        "tipo_registro": "resumen",
        "columna": "__numericas__",
        "tipo_dato": "",
        "total_registros": len(df),
        "valores_nulos": pd.NA,
        "porcentaje_nulos": pd.NA,
        "valores_unicos": pd.NA,
        "duplicados_exactos_fila": pd.NA,
        "observacion": (
            f"Valores infinitos en columnas numéricas: {infinitos}"
        )
    })

    diagnostico = pd.DataFrame(resultados)
    carpeta_salida.mkdir(parents=True, exist_ok=True)
    destino = carpeta_salida / "diagnostico_calidad.csv"

    diagnostico.to_csv(
        destino, index=False, encoding="utf-8-sig"
    )

    print(f"Filas analizadas: {len(df):,}")
    print(f"Columnas analizadas: {len(df.columns)}")
    print(f"Valores nulos: {total_nulos}")
    print(f"Duplicados exactos: {duplicados_exactos}")
    print(f"Valores infinitos: {infinitos}")
    print(f"Diagnóstico guardado en: {destino}")

    return diagnostico


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
        raise ValueError("No se encontraron las columnas numéricas esperadas.")

    desc = df[cols_evaluar].describe(
        percentiles=[0.25, 0.5, 0.75]
    ).T

    desc = desc.rename(columns={
        "50%": "mediana",
        "25%": "Q1",
        "75%": "Q3",
        "std": "desv_std"
    })

    desc["IQR"] = desc["Q3"] - desc["Q1"]

    columnas_orden = [
        "count", "mean", "mediana", "desv_std",
        "min", "Q1", "Q3", "max", "IQR"
    ]
    desc = desc[[c for c in columnas_orden if c in desc.columns]]

    if "depth_category" not in df.columns and "depth" in df.columns:
        bins = [-np.inf, 70, 300, np.inf]
        labels = ["Superficial", "Intermedio", "Profundo"]
        df["depth_category"] = pd.cut(
            df["depth"], bins=bins, labels=labels
        )

    if "depth_category" in df.columns:
        grupo_stats = df.groupby(
            "depth_category", observed=False
        )[cols_evaluar].agg(["count", "mean", "median", "std"])

        grupo_stats.to_csv(TABLAS_DIR / "descriptiva_grupos.csv")

    corr_pearson = df[cols_evaluar].corr(method="pearson")
    corr_spearman = df[cols_evaluar].corr(method="spearman")

    corr_export = pd.concat(
        [corr_pearson, corr_spearman],
        keys=["Pearson", "Spearman"]
    )

    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    desc.to_csv(TABLAS_DIR / "descriptiva_general.csv")
    corr_export.to_csv(TABLAS_DIR / "correlaciones.csv")

    print("Estadísticas y correlaciones exportadas con éxito.")
    return df


def generar_graficos(df):
    """Genera y guarda los gráficos exploratorios principales."""
    print("Generando gráficos...")

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

    # 1. Diagnóstico de calidad: ahora es una función de este archivo.
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
