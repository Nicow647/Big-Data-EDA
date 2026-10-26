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


def configurar_entorno():
    # Creación de carpetas de resultados
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)


def diagnostico_calidad(df):
    # Calcula indicadores de calidad y los devuelve en una tabla
    resultados = []

    def agregar_resultado(tipo, columna, observacion=""):
        serie = df[columna] if columna in df.columns else None
        resultados.append(
            {
                "tipo_registro": tipo,
                "columna": columna,
                "tipo_dato": str(serie.dtype) if serie is not None else "",
                "total_registros": len(df),
                "valores_nulos": (
                    int(serie.isna().sum()) if serie is not None else pd.NA
                ),
                "porcentaje_nulos": (
                    round(float(serie.isna().mean() * 100), 4)
                    if serie is not None
                    else pd.NA
                ),
                "valores_unicos": (
                    int(serie.nunique(dropna=True)) if serie is not None else pd.NA
                ),
                "duplicados_exactos_fila": pd.NA,
                "observacion": observacion,
            }
        )

    # Diagnóstico individual de cada columna
    for columna in df.columns:
        agregar_resultado("columna", columna)

    # Resumen global
    total_celdas = len(df) * len(df.columns)
    total_nulos = int(df.isna().sum().sum())
    porcentaje_nulos = total_nulos / total_celdas * 100 if total_celdas else 0
    duplicados_exactos = int(df.duplicated().sum())

    agregar_resultado("resumen", "__dataset__", "Duplicados exactos de filas")
    resultados[-1].update(
        {
            "valores_nulos": total_nulos,
            "porcentaje_nulos": round(porcentaje_nulos, 4),
            "duplicados_exactos_fila": duplicados_exactos,
        }
    )

    # Validaciones de fechas, horas e identificadores
    for columna in ("date", "time", "event name"):
        if columna not in df.columns:
            continue

        valores = df[columna]
        if columna == "date":
            fechas = pd.to_datetime(valores, format="%Y-%m-%d", errors="coerce")
            invalidos = int((fechas.isna() & valores.notna()).sum())
            detalle = "Fechas inválidas (formato YYYY-MM-DD)"
        elif columna == "time":
            patron_hora = r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?"
            validos = valores.astype("string").str.fullmatch(patron_hora, na=False)
            invalidos = int((~validos & valores.notna()).sum())
            detalle = "Horas inválidas (formato HH:MM:SS)"
        else:
            invalidos = int(valores.duplicated(keep=False).sum())
            detalle = "Filas con identificador de evento repetido"

        agregar_resultado("validacion", columna, f"{detalle}: {invalidos}")

    # Detectar valores infinitos en columnas numéricas
    columnas_numericas = (
        "latitude",
        "longitude",
        "depth",
        "Mrr",
        "Mtt",
        "Mpp",
        "Mrt",
        "Mrp",
        "Mtp",
        "MrrError",
        "MttError",
        "MppError",
        "MrtError",
        "MrpError",
        "MtpError",
        "half duration",
        "moment",
        "magnitude",
        "exponent",
    )

    infinitos = 0
    for columna in columnas_numericas:
        if columna not in df.columns:
            continue

        valores = df[columna]
        numericos = pd.to_numeric(valores, errors="coerce")
        no_numericos = int((valores.notna() & numericos.isna()).sum())
        agregar_resultado(
            "validacion",
            columna,
            f"Valores no numéricos en columna numérica: {no_numericos}",
        )
        infinitos += int(np.isinf(numericos).sum())

    # Límites amplios para marcar valores que requieren revisión. No se eliminan
    rangos = {
        "latitude": (-90, 90),
        "longitude": (-180, 180),
        "depth": (0, None),
        "magnitude": (0, 10),
        "moment": (0, None),
        "half duration": (0, None),
    }
    for columna, (minimo, maximo) in rangos.items():
        if columna not in df.columns:
            continue

        valores = pd.to_numeric(df[columna], errors="coerce")
        fuera_de_rango = pd.Series(False, index=df.index)
        if minimo is not None:
            fuera_de_rango |= valores < minimo
        if maximo is not None:
            fuera_de_rango |= valores > maximo
        cantidad = int((fuera_de_rango & valores.notna()).sum())
        limite = f"[{minimo}, {maximo}]" if maximo is not None else f">= {minimo}"
        detalle = f"Valores fuera del rango de revisión {limite}: {cantidad}"
        agregar_resultado("validacion", columna, detalle)

    agregar_resultado(
        "resumen",
        "__numericas__",
        f"Valores infinitos en columnas numéricas: {infinitos}",
    )
    return pd.DataFrame(resultados)


def validar_y_preparar_datos(df: pd.DataFrame, umbral_filas=50000):
    # Valida y prepara los datos cargados
    filas, columnas = df.shape

    print(f"Filas leídas: {filas} | Columnas: {columnas}")

    # Chequeo automático de mínimo de registros
    if filas < umbral_filas:
        raise ValueError(
            f"Error: el dataset tiene {filas} filas; "
            f"se exigen al menos {umbral_filas}."
        )

    # Limpieza
    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%Y-%m-%d %H:%M:%S.%f",
        errors="coerce",
    )

    fechas_invalidas = df["datetime"].isna().sum()
    print(f"Fechas y horas que no pudieron convertirse: {fechas_invalidas}")

    df = df.drop(columns=["date", "time"])
    df = df.drop_duplicates(subset=["event name"], keep="first")

    print(df.head(12))
    df.info()

    print("Dataset validado con éxito.")
    return df


def estadistica_descriptiva(df):
    # Genera estadísticas descriptivas y matrices de correlación
    num_cols = ["magnitude", "depth", "moment", "half duration"]
    cols_evaluar = [c for c in num_cols if c in df.columns]

    if not cols_evaluar:
        raise ValueError("No se encontraron las columnas numéricas esperadas.")

    # Asegurar existencia de carpeta antes de guardar cualquier tabla
    TABLAS_DIR.mkdir(parents=True, exist_ok=True)

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

    corr_pearson = df[cols_evaluar].corr(method="pearson")
    corr_spearman = df[cols_evaluar].corr(method="spearman")

    # Matrices de correlación (Pearson y Spearman)
    corr_export = pd.concat([corr_pearson, corr_spearman], keys=["Pearson", "Spearman"])

    # Exportación
    desc.to_csv(TABLAS_DIR / "descriptiva_general.csv")
    corr_export.to_csv(TABLAS_DIR / "correlaciones.csv")

    print("Estadísticas y correlaciones exportadas con éxito.")
    return df


def generar_graficos(df):
    print("Generando gráficos...")

    print("Gráfico 1 (Distribuición de Magnitudes)\n")
    df["magnitude"].hist()
    plt.title("Distribución de Magnitudes Sísmicas (1976-2021)")
    plt.xlabel("Magnitud (Mw)")
    plt.ylabel("Frecuencia")
    plt.savefig(FIGURAS_DIR / "01_distribucion_magnitudes.png", dpi=300)
    plt.show()
    plt.close()

    print("Gráfico 2 (Relación de profundidad y energía liberada (momento sismico))\n")
    plt.scatter(df["depth"], df["moment"])
    plt.title("Relación profundidad vs momento sísmico")
    plt.xlabel("Profundidad (km)")
    plt.ylabel("Momento Sísmico (dyn*cm)")
    plt.savefig(FIGURAS_DIR / "02_profundidad_vs_momento.png", dpi=300)
    plt.show()
    plt.close()

    print("Gráfico 3 (Comparación de magnitudes por profundidad)\n")
    df.boxplot(column="magnitude", by="depth_category")
    plt.title("Distribución de magnitudes por profunidad")
    plt.suptitle("")
    plt.xlabel("Categoría de Profunidad")
    plt.ylabel("Magnitud (Mw)")
    plt.savefig(FIGURAS_DIR / "03_magnitud_vs_profundidad.png", dpi=300)
    plt.show()
    plt.close()

    print("Gráfico 4 (Evolución anual de sismos extremos)\n")
    plt.figure()
    sismos_fuertes = df[df["magnitude"] >= 6.0]
    conteo_anual = sismos_fuertes.groupby(sismos_fuertes["datetime"].dt.year).size()
    plt.plot(conteo_anual.index, conteo_anual.values)
    plt.title("Evolución anual de sismos extremos (Mw >= 6.0)")
    plt.xlabel("Año")
    plt.ylabel("Cantidad de Sismos")
    plt.savefig(FIGURAS_DIR / "04_evolucion_anual.png", dpi=300)
    plt.show()
    plt.close()

    print("Gráfico 5 (Magnitud de sismos extremos por año)\n")
    plt.figure()
    sismos_fuertes = df[df["magnitude"] >= 6.0]
    anios = sismos_fuertes["datetime"].dt.year
    magnitudes = sismos_fuertes["magnitude"]
    plt.scatter(anios, magnitudes)
    plt.title("Magnitud de sismos extremos por año (Mw >= 6.0)")
    plt.xlabel("Año")
    plt.ylabel("Magnitud (Mw)")
    plt.savefig(FIGURAS_DIR / "05_magnitud_por_anio.png", dpi=300)
    plt.show()
    plt.close()

    print("Gráficos generados correctamente.")


def main():
    configurar_entorno()
    print("Entorno configurado correctamente. Directorios listos.")

    # Diagnóstico de calidad
    print("\n--- Diagnóstico inicial de calidad ---")
    if not ARCHIVO_CSV.exists():
        raise FileNotFoundError(f"No se encontró el dataset: {ARCHIVO_CSV}")

    df = pd.read_csv(ARCHIVO_CSV, encoding="utf-8")
    diagnostico = diagnostico_calidad(df)
    ruta_diagnostico = TABLAS_DIR / "diagnostico_calidad.csv"
    diagnostico.to_csv(ruta_diagnostico, index=False, encoding="utf-8-sig")
    print(f"Diagnóstico guardado en: {ruta_diagnostico}")

    # Carga y validación de datos
    print("\n--- Carga y validación de datos ---")
    df = validar_y_preparar_datos(df)

    # Estadísticas descriptivas y correlaciones
    print("\n--- Estadística descriptiva ---")
    df = estadistica_descriptiva(df)

    # Generación de gráficos
    print("\n--- Análisis exploratorio ---")
    generar_graficos(df)


if __name__ == "__main__":
    main()
