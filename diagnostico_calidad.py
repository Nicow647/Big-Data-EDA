
from pathlib import Path
import argparse
import re

import pandas as pd
import numpy as np


def diagnostico_calidad(archivo_csv: Path, carpeta_salida: Path) -> pd.DataFrame:
    """
    Revisa la calidad básica del dataset y exporta un CSV con:
    - Indicadores de calidad por columna.
    - Cantidad de filas duplicadas.
    - Validación de fechas, horas e identificadores de eventos.
    - Detección de valores infinitos en columnas numéricas.
    """

    # Cargar dataset
    df = pd.read_csv(archivo_csv)

    resultados = []

    # Diagnóstico individual de cada columna
    for columna in df.columns:
        serie = df[columna]

        resultados.append({
            "tipo_registro": "columna",
            "columna": columna,
            "tipo_dato": str(serie.dtype),
            "total_registros": len(df),
            "valores_nulos": int(serie.isna().sum()),
            "porcentaje_nulos": round(
                float(serie.isna().mean() * 100), 4
            ),
            "valores_unicos": int(serie.nunique(dropna=True)),
            "duplicados_exactos_fila": pd.NA,
            "observacion": ""
        })

    # Indicadores globales de duplicados y valores nulos
    duplicados_exactos = int(df.duplicated().sum())
    total_celdas = len(df) * len(df.columns)
    total_nulos = int(df.isna().sum().sum())

    porcentaje_nulos_global = (
        round(float(total_nulos / total_celdas * 100), 4)
        if total_celdas > 0
        else 0.0
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

    # Validaciones específicas de las columnas disponibles
    for columna in ("date", "time", "event name"):

        if columna not in df.columns:
            continue

        valores = df[columna]

        if columna == "date":
            # Formato esperado: YYYY-MM-DD
            fechas = pd.to_datetime(
                valores,
                format="%Y-%m-%d",
                errors="coerce"
            )

            invalidos = int(
                (fechas.isna() & valores.notna()).sum()
            )

            detalle = "Fechas inválidas (formato YYYY-MM-DD)"

        elif columna == "time":
            # Formato esperado: HH:MM:SS, opcionalmente con decimales
            patron_hora = (
                r"(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d(?:\.\d+)?"
            )

            horas_validas = valores.astype("string").str.fullmatch(
                patron_hora,
                na=False
            )

            invalidos = int(
                (~horas_validas & valores.notna()).sum()
            )

            detalle = "Horas inválidas (formato HH:MM:SS)"

        else:
            # Detecta filas cuyo nombre de evento aparece más de una vez
            invalidos = int(
                valores.duplicated(keep=False).sum()
            )

            detalle = "Filas cuyo identificador de evento está repetido"

        resultados.append({
            "tipo_registro": "validacion",
            "columna": columna,
            "tipo_dato": str(valores.dtype),
            "total_registros": len(df),
            "valores_nulos": int(valores.isna().sum()),
            "porcentaje_nulos": round(
                float(valores.isna().mean() * 100), 4
            ),
            "valores_unicos": int(valores.nunique(dropna=True)),
            "duplicados_exactos_fila": pd.NA,
            "observacion": f"{detalle}: {invalidos}"
        })

    # Detectar valores infinitos en columnas numéricas
    columnas_numericas = df.select_dtypes(
        include=[np.number]
    ).columns

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

    # Crear y exportar el diagnóstico
    diagnostico = pd.DataFrame(resultados)

    carpeta_salida.mkdir(
        parents=True,
        exist_ok=True
    )

    destino = carpeta_salida / "diagnostico_calidad.csv"

    diagnostico.to_csv(
        destino,
        index=False,
        encoding="utf-8-sig"
    )

    # Mostrar resumen en la terminal
    print(f"Filas analizadas: {len(df):,}")
    print(f"Columnas analizadas: {len(df.columns)}")
    print(f"Valores nulos: {total_nulos}")
    print(f"Duplicados exactos: {duplicados_exactos}")
    print(f"Valores infinitos: {infinitos}")
    print(f"Diagnóstico guardado en: {destino}")

    return diagnostico


def main():
    parser = argparse.ArgumentParser(
        description="Diagnóstico inicial de calidad de datos."
    )

    parser.add_argument(
        "csv",
        help="Ruta al archivo CSV del dataset"
    )

    parser.add_argument(
        "--salida",
        default=None,
        help=(
            "Carpeta de salida. Por defecto: "
            "resultados/tablas junto al script."
        )
    )

    args = parser.parse_args()

    ruta_script = Path(__file__).resolve().parent
    archivo_csv = Path(args.csv).expanduser().resolve()

    carpeta_salida = (
        Path(args.salida).expanduser().resolve()
        if args.salida
        else ruta_script / "resultados" / "tablas"
    )

    if not archivo_csv.exists():
        raise FileNotFoundError(
            f"No se encontró el CSV: {archivo_csv}"
        )

    diagnostico_calidad(
        archivo_csv,
        carpeta_salida
    )


if __name__ == "__main__":
    main()
