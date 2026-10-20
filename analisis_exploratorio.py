
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# =============================================================================
# MÓDULO 1: CONFIGURACIÓN DE ENTORNO Y REPRODUCIBILIDAD
# =============================================================================
def configurar_entorno(semilla: int = 42) -> Path:
    """
        Configura la semilla aleatoria global y asegura la existencia
            de la carpeta de salida usando rutas relativas al proyecto.
            
            Args:
                seed (int): Número para replicar los resultados aleatorios.
                nombre_salida (str): Nombre de la carpeta de resultados.
                
            Returns:
                Path: Ruta absoluta de la carpeta de salida (calculada dinámicamente).
    """
    np.random.seed(semilla)
    ruta_raiz = Path(__file__).resolve().parent

    # Crear carpetas de salida si no existen
    (ruta_raiz / "resultados" / "tablas").mkdir(parents=True, exist_ok=True)
    (ruta_raiz / "resultados" / "figuras").mkdir(parents=True, exist_ok=True)

    print(f"[CONFIG] Directorio raíz configurado en: {ruta_raiz}")
    print("[CONFIG] Carpetas 'resultados/tablas/' y 'resultados/figuras/' verificadas.")
    return ruta_raiz

# =============================================================================
# MÓDULO 2: CARGA, LECTURA ROBUSTA Y VALIDACIÓN DE VOLUMEN
# =============================================================================
def cargar_y_validar_datos(ruta_archivo: Path, umbral_filas: int = 50000) -> pd.DataFrame:
    """
    Primero, abre el archivo CSV y confirma que el archivo exista en el disco.
    Después, revisa que esté en UTF-8, verifica el tamaño de las columnas y asegúrarse
    de que tenga todas las filas necesarias
    """
    if not ruta_archivo.is_file():
        raise FileNotFoundError(
            f"Error crítico: No se encontró el dataset en la ruta:\n{ruta_archivo}\n"
            "Asegúrese de colocar el archivo .csv en la carpeta 'datos/' o en la raíz del proyecto."
        )

    print(f"\n[CARGA] Iniciando lectura de '{ruta_archivo.name}'...")

    try:
        # Lectura con codificación UTF-8 explícita
        df = pd.read_csv(ruta_archivo, encoding="utf-8")
        print("[CARGA] El archivo fue leído exitosamente con codificación UTF-8.")
    except UnicodeDecodeError:
        print("[AVISO] Falló UTF-8 estándar. Intentando decodificación alternativa (latin1)...")
        df = pd.read_csv(ruta_archivo, encoding="latin1")
        print("[CARGA] Archivo leído con codificación latin1.")

    filas, columnas = df.shape
    print(f"[INSPECCIÓN] Filas iniciales: {filas:,} | Columnas detectadas: {columnas}")

    # Validación mínimo 50.000 filas
    if filas < umbral_filas:
        raise ValueError(
            f"Incumplimiento de datos: El dataset contiene {filas:,} filas, "
            f"pero se exige un mínimo de {umbral_filas:,} registros."
        )

    print(f"[VALIDACIÓN OK] Cumple con el criterio de volumen (≥ {umbral_filas:,} filas sin contar encabezado).")
    return df