
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