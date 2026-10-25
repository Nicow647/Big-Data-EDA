# Laboratorio 04: Análisis Exploratorio de Datos (EDA) - Terremotos en Chile

# Integrantes (Equipo 6)
- Nicolás Venegas Gaona (Nicow647)
- Kamila Castillo Montes (kamila-castillo-montes)
- Matías Becerra Araneda (matiasbenja7)
- Matías Catalán Moreno (mcbm-uni)
- Matías Palma Sepúlveda (Garluix)
- Rúbén Gorigoitía Barraza (Rub3n-Gori)



Este proyecto contiene un pipeline automatizado en Python
para realizar un Análisis Exploratorio de Datos (EDA) sobre
el historial de terremotos en Chile (1976-2021).

# Requisitos Previos
- **Python:** Versión 3.11
- **Entorno Virtual:** Virtualenv (venv)

# Instalación y Configuración
1. **Clonar repositorio:**
   https://github.com/Nicow647/Big-Data-EDA.git

2. **Crear y activar el entorno virtual:**
   * Para Windows (Recomiendo usar la terminal CMD; Powershell me daba algunos errores):
      - python3.11 -m venv venv_eda // Crear entorno virtual
      - venv_eda\Scripts\activate // Activar entorno
   * Para Linux/macOs: 
      - python3.11 -m venv venv_eda // Crear entorno virtual
      - source venv_eda/bin/activate // Activar entorno

3. **Instalar dependencias:**
  - pip install -r requirements.txt

# Dataset
* Dataset crudo en el directorio:
   - /datos/Chile Earthquake Dataset (1976-2021).csv
* También se puede visualizar desde la página de descarga:
   - https://www.kaggle.com/datasets/reciclador/chile-earthquake-dataset-1976-2021

# Comando de Ejecución
* Ejecutar el comando en la terminal:
   - python analisis_exploratorio.py
