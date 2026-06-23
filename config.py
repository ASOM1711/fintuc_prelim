from pathlib import Path

# Rutas de datos
DATA_DIR = Path(r"C:\Users\aguor\OneDrive - Universidad Católica de Chile\G18 Portafolios\Historical_Stocks\universo_300_acciones")
INFO_FILE = Path(r"C:\Users\aguor\OneDrive - Universidad Católica de Chile\G18 Portafolios\E3final\stocks_info.txt")

# Ventana rodante
TRAIN_WINDOW_YEARS = 4

# Modelo
MAX_WEIGHT = 0.15       # máximo 15% por activo
COMMISSION = 0.01       # 1% sobre AUM mensual
DIV_CAP = 0.15          # tope dividendo por evento

# Perfiles de riesgo: tolerancia máxima de pérdida anual
RISK_PROFILES = {
    "muy_conservador": 0.00,
    "conservador":     0.05,
    "neutro":          0.15,
    "arriesgado":      0.30,
    "muy_arriesgado":  0.40,
}
