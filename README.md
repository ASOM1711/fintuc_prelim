# Fintuc Prelim

Repositorio del modelo preliminar de portafolios. El modelo final combina Black-Litterman hibrido con Markowitz, restricciones por perfil de riesgo y simulaciones de comportamiento de clientes.

## Como correr el asesor interactivo

Este es el codigo nuevo para ver mes a mes, en la terminal, el portafolio actual y el portafolio recomendado.

```powershell
python .\interactive_portfolio_advisor.py
```

Si `python` no toma el entorno correcto:

```powershell
& "C:\Users\Gustavo\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" .\interactive_portfolio_advisor.py
```

El programa pregunta:

- perfil de riesgo;
- año de inicio del testing;
- año final del testing.

Por defecto parte con **USD 1.000**, usa **2019-2025**, y no usa 2026 porque la informacion de ese año es parcial.

Tambien se puede correr directo con argumentos:

```powershell
python .\interactive_portfolio_advisor.py --perfil arriesgado --start 2024-01-01 --end 2024-03-31
```

Para probar pocos meses:

```powershell
python .\interactive_portfolio_advisor.py --perfil arriesgado --start 2024-01-01 --end 2024-12-31 --max-months 3
```

Para que no pregunte si aceptar/rechazar rebalanceos:

```powershell
python .\interactive_portfolio_advisor.py --perfil arriesgado --auto no --max-months 3
```

## Que hace el asesor interactivo

Archivo: `interactive_portfolio_advisor.py`

Hace una simulacion mensual interactiva usando el modelo final:

- parte con USD 1.000;
- el usuario elige el perfil;
- calcula un portafolio inicial con Black-Litterman + Markowitz;
- mes a mes muestra el portafolio actual;
- calcula un portafolio recomendado rebalanceado;
- muestra prediccion anual, varianza anual, volatilidad anual, caso favorable y caso desfavorable;
- el caso favorable es `prediccion + volatilidad`;
- el caso desfavorable es `prediccion - volatilidad`;
- permite aceptar o rechazar el rebalanceo;
- actualiza el valor del portafolio usando retornos historicos reales;
- si no existe un portafolio compatible con la tolerancia del perfil, explica que el riesgo estimado supera lo que el perfil acepta y pregunta si continuar o abandonar.

## Configuracion final del modelo

Archivo: `config.py`

Contiene los parametros finales:

- `TRAIN_WINDOW_YEARS = 5`
- `BL_METHOD = "robust_factor_hybrid"`
- `BL_CONF_BASE = 1.0`
- `BL_LOOKBACK = 126`
- `BL_SKIP = 21`
- `MAX_WEIGHT = 0.025` como fallback
- `PROFILE_MAX_WEIGHTS` por perfil:
  - `muy_conservador`: 2,5%
  - `conservador`: 2,5%
  - `neutro`: 1%
  - `arriesgado`: 2,5%
  - `muy_arriesgado`: 5%

La tolerancia de cada perfil representa la perdida maxima anual que ese perfil esta dispuesto a aceptar.

## Estructura principal

```text
fintuc_prelim/
  config.py
  interactive_portfolio_advisor.py
  README.md

  data/
  models/
  portfolio/
  backtesting/
  visualization/

  experiments/
  scripts/
  resultados_diagnostico/
  resultados_prediccion/
  output/pdf/
```

## Codigo del modelo

### `data/`

- `data/loader.py`: carga los retornos de acciones, dividend yields e informacion de acciones. Es la entrada de datos del proyecto.
- `data/__init__.py`: marcador de paquete.

### `models/`

- `models/black_littermanprelim.py`: implementa Black-Litterman original, Black-Litterman robusto por factores y variantes hibridas. El modelo final usa `robust_factor_hybrid`, una mezcla 65% factor BL y 35% asset momentum BL.
- `models/markowitzprelim.py`: optimizador Markowitz con Gurobi, restriccion de volatilidad anual, `max_weight` y opcion `full_invest`.
- `models/__init__.py`: marcador de paquete.

### `portfolio/`

- `portfolio/engine.py`: motor mensual del portafolio. Actualiza valor diario, dividendos, caja, comision, rebalanceo, drawdown, probabilidad de aceptacion y abandono.
- `portfolio/probabilities.py`: funciones probabilisticas:
  - `p1_abandono`: probabilidad de abandono segun drawdown vs tolerancia.
  - `p2_aceptacion`: probabilidad de aceptar rebalanceo segun retorno esperado vs tolerancia.
- `portfolio/__init__.py`: marcador de paquete.

### `backtesting/`

- `backtesting/runner.py`: funciones centrales de backtesting:
  - `run_backtest`
  - `run_all_profiles`
  - `run_monte_carlo`
  - `resumen_monte_carlo`
  - `resumen_backtest`
  - `calcular_ic`
- `backtesting/prediction_error.py`: rolling test del error predictivo, MAE, RMSE, bias e hit rate.
- `backtesting/sensitivity.py`: utilidades de sensibilidad de parametros.
- `backtesting/__init__.py`: marcador de paquete.

### `visualization/`

- `visualization/plots.py`: funciones de graficos.
- `visualization/__init__.py`: marcador de paquete.

## Experimentos

Los scripts de investigacion y diagnostico quedaron ordenados en `experiments/`. Para correrlos desde la raiz del repo, usar formato modulo:

```powershell
python -m experiments.max_weight.max_weight_hybrid_vs_sin_bl
```

### `experiments/black_litterman/`

- `black_litterman_sensitivity_final.py`: prueba candidatos razonables de Black-Litterman.
- `compare_black_litterman_final.py`: compara BL final contra sin BL usando la configuracion de train window y `max_weight` definida en ese momento.

### `experiments/max_weight/`

- `max_weight_hybrid_vs_sin_bl.py`: compara `max_weight` 1%, 2,5%, 5%, 10%, 15% y 20% con BL nuevo vs sin BL.
- `max_weight_decision_test.py`: test inicial para decidir `max_weight`.
- `max_weight_decision_test_train5.py`: test de `max_weight` con ventana de 5 años.
- `max_weight_train5_add_1pct.py`: agrega el caso 1% al test de ventana 5 años.
- `max_weight_anual_independiente.py`: retornos anuales independientes para diferentes `max_weight`.

### `experiments/montecarlo/`

- `montecarlo_main.py`: Monte Carlo original con clientes simulados.
- `montecarlo_probabilidades_final.py`: Monte Carlo conductual 2019-2025 con 5.000 clientes por perfil.
- `montecarlo_probabilidades_por_anio.py`: Monte Carlo conductual año por año, reiniciando capital y clientes cada año.

Este Monte Carlo no simula precios con movimiento browniano. Usa retornos historicos reales y simula decisiones de clientes: aceptacion de rebalanceos y abandono.

### `experiments/prediction_error/`

- `prediction_error_main.py`: error predictivo base.
- `prediction_error_lambda0.py`: error predictivo usando lambda 0.
- `prediction_error_lambdas_compare.py`: compara errores predictivos entre lambdas.

### `experiments/train_window/`

- `train_window_decision_test.py`: compara ventanas de training.
- `train_window_error_resume.py`: retoma calculos de error por ventana.

### `experiments/annual_returns/`

- `resultados_por_anio.py`: resultados por año.
- `retornos_anuales_independientes.py`: retornos anuales independientes.
- `retornos_anuales_lambda0.py`: retornos anuales con lambda 0.

### `experiments/annual_probabilities/`

- `probabilidades_anuales.py`: probabilidades de aceptacion y abandono por año.

### `experiments/rolling/`

- `rolling_promedios_main.py`: rolling testing y promedios por perfil.

### `experiments/lambda/`

- `lambda_sensitivity_main.py`: sensibilidad a lambda.

### `experiments/sensitivity/`

- `sensitivity_main.py`: sensibilidad general.
- `sens_max_weight.py`: sensibilidad de `max_weight` para 2024.

### `experiments/client_behavior/`

- `comp_p1_lineal.py`: compara probabilidad de abandono sigmoid vs lineal.

### `experiments/model_comparison/`

- `comparacion_2024.py`: compara modelo final, Markowitz sin BL y benchmark aleatorio para 2024.

### `experiments/efficient_frontier/`

- `frontera_eficiente.py`: calculos de frontera eficiente.

### `experiments/legacy/`

- `mainprelim.py`: script preliminar antiguo, conservado como referencia historica.

## Scripts de soporte

### `scripts/reports/`

- `create_model_decision_pdf.py`: genera el informe de decision de parametros.
- `create_max_weight_pdf.py`: genera PDF del analisis inicial de `max_weight`.
- `create_lambda0_max_weight_update_pdf.py`: genera PDF de actualizacion lambda 0 y restricciones.
- `create_hybrid_vs_sin_bl_max_weight_pdf.py`: genera PDF comparando BL nuevo vs sin BL.

Ejemplo:

```powershell
python -m scripts.reports.create_hybrid_vs_sin_bl_max_weight_pdf
```

### `scripts/maintenance/`

- `recompute_max_weight_without_partial.py`: recomputa resumen de `max_weight` excluyendo años parciales.

## Resultados

### `resultados_diagnostico/`

Contiene CSVs de diagnostico y resultados de experimentos:

- sensibilidad de Black-Litterman;
- comparaciones con y sin BL;
- tests de `max_weight`;
- retornos por año;
- Monte Carlo conductual;
- ventana de training.

Los CSV antiguos de Monte Carlo que estaban en la raiz se movieron a:

```text
resultados_diagnostico/legacy_montecarlo/
```

### `resultados_prediccion/`

Contiene salidas de error predictivo:

- errores por activo/mes;
- errores por perfil/mes;
- resumen MAE/RMSE/bias;
- comparaciones lambda 0 vs lambda 1.

### `output/pdf/`

Contiene PDFs generados para informe:

- `analisis_max_weight_resultados.pdf`
- `informe_decision_parametros_modelo.pdf`
- `actualizacion_lambda0_max_weight.pdf`
- `actualizacion_hybrid_bl_vs_sin_bl.pdf`

## Que se hizo durante el proyecto

1. Se reviso el efecto de lambda y se termino usando `lambda = 0` para las pruebas finales.
2. Se compararon ventanas de training y se eligio una ventana de 5 años.
3. Se compararon restricciones `max_weight` y se paso desde un peso unico a pesos por perfil.
4. Se desarrollo y eligio un Black-Litterman hibrido 65/35.
5. Se comparo BL nuevo contra modelo sin BL.
6. Se corrio Monte Carlo conductual para aceptacion y abandono.
7. Se creo el asesor interactivo mensual para mostrar el funcionamiento del modelo en terminal.

## Nota sobre imports despues del orden

Los scripts dentro de `experiments/` deben correrse desde la raiz usando `python -m`. Esto mantiene la raiz del repo en el path de Python y evita errores de import.

