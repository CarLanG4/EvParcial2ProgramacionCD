# face_attributes — Proyecto Kedro

Proyecto de Machine Learning con Kedro que analiza el dataset `Attributes.csv`
(30.141 imágenes de caras con 33 atributos binarios) usando 4 pipelines.

## Estructura del proyecto

```
face-attributes/
├── conf/
│   └── base/
│       ├── catalog.yml       ← define qué datasets existen y dónde están
│       └── parameters.yml    ← hiperparámetros centralizados
├── data/
│   ├── 01_raw/               ← Attributes.csv original
│   ├── 02_intermediate/      ← datos limpios
│   ├── 03_primary/           ← datos con nuevas features
│   ├── 04_feature/           ← X_train, X_test, y_train, y_test
│   ├── 07_model_output/      ← métricas y resultados
│   └── 08_reporting/         ← reporte final
├── src/face_attributes/
│   ├── pipelines/
│   │   ├── data_processing/      ← Pipeline 1
│   │   ├── feature_engineering/  ← Pipeline 2
│   │   ├── classification/       ← Pipeline 3
│   │   ├── clustering/           ← Pipeline 4
│   │   └── reporting/            ← Pipeline 5 (bonus)
│   └── pipeline_registry.py  ← registra todos los pipelines
└── requirements.txt
```

## Instalación desde cero (Git Bash o PowerShell)

```bash
# 1. Crear y activar entorno virtual
python -m venv venv

# En Git Bash:
source venv/Scripts/activate
# En PowerShell:
venv\Scripts\Activate.ps1

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Copiar el dataset
# Pegar Attributes.csv en data/01_raw/
```

## Ejecución

```bash
# Ejecutar todos los pipelines (orden automático)
kedro run

# Ejecutar un pipeline específico
kedro run --pipeline=data_processing
kedro run --pipeline=feature_engineering
kedro run --pipeline=classification
kedro run --pipeline=clustering
kedro run --pipeline=reporting

# Ver el grafo de dependencias
kedro viz
```

## Los 4 Pipelines

| # | Pipeline | Qué hace | Input → Output |
|---|----------|----------|----------------|
| 1 | `data_processing` | Limpia el CSV: elimina duplicados, valida valores binarios | `Attributes.csv` → `clean_attributes.csv` |
| 2 | `feature_engineering` | Crea scores compuestos y divide en train/test | `clean_attributes` → `X_train`, `X_test`, `y_train`, `y_test` |
| 3 | `classification` | Entrena y evalúa 3 modelos para predecir `attractive` | `X_train/test` → `classification_metrics.csv` |
| 4 | `clustering` | Agrupa caras en 3 perfiles con K-Means | `features_attributes` → `clustering_results.csv` |
