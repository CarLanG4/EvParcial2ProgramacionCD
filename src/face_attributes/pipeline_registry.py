"""
PIPELINE REGISTRY — face_attributes
=====================================
Este archivo registra todos los pipelines del proyecto.

Kedro usa este registro para saber qué pipelines existen
y cómo ejecutarlos con `kedro run --pipeline=<nombre>`.

El pipeline "__default__" es el que se ejecuta cuando
corres simplemente `kedro run` (sin especificar cuál).
"""

from kedro.pipeline import Pipeline
from face_attributes.pipelines import (
    data_processing,
    feature_engineering,
    classification,
    clustering,
    reporting,
)


def register_pipelines() -> dict[str, Pipeline]:
    """
    Registra todos los pipelines del proyecto.

    Returns
    -------
    dict con nombre → Pipeline. Kedro los detecta automáticamente.
    """
    # Instanciamos cada pipeline
    dp_pipeline   = data_processing.create_pipeline()
    fe_pipeline   = feature_engineering.create_pipeline()
    clf_pipeline  = classification.create_pipeline()
    clu_pipeline  = clustering.create_pipeline()
    rep_pipeline  = reporting.create_pipeline()

    return {
        # Pipeline individual — solo limpieza de datos
        "data_processing": dp_pipeline,

        # Pipeline individual — solo ingeniería de features
        "feature_engineering": fe_pipeline,

        # Pipeline individual — solo clasificación
        "classification": clf_pipeline,

        # Pipeline individual — solo clustering
        "clustering": clu_pipeline,

        # Pipeline individual — solo reporte final
        "reporting": rep_pipeline,

        # Pipeline completo = todos encadenados en orden
        "__default__": (
            dp_pipeline
            + fe_pipeline
            + clf_pipeline
            + clu_pipeline
            + rep_pipeline
        ),
    }
