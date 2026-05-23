"""
PIPELINE 1 — DATA PROCESSING
==============================
Nodos que limpian y validan el dataset Attributes.csv.

¿Qué hace este pipeline?
- Carga el CSV crudo
- Elimina duplicados y filas con nulos
- Valida que todas las columnas sean binarias (0 o 1)
- Genera estadísticas básicas de cada atributo
- Guarda el dataset limpio para los siguientes pipelines
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


def load_and_validate(df: pd.DataFrame, features_to_drop: list) -> pd.DataFrame:
    """
    Nodo 1: Carga, valida y limpia el dataset.

    Recibe el DataFrame crudo (Kedro lo inyecta desde el catalog).
    - Elimina columnas no útiles (image_id)
    - Elimina filas duplicadas
    - Elimina filas con valores nulos
    - Valida que los valores sean 0 o 1 (binarios)

    Parameters
    ----------
    df : pd.DataFrame — dataset crudo con 34 columnas
    features_to_drop : list — columnas a eliminar (viene de parameters.yml)

    Returns
    -------
    pd.DataFrame limpio
    """
    logger.info(f"Dataset crudo: {df.shape[0]} filas, {df.shape[1]} columnas")

    # Eliminar columnas que no aportan (ej: image_id)
    cols_existentes = [c for c in features_to_drop if c in df.columns]
    df = df.drop(columns=cols_existentes)
    logger.info(f"Columnas eliminadas: {cols_existentes}")

    # Eliminar duplicados
    antes = len(df)
    df = df.drop_duplicates()
    logger.info(f"Duplicados eliminados: {antes - len(df)}")

    # Eliminar nulos
    antes = len(df)
    df = df.dropna()
    logger.info(f"Filas con nulos eliminadas: {antes - len(df)}")

    # Validar que todos los valores sean 0 o 1
    for col in df.columns:
        valores_unicos = df[col].unique()
        if not all(v in [0, 1] for v in valores_unicos):
            logger.warning(f"Columna '{col}' tiene valores no binarios: {valores_unicos}")

    logger.info(f"Dataset limpio final: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def generate_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nodo 2: Genera estadísticas descriptivas de cada atributo.

    Para cada columna calcula:
    - Cantidad de 1s (presencia del atributo)
    - Porcentaje de personas con ese atributo
    - Desviación estándar

    Útil para entender la distribución del dataset.

    Parameters
    ----------
    df : pd.DataFrame — dataset limpio

    Returns
    -------
    pd.DataFrame con estadísticas por columna (se guarda como clean_attributes)
    """
    stats = []
    for col in df.columns:
        stats.append({
            "atributo": col,
            "total_con_atributo": int(df[col].sum()),
            "porcentaje": round(df[col].mean() * 100, 2),
            "std": round(df[col].std(), 4)
        })

    stats_df = pd.DataFrame(stats).sort_values("porcentaje", ascending=False)
    logger.info(f"Estadísticas generadas para {len(stats)} atributos")

    # Loggear los 5 atributos más comunes
    top5 = stats_df.head(5)["atributo"].tolist()
    logger.info(f"Top 5 atributos más comunes: {top5}")

    # Retornamos el df limpio (el stats_df es solo informativo)
    return df
