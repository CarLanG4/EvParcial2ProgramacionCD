"""
PIPELINE 2 — FEATURE ENGINEERING
==================================
Nodos que crean nuevas variables (features) a partir del dataset limpio.

¿Qué hace este pipeline?
- Crea features compuestas agrupando atributos relacionados (ej: score de vello facial)
- Selecciona las features más relevantes usando correlación con el target
- Prepara X_train, X_test, y_train, y_test para los modelos
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def create_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nodo 1: Crea features compuestas combinando atributos relacionados.

    Features nuevas que creamos:
    - score_facial_hair: suma de has_beard + patchy_beard + has_mustache
      → mide cuánto vello facial tiene la persona
    - score_hair_features: long_hair + curly_hair + grey_hair + black_hair
      → mide características generales del cabello
    - score_skin: clear_skin - oily_skin - dark_circles
      → mide calidad de la piel (positivo=buena, negativo=problemas)
    - score_aging: old + wrinkle + double_chin + receeding_hairline
      → mide características de envejecimiento
    - score_feminine: has_makeup + veil + long_hair + big_lips
      → agrupa características típicamente femeninas

    Estas features compuestas capturan patrones que los modelos
    pueden aprovechar mejor que las variables individuales.

    Parameters
    ----------
    df : pd.DataFrame — dataset limpio

    Returns
    -------
    pd.DataFrame con features adicionales
    """
    df = df.copy()

    # Score de vello facial
    cols_facial_hair = [c for c in ['has_beard', 'patchy_beard', 'has_mustache'] if c in df.columns]
    df['score_facial_hair'] = df[cols_facial_hair].sum(axis=1)

    # Score de cabello
    cols_hair = [c for c in ['long_hair', 'curly_hair', 'grey_hair', 'black_hair'] if c in df.columns]
    df['score_hair_features'] = df[cols_hair].sum(axis=1)

    # Score de piel
    if all(c in df.columns for c in ['clear_skin', 'oily_skin', 'dark_circles']):
        df['score_skin'] = df['clear_skin'] - df['oily_skin'] - df['dark_circles']

    # Score de envejecimiento
    cols_aging = [c for c in ['old', 'wrinkle', 'double_chin', 'receeding_hairline'] if c in df.columns]
    df['score_aging'] = df[cols_aging].sum(axis=1)

    # Score features femeninas
    cols_fem = [c for c in ['has_makeup', 'veil', 'long_hair', 'big_lips'] if c in df.columns]
    df['score_feminine'] = df[cols_fem].sum(axis=1)

    logger.info(f"Features compuestas creadas: score_facial_hair, score_hair_features, score_skin, score_aging, score_feminine")
    logger.info(f"Dataset con nuevas features: {df.shape}")

    return df


def select_features_and_split(
    df: pd.DataFrame,
    target_column: str,
    test_size: float,
    random_state: int,
    correlation_threshold: float
):
    """
    Nodo 2: Selecciona features relevantes y hace el split train/test.

    Proceso:
    1. Separa target (y) de features (X)
    2. Calcula correlación de cada feature con el target
    3. Descarta features con correlación absoluta < threshold
    4. Divide en train (80%) y test (20%)

    ¿Por qué seleccionar features?
    Quitar variables poco correlacionadas reduce ruido y mejora
    la velocidad y precisión de los modelos.

    Parameters
    ----------
    df : pd.DataFrame — dataset con features compuestas
    target_column : str — columna objetivo ('attractive')
    test_size : float — proporción para test (ej: 0.2)
    random_state : int — semilla para reproducibilidad
    correlation_threshold : float — mínimo de correlación aceptado

    Returns
    -------
    X_train, X_test, y_train, y_test como DataFrames
    """
    if target_column not in df.columns:
        raise ValueError(f"Target '{target_column}' no encontrado. Columnas: {list(df.columns)}")

    y = df[target_column]
    X = df.drop(columns=[target_column])

    # Calcular correlación con el target
    correlaciones = X.corrwith(y).abs()
    features_seleccionadas = correlaciones[correlaciones >= correlation_threshold].index.tolist()

    logger.info(f"Features totales: {X.shape[1]}")
    logger.info(f"Features seleccionadas (corr >= {correlation_threshold}): {len(features_seleccionadas)}")
    logger.info(f"Features descartadas: {X.shape[1] - len(features_seleccionadas)}")

    X = X[features_seleccionadas]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")

    # Kedro necesita DataFrames para guardarlos en el catalog
    return (
        X_train,
        X_test,
        pd.DataFrame(y_train),
        pd.DataFrame(y_test)
    )
