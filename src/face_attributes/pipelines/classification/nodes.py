"""
PIPELINE 3 — CLASSIFICATION
=============================
Nodos que entrenan y evalúan modelos de clasificación binaria.

Problema: Predecir si una cara es "attractive" (1) o no (0)
basándose en sus atributos faciales.

¿Qué hace este pipeline?
- Entrena 3 modelos: Logistic Regression, Decision Tree, Random Forest
- Evalúa cada uno con métricas: accuracy, precision, recall, F1
- Realiza validación cruzada (5 folds) para mayor robustez
- Guarda las métricas comparativas en un CSV
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import logging

logger = logging.getLogger(__name__)


def train_classifiers(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    n_estimators: int,
    max_depth: int,
    random_state: int
) -> dict:
    """
    Nodo 1: Entrena 3 modelos de clasificación.

    Modelos entrenados:
    1. Logistic Regression — modelo lineal, rápido, interpretable.
       Útil como baseline para comparar contra modelos más complejos.

    2. Decision Tree — árbol de decisión. Muy interpretable:
       se puede visualizar cómo toma decisiones el modelo.
       Riesgo de overfitting si es muy profundo.

    3. Random Forest — conjunto (ensemble) de 100 árboles.
       Más robusto que un solo árbol, mejor generalización.

    Cada modelo usa StandardScaler para normalizar las features,
    lo cual es importante para que la escala no afecte al modelo.

    Parameters
    ----------
    X_train, y_train : DataFrames de entrenamiento
    n_estimators : número de árboles en Random Forest
    max_depth : profundidad máxima del árbol
    random_state : semilla de aleatoriedad

    Returns
    -------
    dict con los 3 modelos entrenados
    """
    y = y_train.iloc[:, 0]  # Convertir DataFrame a Series

    modelos = {
        "Logistic Regression": SklearnPipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=random_state))
        ]),
        "Decision Tree": SklearnPipeline([
            ("scaler", StandardScaler()),
            ("clf", DecisionTreeClassifier(max_depth=max_depth, random_state=random_state))
        ]),
        "Random Forest": SklearnPipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=random_state,
                n_jobs=-1
            ))
        ]),
    }

    entrenados = {}
    for nombre, modelo in modelos.items():
        logger.info(f"Entrenando: {nombre}...")
        modelo.fit(X_train, y)
        entrenados[nombre] = modelo
        logger.info(f"  [OK] {nombre} entrenado")

    return entrenados


def evaluate_classifiers(
    trained_models: dict,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
    cv_folds: int
) -> pd.DataFrame:
    """
    Nodo 2: Evalúa todos los modelos y genera una tabla comparativa.

    Métricas calculadas:
    - Accuracy: % de predicciones correctas (fácil de entender)
    - Precision: de los que predije "attractive", ¿cuántos realmente lo son?
    - Recall: de los realmente "attractive", ¿cuántos detecté?
    - F1: balance entre precision y recall (útil con clases desbalanceadas)
    - CV Score: accuracy promedio en validación cruzada (más confiable)

    La validación cruzada (cv_folds=5) divide el train en 5 partes,
    entrena en 4 y valida en 1, rotando. Así se evita el azar del split.

    Parameters
    ----------
    trained_models : dict con modelos entrenados
    X_test, y_test : datos de evaluación
    cv_folds : número de folds para cross-validation

    Returns
    -------
    pd.DataFrame con métricas por modelo (se guarda en data/07_model_output/)
    """
    y = y_test.iloc[:, 0]
    resultados = []

    for nombre, modelo in trained_models.items():
        y_pred = modelo.predict(X_test)

        # Validación cruzada sobre X_test para robustez
        cv_scores = cross_val_score(modelo, X_test, y, cv=min(cv_folds, 5), scoring='accuracy')

        metricas = {
            "modelo": nombre,
            "accuracy":   round(accuracy_score(y, y_pred), 4),
            "precision":  round(precision_score(y, y_pred, zero_division=0), 4),
            "recall":     round(recall_score(y, y_pred, zero_division=0), 4),
            "f1_score":   round(f1_score(y, y_pred, zero_division=0), 4),
            "cv_accuracy_mean": round(cv_scores.mean(), 4),
            "cv_accuracy_std":  round(cv_scores.std(), 4),
        }
        resultados.append(metricas)
        logger.info(f"{nombre}: accuracy={metricas['accuracy']}, F1={metricas['f1_score']}")

    df_metricas = pd.DataFrame(resultados).sort_values("f1_score", ascending=False)
    logger.info(f"\nMejor modelo: {df_metricas.iloc[0]['modelo']} (F1={df_metricas.iloc[0]['f1_score']})")

    return df_metricas
