"""
PIPELINE 5 (BONUS) — REPORTING
================================
Nodo que consolida los resultados de clasificación y clustering
en un reporte final de resumen ejecutivo.
"""

import pandas as pd
import logging

logger = logging.getLogger(__name__)


def generate_final_report(
    classification_metrics: pd.DataFrame,
    clustering_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Nodo 1: Genera un reporte ejecutivo consolidado.

    Combina:
    - Mejor modelo de clasificación y sus métricas
    - Resumen de clusters encontrados

    Parameters
    ----------
    classification_metrics : resultados de Pipeline 3
    clustering_results : resultados de Pipeline 4

    Returns
    -------
    pd.DataFrame — reporte final
    """
    # Mejor modelo de clasificación
    mejor = classification_metrics.sort_values("f1_score", ascending=False).iloc[0]

    # Resumen de clustering
    n_clusters = len(clustering_results)
    sil = clustering_results['silhouette_score'].iloc[0] if 'silhouette_score' in clustering_results.columns else "N/A"

    reporte = pd.DataFrame([
        {"seccion": "CLASIFICACIÓN", "metrica": "Mejor Modelo", "valor": mejor['modelo']},
        {"seccion": "CLASIFICACIÓN", "metrica": "Accuracy", "valor": round(mejor['accuracy'], 4)},
        {"seccion": "CLASIFICACIÓN", "metrica": "F1 Score", "valor": round(mejor['f1_score'], 4)},
        {"seccion": "CLASIFICACIÓN", "metrica": "Precision", "valor": round(mejor['precision'], 4)},
        {"seccion": "CLASIFICACIÓN", "metrica": "Recall", "valor": round(mejor['recall'], 4)},
        {"seccion": "CLUSTERING", "metrica": "N Clusters", "valor": n_clusters},
        {"seccion": "CLUSTERING", "metrica": "Silhouette Score", "valor": sil},
    ])

    logger.info("=== REPORTE FINAL ===")
    for _, row in reporte.iterrows():
        logger.info(f"  [{row['seccion']}] {row['metrica']}: {row['valor']}")

    return reporte
