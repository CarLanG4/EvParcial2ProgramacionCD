"""
PIPELINE 4 — CLUSTERING
=========================
Nodos que agrupan las caras en perfiles usando K-Means (aprendizaje NO supervisado).

¿Qué hace este pipeline?
- No usa etiquetas (no usa 'attractive')
- Agrupa las caras en N grupos según sus atributos similares
- Calcula métricas de calidad del clustering
- Describe cada grupo con sus atributos más representativos

¿Por qué es útil?
Descubrir perfiles naturales en el dataset (ej: "cara masculina madura",
"cara femenina joven con maquillaje", "cara andrógina") sin haberlos
definido previamente.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
import logging

logger = logging.getLogger(__name__)


def run_kmeans_clustering(
    df: pd.DataFrame,
    n_clusters: int,
    random_state: int
) -> pd.DataFrame:
    """
    Nodo 1: Aplica K-Means y describe los clusters resultantes.

    Proceso paso a paso:
    1. Eliminar el target ('attractive') para que el clustering
       sea puramente no supervisado
    2. Escalar las features (StandardScaler) — muy importante en K-Means
       porque es sensible a la escala de los datos
    3. Aplicar K-Means con n_clusters grupos
    4. Calcular métricas de calidad:
       - Silhouette Score: qué tan bien separados están los clusters
         (0 = malo, 1 = perfecto)
       - Davies-Bouldin Index: qué tan compactos y separados son
         (menor = mejor)
    5. Describir cada cluster con su media de atributos

    Parameters
    ----------
    df : pd.DataFrame — features_attributes (con features compuestas)
    n_clusters : int — número de grupos a crear
    random_state : int — semilla

    Returns
    -------
    pd.DataFrame con descripción de cada cluster
    """
    # Quitar columna target si existe
    cols_excluir = ['attractive', 'image_id']
    X = df.drop(columns=[c for c in cols_excluir if c in df.columns])

    # Escalar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    logger.info(f"Entrenando K-Means con {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # Métricas de calidad
    sil_score = silhouette_score(X_scaled, labels)
    db_score = davies_bouldin_score(X_scaled, labels)
    logger.info(f"Silhouette Score: {sil_score:.4f} (mayor es mejor, max=1)")
    logger.info(f"Davies-Bouldin Index: {db_score:.4f} (menor es mejor)")

    # Añadir cluster al DataFrame original
    df_con_cluster = df.copy()
    df_con_cluster['cluster'] = labels

    # Describir cada cluster: promedio de cada atributo
    cluster_profiles = df_con_cluster.groupby('cluster').mean().round(3)
    cluster_profiles['cluster_size'] = df_con_cluster.groupby('cluster').size()
    cluster_profiles['silhouette_score'] = round(sil_score, 4)
    cluster_profiles['davies_bouldin_index'] = round(db_score, 4)
    cluster_profiles = cluster_profiles.reset_index()

    # Log descripción de cada cluster
    for c in range(n_clusters):
        perfil = cluster_profiles[cluster_profiles['cluster'] == c].iloc[0]
        size = int(perfil['cluster_size'])
        pct = round(size / len(df) * 100, 1)

        # Top atributos del cluster (mayor promedio)
        atributos_top = (
            cluster_profiles[cluster_profiles['cluster'] == c]
            .iloc[0]
            .drop(['cluster', 'cluster_size', 'silhouette_score', 'davies_bouldin_index'])
            .sort_values(ascending=False)
            .head(5)
            .index.tolist()
        )
        logger.info(f"Cluster {c}: {size} personas ({pct}%) | Top atributos: {atributos_top}")

    return cluster_profiles
