"""
Define el grafo del Pipeline 1: Data Processing.

Un pipeline en Kedro es un grafo de nodos (node).
Cada node define:
  - func: la función Python a ejecutar
  - inputs: qué datasets/parámetros recibe
  - outputs: qué dataset produce
  - name: nombre descriptivo del nodo

Kedro conecta automáticamente los nodos según inputs/outputs.
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import load_and_validate, generate_statistics


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=load_and_validate,
            inputs=["raw_attributes", "params:features_to_drop"],
            outputs="validated_df",          # dataset intermedio en memoria
            name="node_load_validate",
        ),
        node(
            func=generate_statistics,
            inputs="validated_df",
            outputs="clean_attributes",      # se guarda en data/02_intermediate/
            name="node_generate_statistics",
        ),
    ])
