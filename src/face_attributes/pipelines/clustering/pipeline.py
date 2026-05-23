from kedro.pipeline import Pipeline, node, pipeline
from .nodes import run_kmeans_clustering


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=run_kmeans_clustering,
            inputs=[
                "features_attributes",
                "params:n_clusters",
                "params:random_state",
            ],
            outputs="clustering_results",    # guardado en data/07_model_output/
            name="node_kmeans_clustering",
        ),
    ])
