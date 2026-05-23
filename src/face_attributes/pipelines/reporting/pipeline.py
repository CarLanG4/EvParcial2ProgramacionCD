from kedro.pipeline import Pipeline, node, pipeline
from .nodes import generate_final_report


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=generate_final_report,
            inputs=["classification_metrics", "clustering_results"],
            outputs="final_report",
            name="node_final_report",
        ),
    ])
