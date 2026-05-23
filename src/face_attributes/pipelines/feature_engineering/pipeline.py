from kedro.pipeline import Pipeline, node, pipeline
from .nodes import create_composite_features, select_features_and_split


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_composite_features,
            inputs="clean_attributes",
            outputs="features_attributes",      # guardado en data/03_primary/
            name="node_create_features",
        ),
        node(
            func=select_features_and_split,
            inputs=[
                "features_attributes",
                "params:target_column",
                "params:test_size",
                "params:random_state",
                "params:correlation_threshold",
            ],
            outputs=["X_train", "X_test", "y_train", "y_test"],  # 4 datasets
            name="node_select_split",
        ),
    ])
