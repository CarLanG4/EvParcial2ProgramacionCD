from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_classifiers, evaluate_classifiers


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_classifiers,
            inputs=[
                "X_train", "y_train",
                "params:n_estimators",
                "params:max_depth",
                "params:random_state",
            ],
            outputs="trained_models",     # dict en memoria (no se persiste)
            name="node_train_classifiers",
        ),
        node(
            func=evaluate_classifiers,
            inputs=[
                "trained_models",
                "X_test", "y_test",
                "params:cv_folds",
            ],
            outputs="classification_metrics",   # guardado en data/07_model_output/
            name="node_evaluate_classifiers",
        ),
    ])
