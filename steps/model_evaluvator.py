from zenml import step
import tensorflow as tf
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


@step
def evaluate_model(model: tf.keras.Model, x_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    Evaluate the trained model on the test data.

    Args:
        model (tf.keras.Model): The trained Keras model.
        x_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.

    Returns:
        dict: A dictionary containing evaluation metrics.
    """
    try:
        # Make predictions
        y_pred = model.evaluate(x_test,y_test)
        y_pred_classes = tf.argmax(y_pred, axis=1).numpy()

        # Calculate evaluation metrics
        accuracy = accuracy_score(y_test, y_pred_classes)
        precision = precision_score(y_test, y_pred_classes, average='weighted')
        recall = recall_score(y_test, y_pred_classes, average='weighted')
        f1 = f1_score(y_test, y_pred_classes, average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
    except Exception as e:
        print(f"An error occurred during model evaluation: {e}")
        raise ValueError("Model evaluation failed. Please check the input data and model configuration.")
