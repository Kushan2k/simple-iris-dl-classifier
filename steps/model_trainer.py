from abc import ABC, abstractmethod
from zenml import step
from tensorflow.keras import Sequential,layers
import tensorflow as tf
import pandas as pd



class Model(ABC):


  @abstractmethod
  def train(self, x_train, y_train):
    """
    Train the model on the training data.
    
    Args:
        x_train: Training features.
        y_train: Training labels.
    """
    pass


class SequentialModel(Model):
    """
    A concrete implementation of the Model class using a Sequential Keras model.
    """
      

    def train(self, x_train, y_train,epochs=10, batch_size=32, input_shape=None, num_classes=None)->tf.keras.Model:
        

        try:
          model:tf.keras.Model = Sequential([
            layers.Input(shape=input_shape),
            layers.Dense(64, activation='relu'),
            layers.Dense(64, activation='relu'),
            layers.Dense(num_classes, activation='softmax')
          ])
          model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
          model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)

          return model
        except Exception as e:
            print(f"An error occurred while training the model: {e}")
            raise ValueError("Model training failed. Please check the input data and model configuration.")


@step
def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> tf.keras.Model:
    """
    Train the model on the training data.

    Args:
        x_train (tf.Tensor): Training features.
        y_train (tf.Tensor): Training labels.
        

    Returns:
        Model: The trained model.
    """


    model=SequentialModel().train(
        x_train=x_train,
        y_train=y_train,
        epochs=10,
        batch_size=32,
        input_shape=(x_train.shape[1],),
        num_classes=len(y_train.unique())
    )
    

    return model
    # tensor=x_train.__tf_tensor__()

    # print('shape ',tensor.shape)
