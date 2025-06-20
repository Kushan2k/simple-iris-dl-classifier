from typing import  Tuple
from typing_extensions import Annotated
from sklearn.model_selection import train_test_split
from zenml import step
import pandas as pd
from pandas import DataFrame, Series


@step
def data_splitter(
        data: pd.DataFrame,
        test_size: float = 0.2, 
        random_state: int = 42) -> Tuple[
            Annotated[pd.DataFrame,'x_train'],
            Annotated[pd.DataFrame,'x_test'],
            Annotated[pd.Series,'y_train'],
            Annotated[pd.Series,'y_test']
            
        ]:
    """
    Splits the dataset into training and testing sets.

    Args:
        data (pd.DataFrame): The dataset to be split.
        test_size (float): The proportion of the dataset to include in the test split.
        random_state (int): Controls the shuffling applied to the data before applying the split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
            - x_train: Training features.
            - x_test: Testing features.
            - y_train: Training labels.
            - y_test: Testing labels.

    
    """
    x_train: pd.DataFrame
    x_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series

    x_train, x_test, y_train, y_test = train_test_split(
        data.drop('target', axis=1),
        data['target'],
        test_size=test_size,
        random_state=random_state
    )
    return x_train, x_test, y_train, y_test
