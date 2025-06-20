from typing import  Tuple
from typing_extensions import Annotated
from sklearn.model_selection import train_test_split
from zenml import step
import pandas as pd


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
        tuple[pd.DataFrame, pd.DataFrame,pd.Series,pd.Series]: Training and testing datasets.
    """
    x_train,x_test,y_train,y_test=train_test_split(data, test_size=test_size, random_state=random_state)
    return x_train, x_test, y_train, y_test
