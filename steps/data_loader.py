import pandas as pd
from zenml import step
from sklearn.datasets import load_iris


@step
def data_loader(data_set_path:str=None)->pd.DataFrame:
    """
    This function is a placeholder for loading data.
    It currently does not perform any operations.

    args:
        data_set_path (str): The path to the dataset to be loaded.

    returns:
        pd.DataFrame: An empty DataFrame as a placeholder.
    """
    
    if data_set_path !=None:
        data,target=load_iris(return_X_y=True,as_frame=True)
        data['target'] = target
        
        return data
    else:
        raise ValueError("data_set_path cannot be None")
