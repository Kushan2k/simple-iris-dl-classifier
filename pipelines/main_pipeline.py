
from zenml import pipeline


@pipeline(enable_cache=True)
def main_pipeline(data_set_path: str = None):
    """
    Main pipeline function that orchestrates the data loading and processing steps.

    Args:
        data_set_path (str): The path to the dataset to be loaded.
    """
    from steps.data_loader import data_loader
    

    # Load the data
    data = data_loader(data_set_path=data_set_path)

    return data
