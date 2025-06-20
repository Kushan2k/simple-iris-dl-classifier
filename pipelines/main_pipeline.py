
from zenml import pipeline


@pipeline(enable_cache=True)
def main_pipeline(data_set_path: str = None):
    """
    Main pipeline function that orchestrates the data loading and processing steps.

    Args:
        data_set_path (str): The path to the dataset to be loaded.
    """
    from steps.data_loader import data_loader
    from steps.data_visualizer import data_visualizer
    from steps.data_splitter import data_splitter

    # Load the data
    data = data_loader(data_set_path=data_set_path)
    data_visualizer(data=data, target_column='target')
    x_train,x_test,y_train,y_test=data_splitter(data, test_size=0.2, random_state=42)

    print(type(x_train), type(x_test), type(y_train), type(y_test))
    

    return data
