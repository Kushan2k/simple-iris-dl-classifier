from zenml import step
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns



@step
def data_visualizer(data: pd.DataFrame, target_column: str = 'target') -> None:
    """
    Visualizes the data using seaborn pairplot.

    Args:
        data (pd.DataFrame): The dataset to visualize.
        target_column (str): The name of the target column for coloring the plots.
    """
    sns.pairplot(data, hue=target_column)
    plt.show()
