

def main():
    """
    Main function to run the pipeline.
    """
    from pipelines.main_pipeline import main_pipeline

    # Define the dataset path
    data_set_path = "data/dataset.csv"

    # Run the main pipeline
    main_pipeline(data_set_path=data_set_path)

if __name__ == "__main__":
    main()
