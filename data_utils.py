import errno
import os
import zipfile

import pandas as pd


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_dataset(name: str) -> pd.DataFrame:
    """
    Load the dataset.

    :param name: the name of the dataset
    :return:
    """
    name = f"{name}.tsv"
    data_dir = os.path.join(ROOT_DIR, "data")
    dataset_path = os.path.join(data_dir, name)
    all_data = os.listdir(data_dir)
    if name in all_data:
        # If the data is already unzipped.
        pass
    elif f"{name}.zip" in all_data:
        # Unzip the dataset.
        with zipfile.ZipFile(f"{dataset_path}.zip", 'r') as zip_ref:
            zip_ref.extractall(data_dir)
    else:
        raise FileNotFoundError(
            errno.ENOENT, f"Nether .tsv nor .tsv.zip version of the dataset weren't found in the directory {data_dir}", name)
    # Load dataset.
    data = pd.read_csv(dataset_path, sep='\t', header=None, error_bad_lines=False)
    # Remove NaNs.
    data = data.dropna()
    # Shuffle data.
    data = data.sample(frac=1).reset_index(drop=True)
    return data
