from engine.dataset_helper.base import DATASET_CLASSES
from engine.dataset_helper.public import *

# from engine.dataset_helper.dummy_phase3 import *


def get_dataset(dataset, arch=None, is_encode=True, notebook_path=None):
    # Mapping dataset names to their corresponding classes

    if dataset not in DATASET_CLASSES:
        raise NotImplementedError(f"The dataset '{dataset}' is not implemented.")

    # Instantiate the dataset object
    d = globals()[DATASET_CLASSES[dataset]]
    try:
        D = d(is_encode=is_encode, notebook_path=notebook_path)
    except:
        D = d(notebook_path=notebook_path)

    # Special case handling
    if dataset == "intrusion" and arch == "copulagan":
        D._drop_duplicates()

    return D


def main():
    s = {}

    for k, v in s.items():
        print(k, v)
        d = globals()[DATASET_CLASSES[k]]()
        print(d.data)

        print(d.data_train)
        print(d.data_train.describe())
        print(d.data_test)
        print(d.features)
        print(d.discrete_columns)
        print(d.target)
        print(len(d.discrete_columns))
        print(len(d.continuous_columns))
        print(len(d.features))

        print(d.key_fields)
        print(d.sensitive_fields)


if __name__ == "__main__":
    main()
