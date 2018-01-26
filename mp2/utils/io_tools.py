"""Input and output helpers to load in data."""

import csv
import numpy as np

def read_dataset(input_csv_file):
    """Read data into a python list.

    Args:
        input_csv_file: Path to the data csv file.

    Returns:
        dataset(dict): A python dictionary with the key value pair of
            (example_id, example_feature).

            example_feature is represented with a tuple
            (Id, BldgType, OverallQual, GrLivArea, GarageArea)

            For example, the first row will be in the train.csv is
            example_id = 1
            example_feature = (1,1Fam,7,1710,548)
    """
    dataset = {}

    # Imeplemntation here.
    with open(input_csv_file) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            dataset[row['Id']] = (int(row['Id']),
                                  row['BldgType'],
                                  int(row['OverallQual']),
                                  int(row['GrLivArea']),
                                  int(row['GarageArea']),
                                  int(row['SalePrice']))

    return dataset


def one_hot_bldg_type(bldg_type):
    """Builds the one-hot encoding vector.

    Args:
        bldg_type(str): String indicating the building type.

    Returns:
        ret(list): A list representing the one-hot encoding vector.
            (e.g. for 1Fam building type, the returned list should be
            [1,0,0,0,0].
    """
    type_to_id = {'1Fam': 0,
                  '2FmCon': 1,
                  'Duplx': 2,
                  'TwnhsE': 3,
                  'TwnhsI': 4,
                  }
    listofzeros = [0] * 5

    listofzeros[type_to_id[bldg_type]]=1
    ret = listofzeros
    pass
    return ret
a = np.zeros((10,5))
b = np.ones((a.shape[1],1))
print(a@b)
print(b)
#print(np.hstack((a,b)))
