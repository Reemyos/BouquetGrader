import pandas as pd

from ex1.uni_class_learner import learn_uni_class_model


def fetch_data():
    # Load the data from the CSV file with the correct column data types
    features = pd.read_csv('labels.csv', dtype={'Image Path': 'string', 'Pink': 'float', 'White': 'float', 'Red': 'float',
                                                'Orange': 'float', 'Yellow': 'float', 'Total Flowers': 'float'})
    features = features.drop('Image Path', axis=1)
    features = features.to_numpy()

    # Normalize the data row by row
    features /= features.sum(axis=1).reshape(-1, 1)

    return features


def learn_flowers_uni_class():
    features = fetch_data()
    return learn_uni_class_model(features.copy())


