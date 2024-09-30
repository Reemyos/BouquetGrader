import glob
from pathlib import Path

import pandas as pd


def preprocess_data(path_to_images_and_labels: Path = Path("bouquets")) -> pd.DataFrame:
    # Create a DataFrame with data from the images and labels (like how many flowers are in each image, etc.)
    df = pd.DataFrame()
    df['Image Path'] = list((path_to_images_and_labels / "all_images").glob("*.png"))

    labels = {0: 'Pink', 1: 'White', 2: 'Red', 3: 'Orange', 4: 'Yellow'}

    # Add columns for each label
    for label in labels.values():
        df[label] = [0] * len(df['Image Path'])

    # Add the number of flowers of each label in each image
    for path_to_image in df['Image Path']:
        image_name = path_to_image.name.replace('.png', '')
        labels_path = path_to_images_and_labels / 'all_labels' / f'{image_name}.txt'
        try:
            with open(labels_path, 'r') as f:
                for line in f.readlines():
                    flower_label = int(line.split()[0])
                    df.loc[df['Image Path'] == path_to_image, labels[flower_label]] += 1
        except FileNotFoundError:
            print(f'No label file found for {image_name}')

    # Add a column for the total number of flowers in each image
    df['Total Flowers'] = df[labels.values()].sum(axis=1)

    # Drop images with no flowers
    df = df[df['Total Flowers'] > 0]

    return df


if __name__ == '__main__':
    path_to_images_and_labels = Path('bouquets')
    df = preprocess_data(path_to_images_and_labels)
    df.to_csv('labels.csv', index=False)
    print(df.head())
