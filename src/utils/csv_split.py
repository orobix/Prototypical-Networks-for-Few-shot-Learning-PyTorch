import pandas as pd
import numpy as np


csv_path = './mini_imagenet/csvsplits/all.csv'
separator = ';'
csv_data = pd.read_csv(csv_path, sep=separator, header=None, squeeze=True)

# Randomly shuffle all the csv indices for the class names
indices = np.arange(0, 100)
np.random.shuffle(indices)

# Get all the class names
class_names = csv_data.values[0]

# Split class names into train, validation and test datasets
tr_data = class_names[indices[:64]]
va_data = class_names[indices[64:80]]
te_data = class_names[indices[80:100]]

# Create dataframes for the new datasets
# In order to convert datasets to .csv files
tr_df = pd.DataFrame(data=tr_data)
va_df = pd.DataFrame(data=va_data)
te_df = pd.DataFrame(data=te_data)

# Write the three new .csv files: Train, validation and test
tr_df.to_csv('./mini_imagenet/csvsplits/train.csv', sep=separator, index=False, header=None)
va_df.to_csv('./mini_imagenet/csvsplits/valid.csv', sep=separator, index=False, header=None)
te_df.to_csv('./mini_imagenet/csvsplits/test.csv',  sep=separator, index=False, header=None)