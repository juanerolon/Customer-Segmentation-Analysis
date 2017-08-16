

import numpy as np
import pandas as pd
from IPython.display import display # Allows the use of display() for DataFrames


# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
print "Dataset preview:\n"
print data.head(5)
print ""
print "Dataset stats preview:\n"

# Display a description of the dataset
display(data.describe())

if False:
    npranlist =  np.random.randint(0, high=439, size=(1,3), dtype='l')
    rindexes =  list(npranlist[0])
    print rindexes
    print type(rindexes)

# TODO: Select three indices of your choice you wish to sample from the dataset
indices = [309, 216, 160]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)

