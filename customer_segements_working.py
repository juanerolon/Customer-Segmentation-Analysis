

import numpy as np
import pandas as pd
import scipy as sp

from scipy.stats import normaltest


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
from IPython.display import display # Allows the use of display() for DataFrames



# Load the wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

print "Data column names:\n"
print list(data.columns)
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

print ""
print "Experimenting: \n"

cols = list(data.columns)
#print data[cols].values[:, 0]
#print data['Fresh'].values

for feat in cols:
    print normaltest(data[feat].values)




log_data = np.log(data)
log_samples = np.log(samples)

# For each feature find the data points with extreme high or low values
for feature in log_data.keys():
    Q1 = np.percentile(log_data[feature], 25)
    Q3 = np.percentile(log_data[feature], 75)
    step = 1.5 * (Q3 - Q1)

    # Display the outliers
    print "Data points considered outliers for the feature '{}':".format(feature)
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])

# OPTIONAL: Select the indices for data points you wish to remove
outliers = []

# Remove the outliers, if any were specified
# good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)











#--------------------------- OTHER PLOTS --------------------------

#Normality tests:
if False:

    print ""
    print "Normality tests\n:"

    cols = list(data.columns)
    value, p = normaltest(data[cols].values[:, 0])

    print(value, p)

    if p >= 0.05:
        print('It is likely that result1 is normal')
    else:
        print('It is unlikely that result1 is normal')


#Create histograms:
if False:

    data.hist()
    plt.ylim(40)
    plt.tight_layout()
    plt._show()

#Create box plots
if False:
    #plt.ylim(40000)
    fdat = data.drop('Frozen', axis=1)
    fdat = fdat.drop('Delicatessen', axis=1)

    fdat.boxplot()
    #data.boxplot()
    plt.tight_layout()
    plt._show()
    plt.grid(None)

#Create scatter matrix
if False:
    cols = list(data.columns)
    sns.pairplot(data[cols], size=2.0)
    plt.show()

#Create correlation matrix heat map
if False:
    cols = list(data.columns)
    corr_matrix = np.corrcoef(data[cols].values.T)
    sns.set(font_scale=1.5)

    heat_map = sns.heatmap(corr_matrix, cbar=True, annot=True, square=True, fmt='.2f',
               annot_kws = {'size': 15}, yticklabels=cols, xticklabels=cols)

    plt.xticks(rotation='vertical')
    plt.yticks(rotation='horizontal')

    plt.tight_layout()
    plt.show()

