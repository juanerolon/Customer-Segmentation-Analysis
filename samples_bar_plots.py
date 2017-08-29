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
cols = list(data.columns)
print cols

print ""
print "Descriptive statistics:\n"
display(data.describe())

#select sample indexes
indices = [309, 216, 22]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)

print ""
print "Experimenting: \n"

print "Descriptive statistics:\n"

stats = data.describe()
print stats

print "Samples:\n"
print samples

print ""

print cols

ratios = []
for index in range(len(samples)):
    rlist = []
    for feat in cols:
        rvalue = float(samples[feat][index])/float(stats[feat]['mean'])
        rlist.append(rvalue)
    ratios.append(rlist)
    print rlist


print "\n"


if False:

    groups = cols
    n_groups = len(groups)
    ind = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8

    plt.figure(1, figsize=(15, 5))

    nrws = 1
    ncol = 3

    for m in range(len(ratios)):

        plt.subplot(nrws, ncol, m+1)

        plt.bar(ind, ratios[m], bar_width, alpha=opacity, color='b', label=None)
        plt.xlabel('Product Categories')
        plt.ylabel('Annual Spending Ratios')
        plt.title('Customer {}'.format(m))
        plt.xticks(ind, groups, rotation='vertical')
        plt.legend(frameon=False, loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.show()

