

import numpy as np
import pandas as pd
import scipy as sp

from scipy.stats import normaltest


import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid', context='notebook')
from IPython.display import display # Allows the use of display() for DataFrames


def pca_results(good_data, pca):
    '''
    Create a DataFrame of the PCA results
    Includes dimension feature weights and explained variance
    '''

    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i) for i in range(1, len(pca.components_) + 1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4), columns=good_data.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4), columns=['Explained Variance'])
    variance_ratios.index = dimensions

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis=1)



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



#Feature scaling
log_data = np.log(data)
log_samples = np.log(samples)





#Outliers

if False:

    print '\n"Frozen" outliers\n'

    feature = 'Frozen'

    Q1 = np.percentile(log_data[feature], 25)
    Q3 = np.percentile(log_data[feature], 75)
    step = 1.5 * (Q3 - Q1)

    print 'Q1,Q3,step = ', Q1, Q3, step

    outlier_data = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]

    print outlier_data.head(5)

    print '\n Outlier data indexes:\n'

    f_indexes = outlier_data.index.values

    print f_indexes




if True:

    # For each feature find the data points with extreme high or low values

    outliers = []
    print outliers
    for feature in log_data.keys():
        Q1 = np.percentile(log_data[feature], 25)
        Q3 = np.percentile(log_data[feature], 75)
        step = 1.5 * (Q3 - Q1)

        # Display the outliers
        print "Data points considered outliers for the feature '{}':".format(feature)
        feat_outliers = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
        outliers += list(feat_outliers.index.values)
        #print outliers
        #display(feat_outliers)

    # OPTIONAL: Select the indices for data points you wish to remove
    import collections
    common_outliers = [item for item, count in collections.Counter(outliers).items() if count > 1]
    outliers = list(np.unique(np.asarray(outliers)))
    #print outliers


    # Remove the outliers, if any were specified
    good_data = log_data.drop(log_data.index[outliers]).reset_index(drop = True)

    """
    sns.reset_orig()
    plt.figure(1, figsize=(10, 9))
    good_data.boxplot(showfliers=True)
    plt.ylim(0,15)
    plt.tight_layout()
    plt.show()

    """

    print "Number of dimensions in original data set: {} \n".format(data.shape[1])
    print ""

    from sklearn.decomposition import PCA
    # TODO: Apply PCA by fitting the good data with the same number of dimensions as features
    pca = PCA(n_components=data.shape[1])
    pca.fit(good_data)

    # TODO: Transform log_samples using the PCA fit above
    pca_samples = pca.transform(log_samples)

    # Generate PCA results plot
    pca_results = pca_results(good_data, pca)

    print pca_results['Explained Variance'][0] +pca_results['Explained Variance'][1]

    print pca_results['Explained Variance'].cumsum()















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

