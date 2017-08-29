
import numpy as np
import pandas as pd
import scipy as sp

from scipy.stats import normaltest

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

sns.set(style='whitegrid', context='notebook')

from IPython.display import display # Allows the use of display() for DataFrames

"""
Auxiliary methods: clustering, pca:
"""

def cluster_results(reduced_data, preds, centers, pca_samples):
	'''
	Visualizes the PCA-reduced cluster data in two dimensions
	Adds cues for cluster centers and student-selected sample data
	'''

	predictions = pd.DataFrame(preds, columns = ['Cluster'])
	plot_data = pd.concat([predictions, reduced_data], axis = 1)

	# Generate the cluster plot
	fig, ax = plt.subplots(figsize = (14,8))

	# Color map
	cmap = cm.get_cmap('gist_rainbow')

	# Color the points based on assigned cluster
	for i, cluster in plot_data.groupby('Cluster'):
	    cluster.plot(ax = ax, kind = 'scatter', x = 'Dimension 1', y = 'Dimension 2', \
	                 color = cmap((i)*1.0/(len(centers)-1)), label = 'Cluster %i'%(i), s=30);

	# Plot centers with indicators
	for i, c in enumerate(centers):
	    ax.scatter(x = c[0], y = c[1], color = 'white', edgecolors = 'black', \
	               alpha = 1, linewidth = 2, marker = 'o', s=200);
	    ax.scatter(x = c[0], y = c[1], marker='$%d$'%(i), alpha = 1, s=100);

	# Plot transformed sample points
	ax.scatter(x = pca_samples[:,0], y = pca_samples[:,1], \
	           s = 150, linewidth = 4, color = 'black', marker = 'x');

	# Set plot title
	ax.set_title("Cluster Learning on PCA-Reduced Data - Centroids Marked by Number\nTransformed Sample Data Marked by Black Cross");



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

# Load the full wholesale customers dataset
try:
    data = pd.read_csv("customers.csv")
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"

print "Data column names:\n"
cols = list(data.columns)
print cols
print "Dataset preview:\n"
print data.head(5)
print ""
print "Dataset stats preview:\n"
display(data.describe())

# Select three indices of your choice you wish to sample from the dataset
indices = [309, 216, 22]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)

print "Samples of wholesale customers dataset:"
display(samples)

#Feature scaling
log_data = np.log(data)
log_samples = np.log(samples)

#Remove outliers
import collections
outliers = []
for feature in log_data.keys():
    #Calculate Q1 (25th percentile of the data) for the given feature
    Q1 = np.percentile(log_data[feature], 25)
    #Calculate Q3 (75th percentile of the data) for the given feature
    Q3 = np.percentile(log_data[feature], 75)
    # Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
    step = 1.5 * (Q3 - Q1)
    # Display the outliers
    #print "Data points considered outliers for the feature '{}':".format(feature)
    feat_outliers = log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))]
    outliers += list(feat_outliers.index.values)
    #display(feat_outliers)

# Select the indices for data points you wish to remove
common_outliers = [item for item, count in collections.Counter(outliers).items() if count > 1]

print 'Outlier data idexes common to features: {}'.format(common_outliers)

# Remove the outliers
outliers = list(np.unique(np.asarray(outliers)))
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop=True)


# Apply PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=data.shape[1])
pca.fit(good_data)

# Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)
pca_results = pca_results(good_data, pca)


print "Cumulative sum of of explained variance by dimension:"
print pca_results['Explained Variance'].cumsum()
print""
print "PCA detailed results:"
print pca_results

pca = PCA(n_components=2)
#pca = PCA(n_components=3)
pca.fit(good_data)

# Transform log_samples using the PCA fit above
reduced_data = pca.transform(good_data)
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])
#reduced_data = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2', 'Dimension 3'])

print "\n"
print "Sample log-data after applying PCA transformation in two dimensions:\n"
# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns=['Dimension 1', 'Dimension 2']))
#display(pd.DataFrame(np.round(pca_samples, 4), columns=['Dimension 1', 'Dimension 2','Dimension 3']))


"""
Clustering methods testing
"""

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

if False:

    #Analyse K-means trials on different cluster numbers and the corresponding silhoutte score

    print "\n"
    print "K-Means Silhouette Scoring Tests:\n"

    for kn in range(2, 9):

        #Apply your clustering algorithm of choice to the reduced data
        clm = KMeans(n_clusters=kn, random_state=0)
        clm.fit(reduced_data)

        #Predict the cluster for each data point
        preds = clm.predict(reduced_data)

        #Find the cluster centers
        centers = clm.cluster_centers_

        #Predict the cluster for each transformed sample data point
        sample_preds = clm.predict(pca_samples)

        #Calculate the mean silhouette coefficient for the number of clusters chosen
        score = silhouette_score(reduced_data, preds, random_state=10)
        print "Number of clusters = {}, Score = {}".format(kn, np.round(score,4))


if False:

    print "\n"
    print "Gaussian Mixtures Silhouette Scoring Tests:\n"

    for kn in range(2, 9):
        clusterer = GaussianMixture(n_components=kn, random_state=10)
        clusterer.fit(reduced_data)
        preds = clusterer.predict(reduced_data)
        centers = clusterer.means_
        sample_preds = clusterer.predict(pca_samples)
        score = silhouette_score(reduced_data, preds, random_state=20)
        print("Number of clusters = {}, Score = {}".format(kn, np.round(score,4)))



clm = KMeans(n_clusters=2, random_state=0)
clm.fit(reduced_data)
preds = clm.predict(reduced_data)
centers = clm.cluster_centers_
sample_preds = clm.predict(pca_samples)
score = silhouette_score(reduced_data, preds, random_state=10)
print "Number of clusters = {}, Score = {}".format(2, np.round(score, 4))

print "\n"
print "Inverse transform of cluster centers data points\n"


#Inverse transform the centers
log_centers = pca.inverse_transform(centers)

#Exponentiate the centers
true_centers = np.exp(log_centers)

# Display the true centers
segments = ['Segment {}'.format(i) for i in range(0,len(centers))]
true_centers = pd.DataFrame(np.round(true_centers), columns = data.keys())
true_centers.index = segments
display(true_centers)

print "\n"
print "Predict to which clusters the sample points belong:\n"

# Display the predictions
for i, pred in enumerate(sample_preds):
    print "Sample point", i, "predicted to be in Cluster", pred


cluster_results(reduced_data, preds, centers, pca_samples)
plt.show()
