

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

#Data column names
cols = list(data.columns)

# Display a description of the dataset
display(data.describe())

if False:
    npranlist =  np.random.randint(0, high=439, size=(1,3), dtype='l')
    rindexes =  list(npranlist[0])
    print rindexes
    print type(rindexes)

# Select three indices of your choice you wish to sample from the dataset
indices = [309, 216, 22]

# Create a DataFrame of the chosen samples
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)

print "Samples of wholesale customers dataset:"
display(samples)

#Feature scaling
log_data = np.log(data)
log_samples = np.log(samples)

print "Transformed samples of wholesale customers dataset:"
# Display the log-transformed sample data
display(log_samples)

import collections

# For each feature find the data points with extreme high or low values
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

# OPTIONAL: Select the indices for data points you wish to remove
common_outliers = [item for item, count in collections.Counter(outliers).items() if count > 1]

print "\n"
print 'Outlier data idexes common to features: {}'.format(common_outliers)

# Remove the outliers, if any were specified
outliers = list(np.unique(np.asarray(outliers)))
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop=True)



from sklearn.decomposition import PCA

# Apply PCA by fitting the good data with the same number of dimensions as features
pca = PCA(n_components=data.shape[1])
pca.fit(good_data)

# Transform log_samples using the PCA fit above
pca_samples = pca.transform(log_samples)
pca_results = pca_results(good_data, pca)

print "\n"
print "Principal Components Analysis Results:\n"
print pca_results

pca = PCA(n_components=2)
pca.fit(good_data)
reduced_data = pca.transform(good_data)
pca_samples = pca.transform(log_samples)

# Create a DataFrame for the reduced data
reduced_data = pd.DataFrame(reduced_data, columns=['Dimension 1', 'Dimension 2'])


print "\n"
print "Sample log-data after applying PCA transformation in two dimensions:\n"
# Display sample log-data after applying PCA transformation in two dimensions
display(pd.DataFrame(np.round(pca_samples, 4), columns=['Dimension 1', 'Dimension 2']))

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

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

print "\n"
print "Apply K-means procedure to Fit data using two clusters:\n"

clm = KMeans(n_clusters=2, random_state=0)
clm.fit(reduced_data)
preds = clm.predict(reduced_data)
centers = clm.cluster_centers_
sample_preds = clm.predict(pca_samples)
score = silhouette_score(reduced_data, preds, random_state=10)
print "Number of clusters = {}, Score = {}".format(2, np.round(score, 4))

print "\n"
print "Inverse transform of cluster centers data points\n"


#----------- OJO--------------
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

print ""
print "------------------ >< ------------------\n"




print type(preds)

print good_data.head(5)

true_good_data = np.exp(good_data)

print good_data.head(5)
print ""
print data.head(5)
print ""
print true_good_data.head(5)
print "Number of samples in cleaned data = {} ".format(len(true_good_data))
print "Number of samples in clustering predictions data = {} ".format(len(preds))
print ""



print "         Ready for classification         "
print "------------------ >< ------------------\n"


from sklearn.cross_validation import train_test_split

#Split the data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(true_good_data, preds, test_size = 0.25, random_state = 0)

#Initialize classifier

from time import time
from sklearn.metrics import fbeta_score
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=0)

start = time()
learner = dt.fit(X_train, y_train)
end = time()
train_time = end-start

start = time()
predictions_test = learner.predict(X_test)
predictions_train = learner.predict(X_train)
end = time()
pred_time = end-start


acc_train_score = accuracy_score(y_train, predictions_train)
acc_test_score= accuracy_score(y_test, predictions_test)

f_train_score = fbeta_score(y_train, predictions_train, beta=0.5)
f_test_score  = fbeta_score(y_test, predictions_test, beta=0.5)

print "Classifier Accuracy = {}, F-Score = {} ".format(acc_test_score, f_test_score)
print ""

predictions_samples = learner.predict(samples)

print predictions_samples


print "Descriptive statistics on good_data:\n"
stats_good_data = true_good_data.describe()

print stats_good_data
print ""

print stats_good_data['Fresh']['mean']
print stats_good_data['Fresh']['std']

std_array = []

for feat in cols:
    std_array.append(stats_good_data[feat]['std'])
print "Standard deviations in reduced space"
print std_array

print "True centers:\n"
print true_centers

print ""
print " ******  Simplified Gaussian generation of samples  ******"

simdata = []
for feat in cols:
    mu = stats_good_data[feat]['mean']
    sigma = stats_good_data[feat]['std']
    rs = list(np.abs(np.random.normal(mu, sigma, 10)))
    simdata.append(rs)

frame = []
for i in range(len(cols)):
    tmp =[]
    for s in simdata:
        tmp.append(s[i])
    frame.append(tmp)

simdata_df = pd.DataFrame.from_records(frame, columns=cols)

print "Gaussian generated samples:"
display(simdata_df)

log_simdata = np.log(simdata_df)
pca2 = PCA(n_components=2)
pca2.fit(good_data)
pca2_simdata = pca2.transform(log_simdata)

#**************************************************************


print "         Ready for visualization         "
print "------------------ >< ------------------\n"

def cluster_results(reduced_data, preds, centers, pca_samples):
    '''
    Visualizes the PCA-reduced cluster data in two dimensions
    Adds cues for cluster centers and student-selected sample data
    '''

    import matplotlib.cm as cm

    predictions = pd.DataFrame(preds, columns=['Cluster'])
    plot_data = pd.concat([predictions, reduced_data], axis=1)

    # Generate the cluster plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Color map
    cmap = cm.get_cmap('gist_rainbow')

    # Color the points based on assigned cluster
    for i, cluster in plot_data.groupby('Cluster'):
        cluster.plot(ax=ax, kind='scatter', x='Dimension 1', y='Dimension 2', \
                     color=cmap((i) * 1.0 / (len(centers) - 1)), label='Cluster %i' % (i), s=30);

    # Plot centers with indicators
    for i, c in enumerate(centers):
        ax.scatter(x=c[0], y=c[1], color='white', edgecolors='black', \
                   alpha=1, linewidth=2, marker='o', s=200);
        ax.scatter(x=c[0], y=c[1], marker='$%d$' % (i), alpha=1, s=100);

    # Plot transformed sample points
    ax.scatter(x=pca_samples[:, 0], y=pca_samples[:, 1], \
               s=150, linewidth=4, color='black', marker='x');

    # Set plot title
    ax.set_title(
        "Cluster Learning on PCA-Reduced Data - Centroids Marked by Number\nTransformed Sample Data Marked by Black Cross");


#Predicting cluster membership via decision tree
predicted_simdata_cls = learner.predict(simdata_df)

print predicted_simdata_cls




