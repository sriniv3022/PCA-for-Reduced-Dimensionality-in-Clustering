#!/usr/bin/env python
# coding: utf-8

# ### PCA for Reduced Dimensionality in Clustering

# **Loading the image data matrix (with rows as images and columns as features), performing min-max normalization on the data matrix so that each feature is scaled**

# In[1]:


# Importing necessary packages
import pandas as pd
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Loading the data
seg_class = pd.read_csv("C:/Users/SRINI/Downloads/segmentation_data/segmentation_classes.txt",delimiter='\t', header = None)
seg_class.head(10)


# In[3]:


seg_data = pd.read_csv("C:/Users/SRINI/Downloads/segmentation_data/segmentation_data.txt", header = None)
seg_data.head(10)


# In[4]:


seg_names = pd.read_csv("C:/Users/SRINI/Downloads/segmentation_data/segmentation_names.txt", header = None)
seg_names.head(10)


# In[5]:


# Scaling the data
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0, 1))

# Normalizing the data matrix
seg_data_norm = scaler.fit_transform(seg_data)

seg_data_norm


# **Performing Kmeans clustering on the image data using Euclidean distance as distance measure for the clustering.**

# In[6]:


# Getting actual labels
ground_truth = seg_class.iloc[:,1]
ground_truth = np.array(ground_truth)
ground_truth


# In[7]:


from sklearn.cluster import KMeans
from sklearn.metrics import completeness_score, homogeneity_score

# K-means clustering
kmeans = KMeans(n_clusters=7, random_state=42)
kmeans.fit(seg_data_norm)

# Cluster labels and centroids
cluster_lbls = kmeans.labels_
cluster_cent = kmeans.cluster_centers_
print("7 Cluster Centroids:")

for i in cluster_cent:
    print(i)
    print("\n")

# Compute Completeness and Homogeneity scores
completeness = completeness_score(ground_truth, cluster_lbls)
homogeneity = homogeneity_score(ground_truth, cluster_lbls)

print("Completeness score:", completeness)
print("Homogeneity score:", homogeneity)


# **Performing PCA on the normalized image data matrix.** 

# In[8]:


X = np.mat(seg_data_norm)
meanVals = X.mean(axis=0)

# Centered matrix
A = X - meanVals

# Covariance matrix
C = np.cov(A, rowvar=0)    
print(C)


# In[9]:


import numpy as np

# Compute eigenvalues and eigenvectors
eigen_values, eigen_vectors = np.linalg.eig(C)

# Create a mapping of eigenvalues to eigenvectors
eigen_mapping = [(eigen_values[i], eigen_vectors[:, i]) for i in range(len(eigen_values))]

# Sort the eigenvectors based on eigenvalues in descending order
sorted_eigen_mapping = sorted(eigen_mapping, key=lambda x: x[0], reverse=True)

# Extract the sorted eigenvalues and eigenvectors
sorted_eigen_values = np.array([eigen_value for eigen_value, _ in sorted_eigen_mapping])
sorted_eigen_vectors = np.array([eigen_vector for _, eigen_vector in sorted_eigen_mapping])

# Print the sorted eigenvalues
print("Sorted Eigen Values:")
print(sorted_eigen_values)

# Print the sorted eigenvectors
print("Sorted Eigen Vectors:")
print(sorted_eigen_vectors)


# In[10]:


newFeatures = sorted_eigen_vectors.T
XTrans = np.dot(newFeatures, A.T)
print(XTrans.T)


# In[11]:


reducedFeatures = sorted_eigen_vectors[:,0:7].T
reducedXTrans = np.dot(reducedFeatures, A.T)
print(reducedXTrans.T)


# In[12]:


reducedXTrans.T.shape


# **Performing Kmeans again and computing the Completeness and Homogeneity values of the new clusters.**

# In[13]:


# Perform K-means clustering
kmeans = KMeans(n_clusters=7, random_state=42)
kmeans.fit(np.asarray(reducedXTrans.T))

# Cluster labels and centroids
cluster_lbls = kmeans.labels_
cluster_cent = kmeans.cluster_centers_
print("7 Cluster Centroids:")

for i in cluster_cent:
    print(i)
    print("\n")

# Compute Completeness and Homogeneity scores
completeness = completeness_score(ground_truth, cluster_lbls)
homogeneity = homogeneity_score(ground_truth, cluster_lbls)

print("Completeness score:", completeness)
print("Homogeneity score:", homogeneity)


# **Observation**

# In[14]:


# without PCA - observation 1
Completeness: 0.613187012485301
Homogeneity: 0.6115021163370862

# with PCA (reduced data) - observation 2
Completeness: 0.6585886426136276
Homogeneity: 0.6057460081163977


# Comparing the two results, we can see that the Completeness score of the reduced data(obs 2) is better than the observation 1.
# And the Homogeneity score of observation 1 is slightly better than observation 2. Comparing the overall performance using the completeness and homogeneity, performing clustering in the reduced data has improved the performance, as the observation 2 captures more data points from the true class than the observation 1.
# 
# In a nutshell, based on the completeness and homogeneity scores, Clustering Result 2 seems to perform slightly better than Result 1. It captures a larger portion of data points from the same true class within the same cluster

# In[ ]:




