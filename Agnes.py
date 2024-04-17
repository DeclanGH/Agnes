#!/usr/bin/env python

import os.path
import numpy as np
import pandas
import matplotlib.pyplot as plt
import seaborn as sns


####################
# DRAW CLUSTERS    #
####################

# Plot the data points in Euclidean space, color-code by cluster
def DrawClusters(dataframe):
    sns.relplot(data=dataframe, x=dataframe.columns[0], y=dataframe.columns[1], hue='clusters', aspect=1.61, palette="tab10")
    plt.show()

####################
# LOAD DATA        #
####################
def LoadData(DB_PATH):

    # Load the input file into a Pandas DataFrame object
    dataframe = pandas.read_csv(DATA_PATH, sep=';', encoding='cp1252')

    # Check how many rows and columns are in the loaded data
    assert dataframe.shape[1] == 22, "Unexpected input data shape."

    # use PROJECT operation to filter down to longitude and Latitude\
    dataframe = dataframe[["latitude", "longitude"]]
    assert dataframe.shape[1] == 2, "Unexpected projected data shape."

    return dataframe

####################
# GET NUM CLUSTERS #
####################
def GetNumClusters(dataframe):

    # Get the number of unique clusters
    num_clusters = dataframe["clusters"].nunique()

    return num_clusters

####################
# GET CLUSTER IDS  #
####################
def GetClusterIds(dataframe):

    # Get the unique IDs of each cluster
    cluster_ids = dataframe["clusters"].unique()

    return cluster_ids

####################
# GET CLUSTER      #
####################
def GetCluster(dataframe, cluster_id):

    # Perform a SELECT operation to return only rows in the specified cluster

    cluster = dataframe[["longitude", "latitude"]][dataframe["clusters"] == cluster_id]

    return cluster.to_numpy()

####################
# DISTANCE         #
####################
def Distance(lhs, rhs):

    # Calculate the Euclidean distance between two rows
    dist = np.linalg.norm(lhs - rhs)

    return dist

####################
# SINGLE LINK DIST #
####################
def SingleLinkDistance(lhs, rhs):

    # Calculate the single-link distance between two clusters
    min_dist = float('inf')
    for lnglat_a in lhs:
        for lnglat_b in rhs:
            dist = Distance(lnglat_a, lnglat_b)
            if dist < min_dist:
                min_dist = dist

    return min_dist

######################
# COMPLETE LINK DIST #
######################
def CompleteLinkDistance(lhs, rhs):

    # Calculate the complete-link distance between two clusters
    max_dist = 0.0
    for lnglat_a in lhs:
        for lnglat_b in rhs:
            dist = Distance(lnglat_a, lnglat_b)
            if dist > max_dist:
                max_dist = dist

    return max_dist

#######################
# RECURSIVELY CLUSTER #
#######################
def RecursivelyCluster(dataframe, K, M):

    # Check if we have reached the desired number of clusters
    if GetNumClusters(dataframe) == K : return dataframe

    # Find the closest 2 clusters
    cluster_ids = GetClusterIds(dataframe)
    closest_pair = None
    min_distance = float('inf')

    for i in range(len(cluster_ids)):
        for j in range(i + 1, len(cluster_ids)):
            dist = M(GetCluster(dataframe, cluster_ids[i]), GetCluster(dataframe, cluster_ids[j]))
            if dist < min_distance:
                min_distance = dist
                closest_pair = (cluster_ids[i], cluster_ids[j])

    # Merge the closest 2 clusters
    id_a, id_b = closest_pair
    dataframe.loc[dataframe["clusters"] == id_b, "clusters"] = id_a

    # Recurse
    result = RecursivelyCluster(dataframe, K, M)

    return result

####################
# AGNES            #
####################
def Agnes(db_path, K, M):

    # Load the data in and select the features/attributes to work with (lat, lon)
    dataframe = LoadData(db_path)
    assert dataframe.shape[1] == 2, "Unexpected input data shape (lat, lon)."

    # Add each datum to its own cluster
    dataframe["clusters"] = dataframe.index.tolist()
    assert dataframe.shape[1] == 3, "Unexpected input data shape (lat, lon, cluster)."

    # Generate clusters from all points and recursively merge
    results = RecursivelyCluster(dataframe, K, M)

    return results

####################
# MAIN             #
####################
if __name__=="__main__":

    RUN_UNIT_TEST = True
    if RUN_UNIT_TEST:
        # Path where you downloaded the data
        DATA_PATH = './unit_test_data.csv'
        K = 2 # The number of output clusters.
        M = SingleLinkDistance # The cluster similarity measure M to be used.

        # Run the AGNES algorithm with the unit test data
        results = Agnes(DATA_PATH, K, M)
        assert results.shape == (5,3), "Unexpected output data shape. {}".format(results.shape)

        # Write results to file
        f = open("agnes_unit_test.txt", "w")
        f.write(results.to_csv(header=False))
        f.close()
        DrawClusters(results)

    # for full dataset, we modify the following line to True
    RUN_FULL_SINGLE_LINK = False
    if RUN_FULL_SINGLE_LINK:
        # Path where you downloaded the data
        DATA_PATH = './apartments_for_rent_classified_100.csv'
        K = 6 # The number of output clusters.
        M = SingleLinkDistance # The cluster similarity measure M to be used.

        # Run the AGNES algorithm using single-link
        results = Agnes(DATA_PATH, K, M)
        assert results.shape == (97,3), "Unexpected output data shape. {}".format(results.shape)

        # Write results to file
        f = open("agnes_single_link.txt", "w")
        f.write(results.to_csv(header=False))
        f.close()
        DrawClusters(results)

    # same here, true for full dataset complete link run
    RUN_FULL_COMPLETE_LINK = False
    if RUN_FULL_COMPLETE_LINK:
        # Path where you downloaded the data
        DATA_PATH = './apartments_for_rent_classified_100.csv'
        K = 6 # The number of output clusters.
        M = CompleteLinkDistance # The cluster similarity measure M to be used.

        # Run the AGNES algorithm using complete-link
        results = Agnes(DATA_PATH, K, M)
        assert results.shape == (97,3), "Unexpected output data shape. {}".format(results.shape)

        # Write results to file
        f = open("agnes_complete_link.txt", "w")
        f.write(results.to_csv(header=False))
        f.close()
        DrawClusters(results)
