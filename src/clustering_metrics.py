import numpy as np
import pandas as pd
from sklearn.metrics.cluster import (
    adjusted_mutual_info_score as ami,
    adjusted_rand_score as ari,
    normalized_mutual_info_score as nmi,
    silhouette_score as sil,
    calinski_harabasz_score as chs,
    davies_bouldin_score as dbs,
)
from jqmcvi.base import dunn
from sklearn.neighbors import LocalOutlierFactor


def mean_lrd(X, k):
    lof = LocalOutlierFactor(n_neighbors=k)
    lof.fit(X)
    lrd = lof._lrd
    return np.mean(lrd)


def leaky_relu_mlrd(
    methods_results: np.ndarray,
    euclidean_feats: pd.DataFrame,
    k_perc: float = 0.98,
    alpha: float = 0.01,
    beta: float = 1.0,
    return_mlrd_per_cluster: bool = False,
):
    """
    Compute the leaky relu mean local reachability density (MLRD) score for each cluster created by each method.
    :param methods_results: The results of the clustering methods.
    :param euclidean_feats: The Euclidean features to compute the LRD from.
    :param k_perc: The percentage of points to consider in a cluster for the LRD computation.
    :param alpha: The scaling factor for the small clusters.
    :param beta: The scaling factor for the large clusters.
    :param return_mlrd_per_cluster: Whether to return the MLRD score per cluster.

    :return: The leaky relu MLRD score for each cluster created by each method.
    """

    def penalizer(lrd: float, return_factor: bool = False):
        """
        Compute the leaky relu value given an LRD value lrd following these cases:
            0 < n < 2 → 0,
            2 <= n < k/N → alpha * n/N,
            n = int(k / N) → mean(alpha * n/N, beta * n/N),
            n > k/N → beta * n/N;

            where:
                n is the number of points in the cluster,
                k is the number of clusters in the dataset,
                N is the total number of points in the dataset,
                alpha is a scaling factor for the small clusters,
                beta is a scaling factor for the large clusters.

        :param lrd: The local reachability density value.
        :param return_factor: Whether to return the scaling factor.

        :return: The leaky relu value.
        """
        n_clusters = len(results_df["cluster"].unique())

        points_in_dataset = len(results_df["cluster"])
        expected_points_per_cluster = points_in_dataset / n_clusters

        if (
            points_in_cluster < 2
        ):  # less than 2 points in the cluster, the score is zero
            return 0

        # less than the average (expected) number of points per cluster
        if points_in_cluster < expected_points_per_cluster:
            scaling_factor = alpha * (points_in_cluster / points_in_dataset)

        elif points_in_cluster == expected_points_per_cluster:
            scaling_factor = np.mean(
                [
                    alpha * (points_in_cluster / points_in_dataset),
                    beta * (points_in_cluster / points_in_dataset),
                ]
            )
        else:
            scaling_factor = beta * points_in_cluster / points_in_dataset

        if return_factor:
            return lrd * scaling_factor, scaling_factor
        else:
            return lrd * scaling_factor

    results_dict = {}
    # prepare data shape for the leaky_relu_mlrd function
    """results_df = pd.concat(
        [
            pd.DataFrame(
                [('Ground Truth', cluster) for cluster in methods_results['ground_truth_labels']],
                columns=['source', 'cluster']
            ),
            pd.DataFrame(
                [('Time2Feat', cluster) for cluster in
                 methods_results["results_labels"]['time2feat']['EW + UD'][0][1]],
                columns=['source', 'cluster']
            ),
            pd.DataFrame(
                [('FeatTS_EW', cluster) for cluster in methods_results["results_labels"]['featts']['EW'][0][1]],
                columns=['source', 'cluster']),
            pd.DataFrame(
                [('FeatTS_UD', cluster) for cluster in methods_results["results_labels"]['featts']['UD'][0][1]],
                columns=['source', 'cluster']),
            pd.DataFrame(
                [('TSInSAR_EW', cluster) for cluster in
                 methods_results['baseline_labels_ew']],
                columns=['source', 'cluster']),
            pd.DataFrame(
                [('TSInSAR_UD', cluster) for cluster in
                 methods_results['baseline_labels_ud']],
                columns=['source', 'cluster'])
        ],
        ignore_index=True
    )"""
    results_df = pd.DataFrame(
        [("method", cluster) for cluster in methods_results],
        columns=["source", "cluster"],
    )
    # concatenate features for mlrd computation
    results_df = pd.concat([results_df, euclidean_feats], axis=1)
    results_df["cluster"] = results_df["cluster"].astype("category")

    if return_mlrd_per_cluster:
        mlrd_per_cluster = dict()
    else:
        mlrd_per_cluster = None

    for cluster_label in results_df["cluster"].unique():
        subset_df = results_df[results_df["cluster"] == cluster_label][
            euclidean_feats.columns.to_list()
        ]

        if len(subset_df) < 2:  # less than 2 points in the cluster, the score is zero
            results_dict[cluster_label] = 0
        else:
            points_in_cluster = len(subset_df)

            mlrd = mean_lrd(subset_df, int(len(subset_df) * k_perc))

            lr_mlrd, factor = penalizer(mlrd, return_factor=True)

            if return_mlrd_per_cluster:
                mlrd_per_cluster[cluster_label] = mlrd

            results_dict[cluster_label] = lr_mlrd

    clustering_mlrd = np.mean(list(results_dict.values()))

    if return_mlrd_per_cluster:
        return clustering_mlrd, mlrd_per_cluster
    else:
        return clustering_mlrd


def inertia(data, labels):
    """
    Compute the inertia of the clustering.
    The inertia is the sum of the squared distances between each training instance and its closest centroid.
    :param data: The data points.
    :param labels: The cluster labels of the data points.
    :return: The inertia of the clustering.
    """
    # Initialize the inertia to zero
    inertia = 0

    # Loop through unique cluster labels
    for cluster_label in np.unique(labels):
        # Get indices of data points belonging to the current cluster
        cluster_data_indices = np.where(labels == cluster_label)[0]

        # Get the centroid of the current cluster
        centroid = np.mean(data[cluster_data_indices], axis=0)

        # Compute the squared distances between each data point and the centroid
        squared_distances = np.sum((data[cluster_data_indices] - centroid) ** 2, axis=1)

        # Add the sum of squared distances to the inertia
        inertia += np.sum(squared_distances)

    return inertia


def distortion(data, labels):
    """
    Compute the distortion_ of the clustering.
    The distortion_ is the mean sum of the squared distances between each training instance and its closest centroid.
    :param data: The data points.
    :param labels: The cluster labels of the data points.
    :return: The distortion_ of the clustering.
    """
    # Initialize the distortion_ to zero
    distortion_ = 0

    # Loop through unique cluster labels
    for cluster_label in np.unique(labels):
        # Get indices of data points belonging to the current cluster
        cluster_data_indices = np.where(labels == cluster_label)[0]

        # Get the centroid of the current cluster
        centroid = np.mean(data[cluster_data_indices], axis=0)

        # Compute the squared distances between each data point and the centroid
        squared_distances = np.sum((data[cluster_data_indices] - centroid) ** 2, axis=1)

        # Add the sum of squared distances to the distortion_
        distortion_ += np.sum(squared_distances)

    return distortion_ / len(data)


supervised = [
    ami,  # [0, 1], higher is better
    ari,  # [0, 1], higher is better
    nmi,  # [0, 1], higher is better
]

unsupervised = [
    sil,  # [-1, 1], higher is better
    chs,  # [0, inf), higher is better
    dbs,  # [0, inf), lower is better
    dunn,  # [0, inf), higher is better
    leaky_relu_mlrd,  # [0, inf), higher is better
    inertia,  # [0, inf), lower is better
    distortion,  # [0, inf), lower is better
]
