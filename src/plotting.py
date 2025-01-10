import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from clustering_metrics import leaky_relu_mlrd
import seaborn as sns


def plot_ts_clusters(cluster_ids, cluster_labels, ts_data):
    cluster_counts = {cluster_id: 0 for cluster_id in cluster_ids}
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(15, 15))
    flattened_axes = axes.flatten()
    for idx, time_series_id in enumerate(ts_data.columns):
        cluster_id = cluster_labels[idx]
        cluster_counts[cluster_id] += 1
        ax = flattened_axes[cluster_id - 1]
        ax.plot(ts_data[time_series_id], label=f"Cluster {cluster_id}")
        ax.set_ylabel("LOS displacement [m]")
        ax.set_xlabel("Time")
    for cluster_id, ax in enumerate(flattened_axes, start=1):
        ax.set_title(f"Cluster {cluster_id} ({cluster_counts[cluster_id]} time series)")
    plt.show()


def scatterplot_clusters(
    ground_truth_labels,
    results_labels_wfeats,
    baseline_labels_ew,
    baseline_labels_ud,
    ew_extra_feats,
    ts_data,
):
    # scatterplot on lat lon
    results_df = pd.concat(
        [
            pd.DataFrame(
                [("Ground Truth", cluster) for cluster in ground_truth_labels],
                columns=["source", "cluster"],
            ),
            pd.DataFrame(
                [
                    ("Time2Feat", cluster)
                    for cluster in results_labels_wfeats["time2feat"]["EW + UD"][0][1]
                ],
                columns=["source", "cluster"],
            ),
            pd.DataFrame(
                [
                    ("FeatTS_EW", cluster)
                    for cluster in results_labels_wfeats["featts"]["EW"][0][1]
                ],
                columns=["source", "cluster"],
            ),
            pd.DataFrame(
                [
                    ("FeatTS_UD", cluster)
                    for cluster in results_labels_wfeats["featts"]["UD"][0][1]
                ],
                columns=["source", "cluster"],
            ),
            pd.DataFrame(
                [("TSInSAR_EW", cluster) for cluster in baseline_labels_ew],
                columns=["source", "cluster"],
            ),
            pd.DataFrame(
                [("TSInSAR_UD", cluster) for cluster in baseline_labels_ud],
                columns=["source", "cluster"],
            ),
        ],
        ignore_index=True,
    )

    coords_df = pd.concat(
        [ew_extra_feats] * len(results_df["source"].unique()), ignore_index=True
    )
    results_df = pd.concat([results_df, coords_df], axis=1)
    results_df["cluster"] = results_df["cluster"].astype("category")
    dict_results = leaky_relu_mlrd(ew_extra_feats, results_df, ts_data)

    for source in results_df["source"].unique():
        for class_ins in results_df["cluster"].unique():
            print(source, class_ins, dict_results[(source, class_ins)])
        _, ax = plt.subplots(figsize=(16, 9))
        palette = sns.color_palette("husl", 17)
        sns.scatterplot(
            data=results_df[results_df["source"] == source],
            x="LAT",
            y="LON",
            hue="cluster",
            ax=ax,
            palette=palette,
        )
        ax.set_title(f"{source} - Cluster distribution on LAT and LON")
        plt.show()


def kdeplot_clusters(
    ground_truth_labels, results_labels_wfeats, baseline_labels_ew, baseline_labels_ud
):
    # kdeplot on
    # - feat ts ud + ew (and ground truth)
    # - tsinsar ud + ew (and ground truth)
    # - time2feat (and ground truth)
    plot_df = pd.concat(
        [
            pd.DataFrame(
                [
                    ("Ground Truth", cluster, count)
                    for cluster, count in zip(
                        *np.unique(ground_truth_labels, return_counts=True)
                    )
                ],
                columns=["source", "cluster", "count"],
            ),
            pd.DataFrame(
                [
                    ("Time2Feat", cluster, count)
                    for cluster, count in zip(
                        *np.unique(
                            results_labels_wfeats["time2feat"]["EW + UD"],
                            return_counts=True,
                        )
                    )
                ],
                columns=["source", "cluster", "count"],
            ),
            pd.DataFrame(
                [
                    ("FeatTS_EW", cluster, count)
                    for cluster, count in zip(
                        *np.unique(
                            results_labels_wfeats["featts"]["EW"], return_counts=True
                        )
                    )
                ],
                columns=["source", "cluster", "count"],
            ),
            pd.DataFrame(
                [
                    ("FeatTS_UD", cluster, count)
                    for cluster, count in zip(
                        *np.unique(
                            results_labels_wfeats["featts"]["UD"], return_counts=True
                        )
                    )
                ],
                columns=["source", "cluster", "count"],
            ),
            pd.DataFrame(
                [
                    ("TSInSAR_EW", cluster, count)
                    for cluster, count in zip(
                        *np.unique(baseline_labels_ew, return_counts=True)
                    )
                ],
                columns=["source", "cluster", "count"],
            ),
            pd.DataFrame(
                [
                    ("TSInSAR_UD", cluster, count)
                    for cluster, count in zip(
                        *np.unique(baseline_labels_ud, return_counts=True)
                    )
                ],
                columns=["source", "cluster", "count"],
            ),
        ],
        ignore_index=True,
    )

    long_df = pd.melt(
        plot_df,
        id_vars=["source", "cluster"],
        value_vars=["count"],
        var_name="variable",
        value_name="value",
    )
    plot_df_1 = long_df[
        long_df["source"].isin(["Ground Truth", "FeatTS_EW", "FeatTS_UD"])
    ]
    plot_df_2 = long_df[
        long_df["source"].isin(["Ground Truth", "TSInSAR_EW", "TSInSAR_UD"])
    ]
    plot_df_3 = long_df[long_df["source"].isin(["Ground Truth", "Time2Feat"])]
    _, (ax1, ax2, ax3) = plt.subplots(figsize=(16, 9), ncols=3, sharex="all")
    sns.kdeplot(
        data=plot_df_1, x="cluster", hue="source", fill=False, alpha=0.8, ax=ax1
    )
    sns.kdeplot(
        data=plot_df_2, x="cluster", hue="source", fill=False, alpha=0.8, ax=ax2
    )
    sns.kdeplot(
        data=plot_df_3, x="cluster", hue="source", fill=False, alpha=0.8, ax=ax3
    )
    plt.show()


def plot_unsupervised_metrics_range():
    # Plot the unsupervised metrics for each number of clusters
    # (e.g. silhouette score, Davies-Bouldin score, etc.)
    """metrics = ['silhouette_score', 'davies_bouldin_score', 'calinski_harabasz_score']
    for metric in metrics:
        scores = [results_labels_per_n_clusters[n_clusters][metric] for n_clusters in
                  results_labels_per_n_clusters.keys()]
        plt.plot(list(results_labels_per_n_clusters.keys()), scores, label=metric)
    plt.xlabel('Number of clusters')
    plt.ylabel('Score')
    plt.legend()
    plt.show()"""

    data = pd.read_csv(
        "metrics_results_methods_results.csv", header=[0, 1, 2], index_col=0
    )
    ground_truth = pd.read_csv(
        "metrics_results_ground_truth_labels.csv", header=[0, 1, 2], index_col=0
    )
    baseline_ew = pd.read_csv(
        "metrics_results_baseline_labels_ew.csv", header=[0, 1, 2], index_col=0
    )
    baseline_ud = pd.read_csv(
        "metrics_results_baseline_labels_ud.csv", header=[0, 1, 2], index_col=0
    )

    ground_truth_n_clusters = ground_truth.columns.levels[0][0]
    baseline_n_clusters = baseline_ew.columns.levels[0][0]

    _, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(
        figsize=(16, 9), ncols=3, nrows=3, sharex="col"
    )

    ax7.axis("off")
    ax9.axis("off")

    for (metric, row), ax in zip(data.iterrows(), [ax1, ax2, ax3, ax4, ax5, ax6, ax8]):
        n_clusters = row.index.levels[0].tolist()
        methods = row.index.levels[1].tolist()
        datasets = row.index.levels[2].tolist()
        for method in methods:
            for dataset in datasets:
                method_metrics = [
                    row.loc[(n_cluster, method, dataset)] for n_cluster in n_clusters
                ]
                sns.barplot(
                    data=data,
                    x=n_clusters,
                    y=method_metrics,
                    ax=ax,
                    zorder=2,
                    legend=False,
                    label=f"{method} {dataset}",
                )
        ground_truth_score = ground_truth.loc[
            metric, (ground_truth_n_clusters, "ground_truth", "EW + UD")
        ]
        baseline_ew_score = baseline_ew.loc[
            metric, (baseline_n_clusters, "baseline", "EW")
        ]
        baseline_ud_score = baseline_ud.loc[
            metric, (baseline_n_clusters, "baseline", "UD")
        ]

        ax.axhline(
            ground_truth_score,
            color="r",
            linestyle="--",
            label="Ground Truth",
            zorder=3,
            linewidth=0.5,
            alpha=0.8,
        )
        ax.axhline(
            baseline_ew_score,
            color="b",
            linestyle="--",
            label="Baseline EW",
            zorder=3,
            linewidth=0.5,
            alpha=0.8,
        )
        ax.axhline(
            baseline_ud_score,
            color="g",
            linestyle="--",
            label="Baseline UD",
            zorder=3,
            linewidth=0.5,
            alpha=0.8,
        )
        ax.set_title(metric)
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("Score")
        ax.grid(axis="x", zorder=0)

    plt.figlegend(
        handles=[
            plt.Line2D([0], [0], color="r", linestyle="--", label="Ground Truth"),
            plt.Line2D([0], [0], color="b", linestyle="--", label="Baseline EW"),
            plt.Line2D([0], [0], color="g", linestyle="--", label="Baseline UD"),
        ],
        loc="lower left",
        ncol=1,
    )
    plt.suptitle("Unsupervised Clustering Metrics - Time2Feat on EW + UD", fontsize=16)
    # plt.title('Time2Feat on EW + UD')  # TODO: get this from data

    plt.show()
