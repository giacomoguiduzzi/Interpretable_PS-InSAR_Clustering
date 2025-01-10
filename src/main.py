#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pickle
import warnings
from time import time

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from tabulate import tabulate
from kneed import KneeLocator
from pipeline import (
    load_and_prepare_data,
    run_experiments,
    online_optimize_clusters,
    compute_metrics,
)
import argparse

from plotting import plot_unsupervised_metrics_range
from utils import realign_labels, show_and_save_data

# from numba import cuda
# try freeing all GPU memory for the process
"""device = cuda.get_current_device()
if device:
    device.reset()"""

parser = argparse.ArgumentParser(
    prog="ClusteringLandslides", description="Run the clustering pipeline."
)
parser.add_argument(
    "--baseline_shapes", action="store_true", help="Use baseline shapes."
)
parser.add_argument(
    "--use_extra_features",
    action="store_true",
    default=False,
    help="Use extra features for clustering.",
)
parser.add_argument(
    "--sr", type=str, default="epsg:3857", help="Spatial reference system."
)
parser.add_argument(
    "--supervised",
    action="store_true",
    default=False,
    help="Use supervised clustering and metrics.",
)
parser.add_argument(
    "--plot_clusters",
    action="store_true",
    default=False,
    help="Plot the clusters in EW and UD components.",
)
parser.add_argument(
    "--num_runs",
    type=int,
    default=10,
    help="Number of runs for the clustering algorithm. The results are averaged over these runs per method and dataset.",
)
parser.add_argument(
    "--n_clusters",
    type=int,
    default=None,
    help="Number of clusters to use in the clustering algorithm.",
)

parser.add_argument(
    "--online_optimization_data",
    type=str,
    default=None,
    help="Path to the shape data file for the online optimization algorithm. For example, this file could contain "
    "information derived from the EW and UD components to determine the best number of clusters.",
)

parser.add_argument(
    "--min_n_clusters",
    type=int,
    default=2,
    help="Minimum number of clusters to test.",
)

parser.add_argument(
    "--methods",
    nargs="+",
    type=str,
    default=["kmeans", "kshape", "time2feat", "featts"],
    help="List of clustering methods to use.",
)

parser.add_argument(
    "--print_explanation",
    action="store_true",
    default=False,
    help="Explain the clustering results.",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
    default=False,
    help="Overwrite the clustering results.",
)

parser.add_argument(
    "--baseline_results",
    type=str,
    nargs="+",
    default=[
        "./data/baseline_labels/labels_ew_17.npy",
        "./data/baseline_labels/labels_ud_17.npy",
    ],
    help="Path to the baseline results files.",
)

parser.add_argument(
    "--patience",
    type=int,
    default=10,
    help="Patience for the online optimization algorithm. This value determines how many clustering solutions "
    "with increasing number of clusters are going to be tested before stopping the optimization.",
)

parser.add_argument(
    "--threshold_val",
    type=float,
    default=0.3,
    help="Threshold value for the online optimization algorithm. "
    "This value determines how much the clustering inertia should decrease in comparison to the average inertia "
    "of the past iterations to continue the optimization, resetting the patience counter.",
)

parser.add_argument(
    "--input_data",
    type=str,
    nargs="+",
    default=[
        "./data/unlabeled_data/EGMS_EW_18_22_3857.shp",
        "./data/unlabeled_data/EGMS_UD_18_22_3857.shp",
    ],
    help="Path to the input data files.",
)

parser.add_argument(
    "--output_filename",
    type=str,
    default="clustering_results.pickle",
    help="Output filename for the clustering results. The explanations file will have the same name with _explanations.",
)

parser.add_argument(
    "--ground_truth_col_name",
    type=str,
    default="CLUSTER",
    help="Column name for the ground truth labels.",
)

parser.add_argument(
    "--use_ground_truth",
    action="store_true",
    default=False,
    help="Use ground truth labels.",
)

args = parser.parse_args()


def main():
    if not args.output_filename.endswith(".pickle"):
        raise ValueError("Output filename should be a pickle file.")

    # init the variables to avoid undefined variable errors
    results_labels_ew = None
    results_labels_ud = None
    results_labels = None
    results_labels_per_method = None
    time2feat_results = None
    time2feat_explanation = None
    n_clusters = None
    explanations_per_dataset = None

    if not os.path.exists(args.output_filename) or args.overwrite:
        loaded_results = False

        # Load and prepare the data
        if len(args.input_data) == 1:
            (
                x_ew_ud,
                ps_ew_ud_2d,
                extra_feats,
                baseline_labels_ew,
                baseline_labels_ud,
                opt_data,
            ) = load_and_prepare_data(args)
            ps_ew_2d = None
            ps_ud_2d = None
            x_ew = None
            x_ud = None
            ew_extra_feats = None
            ud_extra_feats = None
        else:
            (
                x_ew,
                x_ud,
                ps_ew_2d,
                ps_ud_2d,
                ew_extra_feats,
                ud_extra_feats,
                baseline_labels_ew,
                baseline_labels_ud,
                opt_data,
            ) = load_and_prepare_data(args)
            ps_ew_ud_2d = None
            x_ew_ud = None
            extra_feats = None

        data_df = ps_ew_ud_2d if len(args.input_data) == 1 else ps_ew_2d
        # Extract labels for the ground truth
        if "CLUSTER" in data_df.columns:
            ground_truth_labels = data_df["CLUSTER"].values
        elif "cluster" in data_df.columns:
            ground_truth_labels = data_df["cluster"].values
        else:
            if not args.use_ground_truth:
                ground_truth_labels = None
            else:
                raise ValueError(
                    'No cluster labels found in the DataFrame (no columns called "CLUSTER" or "cluster".'
                )

        print(
            "Data points:",
            x_ew.shape[0] if len(args.input_data) > 1 else x_ew_ud.shape[0],
        )

        if args.n_clusters is None:
            if args.use_ground_truth:
                n_clusters = len(np.unique(ground_truth_labels))
            else:
                n_clusters = args.min_n_clusters
        else:
            if args.min_n_clusters != 2:  # if different from the default value
                raise ValueError(
                    "Both n_clusters and min_n_clusters are defined. "
                    "Please define only one accordingly with the supervised "
                    "or unsupervised clustering flag, respectively."
                )
            else:
                n_clusters = args.n_clusters

        n_clusters_ew, n_clusters_ud = None, None

        if not args.supervised:
            # Determine the optimal number of clusters
            min_k = args.min_n_clusters
            if opt_data is None:
                opt_data = x_ew_ud if len(args.input_data) == 1 else x_ew
                print("Optimizing the number of clusters for EW.")
            else:
                print("Optimizing the number of clusters using the provided data.")
            start_time = time()
            k_opt_ew = online_optimize_clusters(
                opt_data,
                min_k=min_k,
                thr=args.threshold_val,
                max_patience=args.patience,
            )
            ooc_time_ew = time() - start_time
            print(f"Execution time: {ooc_time_ew:.2f}s")
            k_opt = k_opt_ew
            k_opt_ud = None

            if len(args.input_data) > 1 and opt_data is None:
                print("Optimizing the number of clusters for UD.")
                start_time = time()
                k_opt_ud = online_optimize_clusters(
                    x_ud,
                    min_k=min_k,
                    thr=args.threshold_val,
                    max_patience=args.patience,
                )
                ooc_time_ud = time() - start_time
                print(f"Execution time: {ooc_time_ud:.2f}s")
                k_opt = None

            if opt_data is not None:
                print(f"Optimal number of clusters for EW+UD: {k_opt}")
            else:
                print(f"Optimal number of clusters for EW: {k_opt_ew}")
                print(f"Optimal number of clusters for UD: {k_opt_ud}")

            if len(args.input_data) == 1:
                n_clusters = k_opt
            else:
                n_clusters_ew = k_opt_ew

                if opt_data is None:
                    n_clusters_ud = k_opt_ud
                else:
                    n_clusters_ud = k_opt_ew

        else:  # supervised clustering
            n_clusters_ew = n_clusters
            n_clusters_ud = n_clusters

        if len(args.input_data) == 1:
            results_labels, explanations = run_experiments(
                x_ew_ud,
                ps_ew_ud_2d,
                merged_signals=True,
                x_ud=None,
                ps_ud_2d=None,
                methods=args.methods,
                labels=ground_truth_labels if args.supervised else None,
                n_clusters=n_clusters,
                supervised_metrics=args.supervised,
                ext_feats=extra_feats if args.use_extra_features else None,
                num_runs=args.num_runs,
            )

            with open(args.output_filename, "wb") as f:
                pickle.dump(results_labels, f)

            with open(
                args.output_filename.replace(".pickle", "_explanations.pickle"), "wb"
            ) as f:
                pickle.dump(explanations, f)
        else:
            if (
                n_clusters_ew is not None
                and n_clusters_ud is not None
                and n_clusters_ew == n_clusters_ud
            ):
                n_clusters = n_clusters_ew

            else:
                warnings.warn(
                    "The number of clusters for EW and UD components are different. "
                    "Using the EW number of clusters.",
                    category=UserWarning,
                )

            results_labels_ew_ud, explanations_ew_ud = run_experiments(
                x_ew=x_ew,
                ps_ew_2d=ps_ew_2d,
                x_ud=x_ud,
                ps_ud_2d=ps_ud_2d,
                methods=args.methods,
                labels=ground_truth_labels if args.supervised else None,
                n_clusters=n_clusters,
                supervised_metrics=args.supervised,
                ext_feats=ew_extra_feats if args.use_extra_features else None,
                num_runs=args.num_runs,
            )

            results_labels_ew, results_labels_ud = (
                results_labels_ew_ud["EW"] if "EW" in results_labels_ew_ud else None,
                results_labels_ew_ud["UD"] if "UD" in results_labels_ew_ud else None,
            )
            if "EW + UD" in results_labels_ew_ud or "time2feat" in results_labels_ew_ud:
                if "EW + UD" in results_labels_ew_ud:
                    time2feat_results = results_labels_ew_ud["EW + UD"]
                else:
                    time2feat_results = results_labels_ew_ud["time2feat"]
            else:
                time2feat_results = None

            explanations_ew, explanations_ud = (
                explanations_ew_ud["EW"] if "EW" in explanations_ew_ud else None,
                explanations_ew_ud["UD"] if "UD" in explanations_ew_ud else None,
            )
            if (
                "EW + UD" in explanations_ew_ud or "time2feat" in explanations_ew_ud
            ):  # TODO: fix time2feat explanation object in compute_clusters
                if "EW + UD" in explanations_ew_ud:
                    time2feat_explanation = explanations_ew_ud["EW + UD"]
                else:
                    # time2feat_explanation = explanations_ew_ud["time2feat"]  # TODO: fix
                    time2feat_explanation = None
            else:
                time2feat_explanation = None

            with open(args.output_filename, "wb") as results_file:
                results_dict = {  # TODO: should the keys be the component (EW, UD9 or the method name?
                    "EW": results_labels_ew,
                    "UD": results_labels_ud,
                }
                if time2feat_results is not None:
                    results_dict["time2feat"] = time2feat_results

                pickle.dump(
                    results_dict,
                    results_file,
                )

            with open(
                args.output_filename.replace(".pickle", "_explanations.pickle"), "wb"
            ) as explanations_file:
                explanations_dict = {
                    "EW": explanations_ew,
                    "UD": explanations_ud,
                }
                if time2feat_explanation is not None:
                    explanations_dict["time2feat"] = time2feat_explanation

                pickle.dump(explanations_dict, explanations_file)
    else:
        loaded_results = True
        # TODO: fix this in case of a single file input
        (
            ground_truth_labels,
            ew_extra_feats,
            baseline_labels_ew,
            baseline_labels_ud,
        ) = load_and_prepare_data(args)

        with open(args.output_filename, "rb") as f:
            results_labels_per_method = pickle.load(f)

        if args.print_explanation:
            with open(
                args.output_filename.replace(".pickle", "_explanations.pickle"), "rb"
            ) as f:
                explanations_per_dataset = pickle.load(f)
        else:
            explanations_per_dataset = None

        if (
            "EW + UD" in results_labels_per_method
            or "time2feat" in results_labels_per_method
        ):
            if "EW + UD" in results_labels_per_method:
                time2feat_results = results_labels_per_method["EW + UD"]
            else:
                time2feat_results = results_labels_per_method["time2feat"]

        else:
            results_labels_ew = results_labels_per_method["EW"]
            results_labels_ud = results_labels_per_method["UD"]

    if not loaded_results:
        results_labels_per_method = {
            "EW": (
                results_labels_ew["EW"]
                if results_labels_ew is not None and "EW" in results_labels_ew
                else None
            ),
            "UD": (
                results_labels_ud["UD"]
                if results_labels_ud is not None and "UD" in results_labels_ud
                else None
            ),
            "time2feat": time2feat_results if time2feat_results is not None else None,
        }

    # Realign labels in all datasets for coherence.
    # All labels should start from 1
    (
        ground_truth_labels,
        baseline_labels_ew,
        baseline_labels_ud,
        results_labels_per_method,
    ) = realign_labels(
        ground_truth_labels,
        baseline_labels_ew,
        baseline_labels_ud,
        results_labels_per_method,
    )

    # Compute metrics
    data = {
        "ground_truth_labels": ground_truth_labels,
        "results_labels": results_labels_per_method,
        "baseline_labels_ew": baseline_labels_ew,
        "baseline_labels_ud": baseline_labels_ud,
    }

    # TODO: workaround to make the compute_metrics function work, maybe.
    #  Remove this when the function is fixed.

    if loaded_results:
        # get a method that is not None and compute the length of the unique labels list
        for method in results_labels_per_method:
            if results_labels_per_method[method] is not None:
                # get the first dataset computed
                first_dataset = list(results_labels_per_method[method].keys())[0]
                # get first result from the list of runs
                first_pred_labels = results_labels_per_method[method][first_dataset][0][
                    1
                ]
                n_clusters = len(np.unique(first_pred_labels))

                break
    # n_clusters should be defined otherwise

    data["results_labels"] = {n_clusters: data["results_labels"]}

    metrics_results = compute_metrics(
        data,
        supervised_metrics=False,
        euclidean_feats=ew_extra_feats,
        num_runs=args.num_runs,
    )

    baseline_n_clusters = len(np.unique(baseline_labels_ew))
    ground_truth_n_clusters = (
        len(np.unique(ground_truth_labels)) if args.use_ground_truth else None
    )

    show_and_save_data(
        metrics_results,
        explanations_per_dataset,
        args.n_clusters,
        baseline_n_clusters,
        ground_truth_n_clusters,
        args.print_explanation,
    )


if __name__ == "__main__":
    main()
