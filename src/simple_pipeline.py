import argparse
import os
import pickle
import warnings
from time import time

import numpy as np
import pandas as pd
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

from geoutils import read_shapefile, extract_ts
from pipeline import (
    scale_dataset,
    scale_extra_features,
    online_optimize_clusters,
    compute_clusters,
    compute_metrics,
)
from src.utils import realign_labels

parser = argparse.ArgumentParser(
    prog="ClusteringLandslides", description="Run the clustering pipeline."
)
parser.add_argument(
    "--baseline_shapes", action="store_true", help="Use baseline shapes."
)
parser.add_argument(
    "--use_extra_features",
    action="store_true",
    default="True",
    help="Use extra features for clustering.",
)

parser.add_argument(
    "--selection_extra_features",
    action="store_true",
    default="False",
    help="Apply features selection on extra features for clustering.",
)

parser.add_argument(
    "--online_cluster_optimization",
    action="store_true",
    default=False,
    help="Use online cluster optimization.",
)

parser.add_argument(
    "--pfa_value",
    type=float,
    default=0.5,
    help="PFA value for feature selection.",
)

parser.add_argument(
    "--mismatch_opt",
    type=str,
    default="UD",
    help="PFA value for feature selection.",
)

parser.add_argument(
    "--dataset_name",
    type=str,
    default="last_experiments",
    help="PFA value for feature selection.",
)

parser.add_argument(
    "--plot_online_optimization_inertia",
    action="store_true",
    default=False,
    help="Plot the inertia values for the online optimization algorithm.",
)

parser.add_argument(
    "--folder_save",
    type=str,
    default="last_experiments",
    help="Where save the features and results obtained.",
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
    default=5,
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
    default=6,
    help="Minimum number of clusters to test.",
)

parser.add_argument(
    "--methods",
    nargs="+",
    type=str,
    # default=["kmeans", "kshape", "time2feat", "featts"],
    default=["time2feat"],
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
    default=True,
    help="Overwrite the clustering results.",
)

parser.add_argument(
    "--old_results",
    default=False,
    help="Use the old values.",
)

parser.add_argument(
    "--baseline_results",
    type=str,
    nargs="+",
    default=[
        "data/baseline_labels/labels_ew_17.npy",
        "data/baseline_labels/labels_ud_17.npy",
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
    default=0.1,
    help="Threshold value for the online optimization algorithm. "
    "This value determines how much the clustering inertia should decrease in comparison to the average inertia "
    "of the past iterations to continue the optimization, resetting the patience counter.",
)

parser.add_argument(
    "--input_data",
    type=str,
    nargs="+",
    default=[
        "",
        "",
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


def load_and_prepare_data():
    global args
    base_path = os.getcwd()
    sr = args.sr

    if not isinstance(args.input_data, list):
        raise ValueError(
            "The input_data parameter should be a list. Normally, this exception shouldn't be raised."
        )

    if len(args.input_data) > 2:
        raise ValueError(
            "Please provide at most two input data files: only one for both EW and UD, "
            "or one for the EW component and one for the UD component."
        )

    if len(args.input_data) == 1:
        print(
            "UserWarning: Only one input data file provided. Assuming it is the same for both components."
        )

    if len(args.input_data) >= 1:
        if not os.path.exists(args.input_data[0]):
            raise FileNotFoundError("Input data file not found.")

        if len(args.input_data) == 2:
            if not os.path.exists(args.input_data[1]):
                raise FileNotFoundError("Input data file not found.")

    if not isinstance(args.baseline_results, list):
        raise ValueError(
            "The baseline_results parameter should be a list. "
            "Normally, this exception shouldn't be raised."
        )

    if len(args.baseline_results) == 1:
        print(
            "UserWarning: Only one baseline results file provided. Assuming it is the same for both components."
        )

    if len(args.baseline_results) > 2:
        raise ValueError(
            "Please provide at most two baseline results files: only one for both EW and UD, "
            "or one for the EW component and one for the UD component."
        )

    if len(args.baseline_results) >= 1:
        if not os.path.exists(args.baseline_results[0]):
            raise FileNotFoundError("Baseline results file not found.")

        if len(args.baseline_results) == 2:
            if not os.path.exists(args.baseline_results[1]):
                raise FileNotFoundError("Baseline results file not found.")

    if args.baseline_shapes:
        print("Using baseline shapes.")
        ps_ew = read_shapefile(
            os.path.join(
                base_path,
                "InSAR-Time-Series-Clustering",
                "output",
                "Vh_df_array_EW.shp",
            ),
            sr,
            "MultiPoint",
        )
        ps_ud = read_shapefile(
            os.path.join(
                base_path,
                "InSAR-Time-Series-Clustering",
                "output",
                "Vv_df_array_UD.shp",
            ),
            sr,
            "MultiPoint",
        )

    else:
        ps_ew = read_shapefile(args.input_data[0], sr, "MultiPoint")
        ps_ud = read_shapefile(args.input_data[1], sr, "MultiPoint")

        # Fix a typo in a column
        if "tpi" in ps_ew.columns:
            ps_ew.rename(columns={"tpi": "twi"}, inplace=True)
            ps_ud.rename(columns={"tpi": "twi"}, inplace=True)

    extra_features_col_names = [
        "LAT",
        "LON",
        "dem",
        "slope",
        "aspect",
        "twi",
        "c_tot",
        "c_plan",
        "c_prof",
    ]
    if args.use_extra_features:
        # Extract extra features
        # Excluding the categorical external features which are not suitable to compute a Euclidean metric
        try:
            ew_extra_feats = ps_ew[
                [
                    "LAT",
                    "LON",
                    "dem",
                    "slope",
                    "aspect",
                    "twi",
                    "c_tot",
                    "c_plan",
                    "c_prof",
                ]
            ]
            ud_extra_feats = ps_ud[
                [
                    "LAT",
                    "LON",
                    "dem",
                    "slope",
                    "aspect",
                    "twi",
                    "c_tot",
                    "c_plan",
                    "c_prof",
                ]
            ]
        except KeyError:
            missing_cols = list(set(extra_features_col_names) - set(ps_ew.columns))
            present_cols = list(
                set(extra_features_col_names).intersection(ps_ew.columns)
            )
            warnings.warn(
                f"Missing extra features in data: {', '.join(missing_cols)}. "
                f"Keeping only {', '.join(present_cols)}.",
                category=UserWarning,
            )
            ew_extra_feats = ps_ew.loc[:, present_cols]
            ud_extra_feats = ps_ud.loc[:, present_cols]
    else:
        ew_extra_feats = None
        ud_extra_feats = None

    # Calculate lat e lon
    if None in ps_ew.geometry.values and not (
        "LAT" in ps_ew.columns and "LON" in ps_ew.columns
    ):
        raise ValueError(
            "The shapefile for EW does not contain geometry information (no 'geometry' column) and the LAT and LON columns are missing."
        )
    if None in ps_ud.geometry.values and not (
        "LAT" in ps_ud.columns and "LON" in ps_ud.columns
    ):
        raise ValueError(
            "The shapefile for UD does not contain geometry information (no 'geometry' column) and the LAT and LON columns are missing."
        )

    ps_ew["LAT"] = (
        ps_ew.geometry.y if None not in ps_ew.geometry.values else ps_ew["LAT"]
    )
    ps_ew["LON"] = (
        ps_ew.geometry.x if None not in ps_ew.geometry.values else ps_ew["LON"]
    )
    ps_ud["LAT"] = (
        ps_ew.geometry.y if None not in ps_ew.geometry.values else ps_ud["LAT"]
    )
    ps_ud["LON"] = (
        ps_ew.geometry.x if None not in ps_ew.geometry.values else ps_ud["LON"]
    )

    # Remove unnecessary columns
    ps_ew_2d = ps_ew.drop(
        columns=[
            "lat",
            "lon",
            "height",
            "rmse",
            "vel",
            "vel_std",
            "acc",
            "acc_std",
            "seas",
            "seas_std",
            "geometry",
        ],
        errors="ignore",
    )
    ps_ud_2d = ps_ud.drop(
        columns=[
            "lat",
            "lon",
            "height",
            "rmse",
            "vel",
            "vel_std",
            "acc",
            "acc_std",
            "seas",
            "seas_std",
            "geometry",
        ],
        errors="ignore",
    )

    # Reorder columns

    ps_ew_2d = ps_ew_2d[
        ["LAT", "LON"] + [col for col in ps_ew_2d.columns if col not in ["LAT", "LON"]]
    ]
    ps_ud_2d = ps_ud_2d[
        ["LAT", "LON"] + [col for col in ps_ud_2d.columns if col not in ["LAT", "LON"]]
    ]

    # Extract EW and UD time series
    ts_data = extract_ts(ps_ew_2d).copy().T

    if not args.baseline_shapes and not args.use_extra_features:
        x_ew = ts_data.copy()
        x_ew = x_ew.T.values

    else:
        x_ew = ts_data.copy().T.values

    # --- UD ---

    ts_data = extract_ts(ps_ud_2d).copy().T
    if not args.baseline_shapes and not args.use_extra_features:
        x_ud = ts_data.copy()
        x_ud = x_ud.T.values

    else:
        x_ud = ts_data.copy().T.values

    if x_ew.dtype == np.object_:
        x_ew = x_ew.astype(np.float32)
    if x_ud.dtype == np.object_:
        x_ud = x_ud.astype(np.float32)

    # Scale the data
    x_ew, x_ud = scale_dataset(x_ew, x_ud)

    # Use Ext. Feature available for compute metrics
    dict_feat_available = {}
    for i in extra_features_col_names:
        if i in ps_ew.columns:
            dict_feat_available[i] = ps_ew[i]

    print("Feature available: ", list(dict_feat_available.keys()))

    unsupervised_metrics_data = pd.DataFrame.from_dict(dict_feat_available)
    # Scale extra features
    if ew_extra_feats is not None or ud_extra_feats is not None:
        ps_datasets, extra_feats = scale_extra_features(
            {"ew": ps_ew_2d, "ud": ps_ud_2d},
            {"ew": ew_extra_feats, "ud": ud_extra_feats},
        )
        ps_ew_2d, ps_ud_2d = ps_datasets["ew"], ps_datasets["ud"]
        ew_extra_feats, ud_extra_feats = extra_feats["ew"], extra_feats["ud"]

    # Load baseline results
    baseline_labels_ew = np.load(args.baseline_results[0], allow_pickle=True)
    baseline_labels_ud = np.load(args.baseline_results[1], allow_pickle=True)

    # extract only the time series data from the shape file
    if args.online_optimization_data is not None:
        opt_data = extract_ts(
            read_shapefile(args.online_optimization_data, sr, "MultiPoint")
        )
        # apply standardization on time series data
        opt_data = TimeSeriesScalerMeanVariance().fit_transform(opt_data)
    else:
        opt_data = None

    if not args.use_ground_truth:
        ground_truth_labels = None
    else:
        # Extract labels for the ground truth
        if "CLUSTER" in ps_ew.columns:
            ground_truth_labels = ps_ew["CLUSTER"].values
        elif "cluster" in ps_ew.columns:
            ground_truth_labels = ps_ew["cluster"].values
        else:
            raise ValueError(
                "No cluster labels found in the DataFrame (no columns called 'CLUSTER' or 'cluster'."
            )

    return (
        ground_truth_labels,
        x_ew,
        x_ud,
        ew_extra_feats,
        ud_extra_feats,
        baseline_labels_ew,
        baseline_labels_ud,
        opt_data,
        unsupervised_metrics_data,
    )


def main():
    # Load and prepare data
    filename = args.folder_save + "/" + args.dataset_name + "__load__data.pkl"
    if os.path.exists(filename) and not args.old_results:
        with open(filename, "rb") as f:
            (
                ground_truth_labels,
                x_ew,
                x_ud,
                ew_extra_feats,
                ud_extra_feats,
                baseline_labels_ew,
                baseline_labels_ud,
                opt_data,
                unsupervised_metrics_data,
            ) = pickle.load(f)
    else:
        (
            ground_truth_labels,
            x_ew,
            x_ud,
            ew_extra_feats,
            ud_extra_feats,
            baseline_labels_ew,
            baseline_labels_ud,
            opt_data,
            unsupervised_metrics_data,
        ) = load_and_prepare_data()
        with open(filename, "wb") as f:
            pickle.dump(
                (
                    ground_truth_labels,
                    x_ew,
                    x_ud,
                    ew_extra_feats,
                    ud_extra_feats,
                    baseline_labels_ew,
                    baseline_labels_ud,
                    opt_data,
                    unsupervised_metrics_data,
                ),
                f,
            )

    if args.use_extra_features and (ew_extra_feats is None or ud_extra_feats is None):
        print(
            "Extra features are not available in the saved data file. Re-extracting them."
        )
        (
            ground_truth_labels,
            x_ew,
            x_ud,
            ew_extra_feats,
            ud_extra_feats,
            baseline_labels_ew,
            baseline_labels_ud,
            opt_data,
            unsupervised_metrics_data,
        ) = load_and_prepare_data()
        with open(filename, "wb") as f:
            pickle.dump(
                (
                    ground_truth_labels,
                    x_ew,
                    x_ud,
                    ew_extra_feats,
                    ud_extra_feats,
                    baseline_labels_ew,
                    baseline_labels_ud,
                    opt_data,
                    unsupervised_metrics_data,
                ),
                f,
            )
    elif not args.use_extra_features and (
        ew_extra_feats is not None or ud_extra_feats is not None
    ):
        print(
            "Extra features found in the dataset, but they are not used for this run. Removing them."
        )
        ew_extra_feats = None
        ud_extra_feats = None

    n_clusters = args.n_clusters
    n_clusters_ew = n_clusters
    n_clusters_ud = n_clusters
    k_opt_ew = None
    k_opt_ud = None

    if args.overwrite:
        if args.online_cluster_optimization:
            # Determine the optimal number of clusters
            min_k = args.min_n_clusters
            print(
                f"Optimizing the number of clusters{' for EW' if opt_data is None else ''}."
            )
            start_time = time()

            k_opt_ew = online_optimize_clusters(
                opt_data if opt_data is not None else x_ew,
                min_k=min_k,
                thr=args.threshold_val,
                max_patience=args.patience,
                plot_inertia=args.plot_online_optimization_inertia,
            )
            ooc_time_ew = time() - start_time
            print(f"Execution time: {ooc_time_ew:.2f}s")
            k_opt = k_opt_ew
            k_opt_ud = None

            if opt_data is None:
                print("Optimizing the number of clusters for UD.")
                start_time = time()
                k_opt_ud = online_optimize_clusters(
                    x_ud,
                    min_k=min_k,
                    thr=args.threshold_val,
                    max_patience=args.patience,
                    plot_inertia=args.plot_online_optimization_inertia,
                )
                ooc_time_ud = time() - start_time
                print(f"Execution time: {ooc_time_ud:.2f}s")
                k_opt = None

            if opt_data is not None:
                print(f"Optimal number of clusters for EW+UD: {k_opt}")
            else:
                print(f"Optimal number of clusters for EW: {k_opt_ew}")
                print(f"Optimal number of clusters for UD: {k_opt_ud}")

            n_clusters_ew = k_opt_ew

            if opt_data is None:
                n_clusters_ud = k_opt_ud
            else:
                n_clusters_ud = k_opt_ew

            if n_clusters_ew != n_clusters_ud:
                if args.mismatch_opt != "EW":
                    n_clusters_ew = n_clusters_ud

                warnings.warn(
                    "The optimal number of clusters for EW and UD components are different. "
                    "Using "
                    + args.mismatch_opt
                    + " as the optimal number of clusters.",
                    category=UserWarning,
                )

        # Clustering
        results_labels, explanations = compute_clusters(
            methods=args.methods,
            x_datasets=[x_ew, x_ud],
            y_labels=ground_truth_labels,
            random_labels_perc=0.0,
            dataset_names=["EW", "UD"],
            n_lengths=[5, 10, 15],
            n_clusters=n_clusters_ew if n_clusters_ew is not None else n_clusters,
            external_features=[ew_extra_feats, ud_extra_feats],
            explain=True,
            num_runs=args.num_runs,
            dataset_name=args.folder_save
            + "/"
            + args.dataset_name
            + "__saved__models.pkl",
            overwrite_results=True,
            pfa_value=args.pfa_value,
            selection_external=args.selection_extra_features,
            saved_features_extracted=args.folder_save
            + "/"
            + args.dataset_name
            + "__saved__features.pkl",
        )

        # Save results
        with open(
            args.folder_save
            + os.sep
            + args.dataset_name
            + "_"
            + "_".join(
                [
                    f"uxf{args.use_extra_features}",
                    f"sef{args.selection_extra_features}",
                    f"pfa{args.pfa_value}",
                ]
            )
            + "__saved__clusters.pkl",
            "wb",
        ) as results_file:
            pickle.dump(
                results_labels,
                results_file,
            )
        with open(
            args.folder_save
            + os.sep
            + args.dataset_name
            + "_"
            + "_".join(
                [
                    f"uxf{args.use_extra_features}",
                    f"sef{args.selection_extra_features}",
                    f"pfa{args.pfa_value}",
                ]
            )
            + "__saved__explanations.pkl",
            "wb",
        ) as explanations_file:
            pickle.dump(
                explanations,
                explanations_file,
            )

        if args.mismatch_opt == "EW":
            n_clusters = n_clusters_ew
        else:
            n_clusters = n_clusters_ud
    else:
        # Load results
        with open(args.output_filename, "rb") as results_file:
            results_labels = pickle.load(results_file)

        explanations_file = args.output_filename.replace(
            ".pickle", "_explanations.pickle"
        )
        if os.path.exists(explanations_file):
            with open(explanations_file, "rb") as explanations_file:
                explanations = pickle.load(explanations_file)
        else:
            explanations = None

        # get a method that is not None and compute the length of the unique labels list
        for method in results_labels:
            if results_labels[method] is not None:
                # get the first dataset computed
                first_dataset = list(results_labels[method].keys())[0]
                # get the first result from the list of runs
                first_pred_labels = results_labels[method][first_dataset][0][1]
                n_clusters = len(np.unique(first_pred_labels))
                break

    # Compute metrics
    (
        ground_truth_labels,
        baseline_labels_ew,
        baseline_labels_ud,
        results_labels,
    ) = realign_labels(
        ground_truth_labels,
        baseline_labels_ew,
        baseline_labels_ud,
        results_labels,
    )

    if ground_truth_labels is not None:
        data = {
            "ground_truth_labels": ground_truth_labels,
            "results_labels": {n_clusters: results_labels},
            "baseline_labels_ew": baseline_labels_ew,
            "baseline_labels_ud": baseline_labels_ud,
        }
    else:
        data = {
            "ground_truth_labels": ground_truth_labels,
            "results_labels": {n_clusters: results_labels},
        }

    """unsupervised_metrics_data = ew_extra_feats

    # Get columns unique to df2 that are not in df1
    unique_columns_df2 = ew_extra_feats.columns.difference(unsupervised_metrics_data.columns)

    # Concatenate df1 with only unique columns of df2
    unsupervised_metrics_data = pd.concat([unsupervised_metrics_data, ew_extra_feats[unique_columns_df2]], axis=1)"""

    # save the cluster labels in the shape file
    ps_ew = read_shapefile(args.input_data[0], sr, "MultiPoint")
    ps_ud = read_shapefile(args.input_data[1], sr, "MultiPoint")
    for method_ in results_labels:
        for dataset in results_labels[method_]:
            run_results = results_labels[method_][dataset]
            if isinstance(run_results, list):
                run_results: list = run_results[0]

            if "EW" in dataset:
                ps_ew[method_] = run_results[1]
            if "UD" in dataset:
                ps_ud[method_] = run_results[1]

    ps_ew_savefile = (
        args.folder_save
        + os.sep
        + args.dataset_name
        + "_EW_wlabels_"
        + "_".join(
            [
                f"uxf{args.use_extra_features}",
                f"sef{args.selection_extra_features}",
                f"pfa{args.pfa_value}",
            ]
        )
        + ".shp"
    )
    ps_ud_savefile = (
        args.folder_save
        + os.sep
        + args.dataset_name
        + "_UD_wlabels_"
        + "_".join(
            [
                f"uxf{args.use_extra_features}",
                f"sef{args.selection_extra_features}",
                f"pfa{args.pfa_value}",
            ]
        )
        + ".shp"
    )
    print(f"Saving clustering labels in {ps_ew_savefile} and {ps_ud_savefile}")
    ps_ew.to_file(ps_ew_savefile)
    ps_ud.to_file(ps_ud_savefile)

    metrics_results = compute_metrics(
        data,
        supervised_metrics=False,
        euclidean_feats=unsupervised_metrics_data,
        num_runs=args.num_runs,
    )

    baseline_n_clusters = len(np.unique(baseline_labels_ew))
    ground_truth_n_clusters = (
        len(np.unique(ground_truth_labels)) if args.use_ground_truth else None
    )

    save_base = {
        "Dataset": filename,
        "patience": args.patience,
        "threshold": args.threshold_val,
        "k_ew_opt": k_opt_ew,
        "k_ud_opt": k_opt_ud,
        "n_clusters_used": n_clusters,
    }
    list_dict = []
    for clusters_key in metrics_results["methods_results"]:
        for results_method in metrics_results["methods_results"][clusters_key]:
            dict_full = save_base.copy()
            dict_full["method"] = results_method
            if results_method != "time2feat":
                for key in metrics_results["methods_results"][clusters_key][
                    results_method
                ]["EW"].keys():
                    dict_full["EW_" + key] = metrics_results["methods_results"][
                        clusters_key
                    ][results_method]["EW"][key]
                for key in metrics_results["methods_results"][clusters_key][
                    results_method
                ]["UD"].keys():
                    dict_full["UD_" + key] = metrics_results["methods_results"][
                        clusters_key
                    ][results_method]["UD"][key]

                if results_method == "featts":
                    dict_full["pfa_scores"] = args.pfa_value
                    dict_full["use_extra_features"] = args.use_extra_features
                    dict_full["selection_extra_features"] = (
                        args.selection_extra_features
                    )
            else:
                for key in metrics_results["methods_results"][clusters_key][
                    results_method
                ]["EW + UD"].keys():
                    dict_full["EW_UD_" + key] = metrics_results["methods_results"][
                        clusters_key
                    ][results_method]["EW + UD"][key]

                dict_full["pfa_scores"] = args.pfa_value
                dict_full["use_extra_features"] = args.use_extra_features
                dict_full["selection_extra_features"] = args.selection_extra_features

            list_dict.append(dict_full)

    # Step 1: Find all unique keys
    # Convert the new list of dictionaries to a DataFrame
    new_df = pd.DataFrame(list_dict)

    # Specify the CSV file path
    csv_file_path = (
        args.folder_save
        + "/"
        + "results__"
        + dataset_name_file
        + "__"
        + str(n_clusters)
        + "___"
        + args.mismatch_opt
        + ".csv"
    )

    # Step 1: Load existing data if the file exists
    if os.path.exists(csv_file_path):
        existing_df = pd.read_csv(csv_file_path)

        # Step 2: Find all unique keys from both existing and new data
        all_columns = set(existing_df.columns).union(set(new_df.columns))

        # Step 3: Reindex both DataFrames to include all columns, filling missing values with '-'
        existing_df = existing_df.reindex(columns=all_columns, fill_value="-")
        new_df = new_df.reindex(columns=all_columns, fill_value="-")

        # Step 4: Concatenate existing and new data
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        # If the file doesn't exist, just use the new DataFrame
        combined_df = new_df

    # Step 5: Write the updated DataFrame back to the CSV file
    combined_df.to_csv(csv_file_path, index=False)
    """show_and_save_data(
        metrics_results,
        explanations,
        args.n_clusters,
        baseline_n_clusters,
        ground_truth_n_clusters,
        args.print_explanation,
    )"""


if __name__ == "__main__":
    """folder_path = "../data/unlabeled_data/ts_total"
    dataset_name_file = "ts_total"
    name_ew_shape = "combined_ts_2018_2022_trend"
    name_ud_shape = "combined_ts_2018_2022_trend"""

    folder_path = "data/unlabeled_data/offida_data"
    dataset_name_file = "offida_data"
    name_ew_shape = "EGMS_decomposed_10m_average_EW"
    name_ud_shape = "EGMS_decomposed_10m_average_UD"
    sr = "epsg:4326"

    cluster_different_use = "EW"
    # use_extra_features = True
    # selection_extra_features = True
    use_extra_features = True
    selection_extra_features = False
    pfa_value = 0.9
    methods = ["kmeans", "kshape", "time2feat", "featts"]
    # methods = ["featts"]
    min_cluster = 6
    n_clusters = 8
    threshold_val = 0.1
    patience = 10
    online_cluster_optimization = False
    num_runs = 1
    overwrite = True

    args.input_data = [
        os.path.join(folder_path, name_ew_shape + ".shp"),
        os.path.join(folder_path, name_ud_shape + ".shp"),
    ]
    args.sr = sr
    args.use_extra_features = use_extra_features
    args.selection_extra_features = selection_extra_features
    args.pfa_value = pfa_value
    args.methods = methods
    args.min_cluster = min_cluster
    args.threshold_val = threshold_val
    args.patience = patience
    args.mismatch_opt = cluster_different_use
    args.dataset_name = dataset_name_file
    args.folder_save = folder_path
    args.online_cluster_optimization = online_cluster_optimization
    args.n_clusters = 8
    args.plot_online_optimization_inertia = False
    args.num_runs = num_runs
    args.overwrite = overwrite

    """for use_extra_features in [False, True]:
        args.use_extra_features = use_extra_features
        if use_extra_features:
            for pfa_value in [0.5, 0.6, 0.7, 0.8, 0.9]:
                args.pfa_value = pfa_value
                selection_extra_features = False
                args.selection_extra_features = selection_extra_features
                print(args)
                main()
                selection_extra_features = True
                args.selection_extra_features = selection_extra_features
                print(args)
                main()
        else:
            selection_extra_features = False
            args.selection_extra_features = selection_extra_features
            args.pfa_value = 0.9
            print(args)
            main()"""

    """for method in methods:
        for pfa_values in [0.5, 0.6, 0.7, 0.8, 0.9]:
            for use_extra_feature in [True, False]:
                for selection_extra_feature in [True, False]:
                    if use_extra_feature == False and selection_extra_feature == True:
                        continue
                    else:
                        args.input_data = [
                            folder_path + "/" + name_ew_shape + ".shp",
                            folder_path + "/" + name_ud_shape + ".shp",
                        ]
                        args.use_extra_features = use_extra_feature
                        args.selection_extra_features = selection_extra_feature
                        args.pfa_value = pfa_values
                        args.methods = methods
                        args.min_cluster = min_cluster
                        args.threshold_val = threshold_val
                        args.patience = patience
                        args.mismatch_opt = cluster_different_use
                        args.name_dataset = dataset_name_file
                        args.folder_save = folder_path
                        args.n_clusters = n_clusters
                        print(args)
                        main()
                        if method == "kshape" or method == "kmeans":
                            break
                if method == "kshape" or method == "kmeans":
                    break
            if method == "kshape" or method == "kmeans":
                break"""

    for method in methods:
        # for pfa_values in [0.5, 0.6, 0.7, 0.8, 0.9]:
        for pfa_values in [0.9]:
            for use_extra_feature in [True]:
                for selection_extra_feature in [False]:
                    if use_extra_feature == False and selection_extra_feature == True:
                        continue
                    else:
                        args.input_data = [
                            folder_path + "/" + name_ew_shape + ".shp",
                            folder_path + "/" + name_ud_shape + ".shp",
                        ]
                        args.use_extra_features = use_extra_feature
                        args.selection_extra_features = selection_extra_feature
                        args.pfa_value = pfa_values
                        args.methods = methods
                        args.min_cluster = min_cluster
                        args.threshold_val = threshold_val
                        args.patience = patience
                        args.mismatch_opt = cluster_different_use
                        args.name_dataset = dataset_name_file
                        args.folder_save = folder_path
                        args.n_clusters = n_clusters
                        print(args)
                        main()
