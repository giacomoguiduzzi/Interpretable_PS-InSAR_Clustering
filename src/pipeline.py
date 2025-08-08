import os
import pickle
import warnings

from kneed import KneeLocator
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.tests.test_x13 import dataset
from tslearn.clustering import KShape

from FeatTS.FeatTS.FeatTS import FeatTS
from time2feat.time2feat.time2feat import Time2Feat
from sklearn.cluster import MiniBatchKMeans, KMeans
from typing import Sequence, Optional, Union
import numpy as np
import pandas as pd
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from matplotlib import pyplot as plt
import seaborn as sns

import clustering_metrics
from src.utils import select_random_percent, get_idle_cpu_cores
from geoutils import read_shapefile, extract_ts
from plotting import plot_ts_clusters


def load_and_prepare_data(args):
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
        warnings.warn(
            "Only one baseline results file provided. Assuming it is the same for both components.",
            category=UserWarning,
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

    if not args.overwrite:
        print("Loading ground truth labels...")
    else:
        print("Loading data...")

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

        ps_ew_ud = None

    else:
        if args.input_data != [
            "data/unlabeled_data/EGMS_EW_18_22_3857.shp",
            "data/unlabeled_data/EGMS_UD_18_22_3857.shp",
        ]:  # if the specified data is not the default data
            print("Using specified data.")
        else:
            print("Using default data.")

        if len(args.input_data) == 1:
            ps_ew_ud = read_shapefile(args.input_data[0], sr, "MultiPoint")
            ps_ew = None
            ps_ud = None

            if "tpi" in ps_ew_ud.columns:
                ps_ew_ud.rename(columns={"tpi": "twi"}, inplace=True)

        else:
            ps_ew_ud = None
            ps_ew = read_shapefile(args.input_data[0], sr, "MultiPoint")
            ps_ud = read_shapefile(args.input_data[1], sr, "MultiPoint")

            # Fix a typo in a column
            if "tpi" in ps_ew.columns:
                ps_ew.rename(columns={"tpi": "twi"}, inplace=True)
                ps_ud.rename(columns={"tpi": "twi"}, inplace=True)

    # init to avoid undefined variable error
    ew_ud_extra_feats = None
    ew_extra_feats = None
    ud_extra_feats = None

    if args.use_extra_features:
        # Extract extra features
        # Excluding the categorical external features which are not suitable to compute a Euclidean metric

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

        try:
            if len(args.input_data) == 1:
                ew_ud_extra_feats = ps_ew_ud[
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
                ew_extra_feats = None
                ud_extra_feats = None
            else:
                ew_ud_extra_feats = None
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
        if len(args.input_data) == 1:
            ew_ud_extra_feats = ps_ew_ud[["LAT", "LON"]]
            ew_extra_feats = None
            ud_extra_feats = None
        else:
            ew_ud_extra_feats = None
            ew_extra_feats = ps_ew[["LAT", "LON"]]
            ud_extra_feats = ps_ud[["LAT", "LON"]]

    # Calculate lat e lon
    if len(args.input_data) == 1:
        ew_ud_extra_feats["LAT"] = ew_ud_extra_feats.geometry.y
        ew_ud_extra_feats["LON"] = ew_ud_extra_feats.geometry.x
    else:
        ps_ew["LAT"] = ps_ew.geometry.y
        ps_ew["LON"] = ps_ew.geometry.x
        ps_ud["LAT"] = ps_ew.geometry.y
        ps_ud["LON"] = ps_ew.geometry.x

    # Remove unnecessary columns
    if len(args.input_data) == 1:
        ps_ew_ud_2d = ps_ew_ud.drop(
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
        ps_ew_2d = None
        ps_ud_2d = None

    else:
        ps_ew_ud_2d = None
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
    if len(args.input_data) == 1:
        ps_ew_ud_2d = ps_ew_ud_2d[
            ["LAT", "LON"]
            + [col for col in ps_ew_ud_2d.columns if col not in ["LAT", "LON"]]
        ]
    else:
        ps_ew_2d = ps_ew_2d[
            ["LAT", "LON"]
            + [col for col in ps_ew_2d.columns if col not in ["LAT", "LON"]]
        ]
        ps_ud_2d = ps_ud_2d[
            ["LAT", "LON"]
            + [col for col in ps_ud_2d.columns if col not in ["LAT", "LON"]]
        ]

    # init to avoid undefined variable error
    x_ew_ud, x_ew, x_ud = None, None, None

    if args.overwrite or not os.path.exists(args.output_filename):
        # Extract EW and UD time series
        if len(args.input_data) == 1:
            ts_data = extract_ts(ps_ew_ud_2d).copy().T
            x_ew_ud = ts_data.copy().T.values

            if x_ew_ud.dtype == np.object_:
                x_ew_ud = x_ew_ud.astype(np.float32)

        else:
            x_ew_ud = None
            ts_data = extract_ts(ps_ew_2d).copy().T

        if not args.baseline_shapes and not args.use_extra_features:
            if args.plot_clusters:
                cluster_labels = ps_ew_2d["CLUSTER"].values
                cluster_ids = np.unique(cluster_labels)

                plot_ts_clusters(cluster_ids, cluster_labels, ts_data)

            x_ew = ts_data.copy()

            x_ew = x_ew.T.values

        else:
            x_ew = ts_data.copy().T.values

        # --- UD ---
        if len(args.input_data) > 1:
            ts_data = extract_ts(ps_ud_2d).copy().T
            if not args.baseline_shapes and not args.use_extra_features:
                # ground_truth_labels = ps_ud_2d["CLUSTER"].values
                # cluster_ids = np.unique(ground_truth_labels)

                # plot_ts_clusters(cluster_ids, ground_truth_labels, ts_data)

                x_ud = ts_data.copy()
                x_ud = x_ud.T.values

            else:
                x_ud = ts_data.copy().T.values
            if x_ew.dtype == np.object_:
                x_ew = x_ew.astype(np.float32)
            if x_ud.dtype == np.object_:
                x_ud = x_ud.astype(np.float32)

        # X_EW_orig = x_ew.copy()
        # X_UD_orig = x_ud.copy()

        # Scale the data
        if len(args.input_data) == 1:
            x_ew_ud = scale_dataset(x_ew_ud)
            x_ew, x_ud = None, None
        else:
            x_ew_ud = None
            x_ew, x_ud = scale_dataset(x_ew, x_ud)

    # Scale extra features
    if len(args.input_data) == 1:
        ps_ew_ud_2d, ew_ud_extra_feats = scale_extra_features(
            {"ew_ud": ps_ew_ud_2d},
            {"ew_ud": ew_ud_extra_feats},
        )
    else:
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

    if (not args.overwrite and os.path.exists(args.output_filename)) or (
        args.overwrite and not os.path.exists(args.output_filename)
    ):
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

        if len(args.input_data) == 1:
            return (
                ground_truth_labels,
                ew_ud_extra_feats,
                baseline_labels_ew,
                baseline_labels_ud,
            )
        else:
            return (
                ground_truth_labels,
                ew_extra_feats,
                baseline_labels_ew,
                baseline_labels_ud,
            )
    else:
        if len(args.input_data) == 1:
            return (
                x_ew_ud,
                ps_ew_ud_2d,
                ew_ud_extra_feats,
                baseline_labels_ew,
                baseline_labels_ud,
                opt_data,
            )
        else:
            return (
                x_ew,
                x_ud,
                ps_ew_2d,
                ps_ud_2d,
                ew_extra_feats,
                ud_extra_feats,
                baseline_labels_ew,
                baseline_labels_ud,
                opt_data,
            )


def scale_extra_features(ps_datasets: dict, extra_feats: dict):
    """
    Scale the points' coordinates and the extra features.
    Note that both dictionaries in input need to contain the same keys.
    :param ps_datasets: A dictionary containing the datasets as pd.DataFrames with the points defining the time series.
    :param extra_feats: A dictionary containing the datasets as pd.DataFrames with the extra features.
    :return: The scaled datasets as dictionaries, exactly as they were given in input.
    """
    for dataset_name in ps_datasets.keys():
        ps_dataset = ps_datasets[dataset_name]
        dataset_extra_feats = extra_feats[dataset_name]

        if dataset_extra_feats is not None:
            dataset_extra_feats = dataset_extra_feats.copy()

        ps_dataset[["LAT", "LON"]] = MinMaxScaler().fit_transform(
            ps_dataset[["LAT", "LON"]]
        )
        # apply minmax on extra features: LAT and LON together, every other feature by itself

        if (
            dataset_extra_feats is not None
            and "LAT" in dataset_extra_feats.columns
            and "LON" in dataset_extra_feats.columns
        ):
            dataset_extra_feats.loc[:, ["LAT", "LON"]] = MinMaxScaler().fit_transform(
                dataset_extra_feats.loc[:, ["LAT", "LON"]].copy()
            )
            for feat in dataset_extra_feats.columns:
                if feat in ["LAT", "LON"]:
                    continue

                dataset_extra_feats[feat] = (
                    MinMaxScaler()
                    .fit_transform(dataset_extra_feats[[feat]].values)
                    .astype(np.float64)
                )

        ps_datasets[dataset_name] = ps_dataset
        extra_feats[dataset_name] = dataset_extra_feats

    return ps_datasets, extra_feats


def scale_dataset(*args):
    # Scale data
    scaler = TimeSeriesScalerMeanVariance()
    print("Scaling data with", scaler.__class__.__name__)

    return [scaler.fit_transform(dataset) for dataset in args]


def online_optimize_clusters(
    dataset, thr=0.5, min_k=10, max_patience=10, plot_inertia: bool = False
) -> int:
    def average_difference(list_: list):
        """
        Compute the average of the difference between consecutive elements in a list.

        Args:
        x (list): A list of numbers.

        Returns:
        float: The average difference between consecutive elements.
        """
        if len(list_) < 2:
            raise ValueError("List must have at least two elements.")

        differences = [list_[i - 1] - list_[i] for i in range(1, len(list_))]
        return sum(differences) / len(differences)

    patience = 0
    best_diff = 0
    k = min_k  # Start with the minimum number of clusters
    best_k = k
    inertia_values = list()
    patience_values = list()
    best_k_found_list = list()
    while patience < max_patience:
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42, n_init="auto")
        kmeans.fit(np.squeeze(dataset, axis=2))
        inertia = kmeans.inertia_
        if len(inertia_values) > 1:
            avg = average_difference(inertia_values)
            if inertia_values[-1] - inertia > ((thr * avg) + avg):
                diff = inertia_values[-1] - inertia - ((thr * avg) + avg)
                if best_diff == 0:
                    best_diff = diff
                    best_k = k

                if patience > 0 and best_diff < diff:
                    patience = 0
                    best_diff = diff
                    best_k = k
                    print("New best number of clusters: ", best_k)
                    print("Inertia: ", inertia)

            else:
                patience += 1

        inertia_values.append(inertia)
        best_k_found_list.append(best_k)
        patience_values.append(patience)
        k += 1

    kl = KneeLocator(
        range(min_k, k), inertia_values, curve="convex", direction="decreasing"
    )
    print("Best K found through elbow method is", kl.elbow)

    if plot_inertia:
        data = pd.DataFrame(
            {
                "inertia": inertia_values,
                "k": range(min_k, k),
                "patience": patience_values,
            }
        )
        _, ax = plt.subplots(figsize=(16, 9))
        sns.lineplot(data, x="k", y="inertia", ax=ax, label="Inertia")
        ax2 = ax.twinx()
        sns.lineplot(
            data, x="k", y="patience", ax=ax2, label="Patience", color="orange"
        )
        # filter the best k values to scatter-plot them
        seen = set()
        for idx, item in enumerate(best_k_found_list[:]):
            if item in seen:
                best_k_found_list[idx] = np.nan
            else:
                seen.add(item)

        # get inertia values for the best k values
        best_inertia = list()
        for idx in range(len(best_k_found_list)):
            if not np.isnan(best_k_found_list[idx]):
                best_inertia.append(inertia_values[idx])
            else:
                best_inertia.append(np.nan)

        data = pd.DataFrame(
            {
                "best_k_found": best_inertia,
                "k": range(min_k, k),
            }
        )
        sns.scatterplot(
            data, x="k", y="best_k_found", ax=ax, label="Best K found", color="green"
        )
        plt.title("Inertia, Patience and Best K values")
        ax.set_xlabel("Number of clusters")
        ax2.set_xlabel("Number of clusters")
        ax.set_ylabel("Inertia")
        ax2.set_ylabel("Patience")
        plt.legend()
        plt.show()

    return best_k


'''def online_optimize_clusters_original(x, thr=0.5, min_k=10, pat_max=10):
    def average_difference(list_: list):
        """
        Compute the average of the difference between consecutive elements in a list.

        Args:
        x (list): A list of numbers.

        Returns:
        float: The average difference between consecutive elements.
        """
        if len(list_) < 2:
            raise ValueError("List must have at least two elements.")

        differences = [list_[i - 1] - list_[i] for i in range(1, len(list_))]
        return sum(differences) / len(differences)

    pat = 0
    flag = 0
    diff_best = 0
    i = min_k
    best_i = 0
    inertia_values = []
    while pat < pat_max:
        kmeans = KMeans(n_clusters=i, random_state=42, n_init="auto")
        kmeans.fit(np.squeeze(x, axis=2))
        inertia = kmeans.inertia_
        if len(inertia_values) > 1:
            avg = average_difference(inertia_values)
            if inertia_values[-1] - inertia > ((thr * avg) + avg):
                flag = 1
                diff = inertia_values[-1] - inertia - ((thr * avg) + avg)
                if pat > 0 and diff_best < diff:
                    pat = 0
                    diff_best = diff
                    best_i = i
                elif diff_best == 0:
                    diff_best = diff
        inertia_values.append(inertia)
        if flag == 1:
            pat += 1
        i += 1
    return best_i'''


def prepare_data_for_clustering(
    x_datasets: Sequence[np.ndarray],
    y_labels: Optional[Sequence[np.ndarray]],
    dataset_names: Sequence[str],
    external_features: Optional[Sequence[pd.DataFrame]] = None,
    n_clusters: Optional[int] = None,
    random_labels_perc: float = 0.2,
) -> dict:
    if y_labels is None or all(label is None for label in y_labels):
        y_labels = [None] * len(x_datasets)

    if y_labels and not all(label is None for label in y_labels):
        num_clusters = len(np.unique(y_labels[0])) if not n_clusters else n_clusters
    else:
        num_clusters = n_clusters

    if external_features is None or all(
        ext_feat is None for ext_feat in external_features
    ):
        external_features = [None] * len(x_datasets)

    elif isinstance(external_features, (pd.DataFrame, list)):
        if isinstance(external_features, pd.DataFrame):
            external_features = [external_features.copy()] * len(x_datasets)
        else:
            external_features = [ext_feat.copy() for ext_feat in external_features]

    data = {
        "x_datasets": x_datasets,
        "y_labels": y_labels,
        "dataset_names": dataset_names,
        "external_features": external_features,
        "num_clusters": num_clusters,
        "random_labels_perc": random_labels_perc,
    }

    return data


def run_models(
    methods: Sequence[str],
    data: dict,
    num_runs: int,
    explain: bool,
    jobs: int,
    n_lengths: Optional[list] = None,
    pfa_value: float = 0.9,
    selection_external: bool = False,
    save_features_extracted: str = "",
) -> dict:
    methods: list

    methods_labels = {
        method: {dataset_name: list() for dataset_name in data["dataset_names"]}
        for method in methods
        if method != "time2feat"
    }
    if "time2feat" in methods:
        methods_labels.update({"time2feat": {"EW + UD": list()}})

    explanation = None

    for method in methods:
        if method.startswith("time2feat"):
            if any(len(dataset.shape) > 2 for dataset in data["x_datasets"]):
                combined_signal = np.stack(
                    [signal.squeeze() for signal in data["x_datasets"]], axis=-1
                )
            else:
                combined_signal = np.stack(
                    [signal for signal in data["x_datasets"]], axis=-1
                )

            y = data["y_labels"][0]

            for nr in range(num_runs):
                print("Run", nr + 1)
                print(f"Time2Feat on EW + UD with Hierarchical model")
                model = Time2Feat(
                    n_clusters=data["num_clusters"],
                    p=jobs,
                    model_type="Hierarchical",
                    pfa_value=pfa_value,
                )
                if data["random_labels_perc"] > 0.0:
                    semi_supervision_labels = select_random_percent(
                        y, data["random_labels_perc"]
                    )
                else:
                    semi_supervision_labels = None
                y_pred = model.fit_predict(
                    combined_signal,
                    labels=semi_supervision_labels,
                    external_feat=data["external_features"][0],
                    select_external_feat=selection_external,
                    save_features_extracted=save_features_extracted,
                )
                methods_labels[method]["EW + UD"].append(
                    [y, y_pred] if explain else y_pred
                )
                if model.top_ext_feats and selection_external:
                    print(
                        "Time2Feat top external features: ",
                        ", ".join(model.top_ext_feats),
                    )
                if explain:
                    explanation = {
                        "ts_feats": model.top_feats,
                        "ext_feats": model.top_ext_feats,
                        "ts_feats_variance": model.top_feats_variance,
                        "ext_feats_variance": model.top_ext_feats_variance,
                    }
        else:
            for dataset_name, X, y, ext_feats in zip(
                data["dataset_names"],
                data["x_datasets"],
                data["y_labels"],
                data["external_features"],
            ):
                if method.lower() == "kshape":
                    print(f"KShape on {dataset_name}")
                    x_3d = (
                        X.reshape(X.shape[0], X.shape[1], 1) if len(X.shape) <= 2 else X
                    )
                    n_clusters = data["num_clusters"]
                    if n_clusters is None:
                        raise ValueError(
                            "The number of clusters must be specified for KShape."
                        )
                    pred_labels = KShape(
                        n_clusters=n_clusters, verbose=True
                    ).fit_predict(x_3d)
                    methods_labels[method][dataset_name].append(
                        [y, pred_labels] if explain else pred_labels
                    )
                elif method.lower() == "kmeans":
                    print(f"KMeans on {dataset_name}")
                    x_2d = X.reshape(X.shape[0], X.shape[1]) if len(X.shape) > 2 else X
                    n_clusters = data["num_clusters"]
                    if n_clusters is None:
                        raise ValueError(
                            "The number of clusters must be specified for KMeans."
                        )
                    pred_labels = KMeans(
                        n_clusters=n_clusters, verbose=False, n_init="auto"
                    ).fit_predict(x_2d)
                    methods_labels[method][dataset_name].append(
                        [y, pred_labels] if explain else pred_labels
                    )
                elif method.lower() == "featts":
                    print(f"FeatTS on {dataset_name}")
                    for nr in range(num_runs):
                        print("Run", nr + 1)
                        model = FeatTS(
                            n_clusters=data["num_clusters"],
                            n_jobs=jobs,
                            community_detection_jobs=jobs,
                            pfa_ext_feats=selection_external,
                            pfa_value=pfa_value,
                        )
                        if len(X.shape) > 2:
                            X = np.squeeze(X, axis=-1)

                        if data["random_labels_perc"] > 0.0:
                            semi_supervision_labels = select_random_percent(
                                y, data["random_labels_perc"]
                            )
                        else:
                            semi_supervision_labels = None

                        y_pred = model.fit_predict(
                            X, labels=semi_supervision_labels, external_feat=ext_feats
                        )
                        methods_labels[method][dataset_name].append(
                            [y, y_pred] if explain else y_pred
                        )
                        if explain:
                            explanation = {
                                "ts_feats": model.feats_selected_,
                                "ext_feats": model.ext_feats_selected_,
                                "ts_feats_variance": model.feats_variance_,
                                "ext_feats_variance": model.ext_feats_variance_,
                            }
                else:
                    raise NotImplementedError(f"Method {method} not implemented")

    return {"methods_labels": methods_labels, "explanation": explanation}


def compute_metrics(
    methods_results: dict,
    supervised_metrics: bool,
    euclidean_feats: Optional[pd.DataFrame],
    kneighbors_percentage: float = 0.9,
    return_mlrd_per_cluster: bool = False,
    num_runs: int = 10,
) -> Union[tuple[dict, dict], dict]:
    """
    Compute clustering metrics for the given methods results
    :param methods_results: The results of the clustering methods.
    :param supervised_metrics: Whether to compute supervised metrics.
    :param euclidean_feats: The Euclidean features to compute the LRD from.
    :param kneighbors_percentage: The percentage of points to consider in a cluster for the LRD computation.
    :param return_mlrd_per_cluster: Whether to return the MLRD score per cluster.
    :param num_runs: The number of runs used to compute the metrics so to compute the mean value.
    :return: The clustering metrics for the given methods results.

    methods_results: {
        'ground_truth_labels': array(...),
        'results_labels': {
            2: {
                'kmeans': {
                    'EW': [
                        [
                            None,
                            array(...)
                        ]
                    ],
                    'UD': [
                        [
                            None,
                            array(...)
                        ]
                    ]
                },
                'kshape': {
                    'EW': [
                        [
                            None,
                            array(...)
                        ]
                    ],
                    'UD': [
                        [
                            None,
                            array(...)
                        ]
                    ]
                },
                'featts': {
                    'EW': [
                        [
                            None,
                            array(...)
                        ]
                    ],
                    'UD': [
                        [
                            None,
                            array(...)
                        ]
                    ]
                },
                'time2feat': {
                    'EW + UD': [
                        [
                            None,
                            array(...)
                        ]
                    ]
                }
            },
            3: {
                ...
            }
        },
        'baseline_labels_ew': array(...),
        'baseline_labels_ud': array(...)
    }

    euclidean_feats: pd.DataFrame(997x9): ['LAT', 'LON', 'dem', 'slope', 'aspect', 'twi', 'c_tot', 'c_plan', 'c_prof']
    """
    metrics = (
        clustering_metrics.supervised
        if supervised_metrics
        else clustering_metrics.unsupervised
    )

    # Remove methods without results
    for cluster_value in methods_results["results_labels"]:
        for method in list(methods_results["results_labels"][cluster_value].keys()):
            if methods_results["results_labels"][cluster_value][method] is None:
                del methods_results["results_labels"][cluster_value][method]

    # get the dataset names per method
    n_clusters = {
        labels_name: list(methods_results["results_labels"].keys())
        for labels_name in methods_results.keys()
    }

    # check if all methods have the same number of clusters and labels
    for labels_name, cluster_values in n_clusters.items():
        if labels_name == "ground_truth_labels":
            continue
        if not np.array_equal(cluster_values, n_clusters["ground_truth_labels"]):
            print(
                "The number of clusters is not the same for all methods or the labels are not aligned:"
            )
            for labels_name_, cluster_values_ in n_clusters.items():
                print(f"{labels_name_}: {cluster_values_}")

            raise ValueError(
                "The number of clusters is not the same for all methods or the labels are not aligned."
            )

    n_clusters = n_clusters["ground_truth_labels"]

    methods = [
        result_label
        for result_label in methods_results.keys()
        if result_label != "results_labels"
    ] + list(methods_results["results_labels"][n_clusters[0]].keys())

    # create a list with unique dataset names per method
    dataset_names_by_method = {
        "ground_truth_labels": ["EW + UD"],
        "baseline_labels_ew": ["EW"],
        "baseline_labels_ud": ["UD"],
    }
    dataset_names_by_method.update(
        {
            method: list(
                methods_results["results_labels"][n_clusters[0]][method].keys()
            )
            for method in methods_results["results_labels"][n_clusters[0]]
        }
    )

    # prepare the results data structure
    methods_metrics = {
        cluster_value: {
            method: {
                dataset_name: {metric.__name__: float("inf") for metric in metrics}
                for dataset_name in dataset_names_by_method[method]
            }
            for method in methods
            if method
            not in ["ground_truth_labels", "baseline_labels_ew", "baseline_labels_ud"]
        }
        for cluster_value in n_clusters
    }

    ground_truth_metrics = {
        dataset_name: {metric.__name__: float("inf") for metric in metrics}
        for dataset_name in dataset_names_by_method["ground_truth_labels"]
    }

    baseline_metrics_ew = {
        dataset_name: {metric.__name__: float("inf") for metric in metrics}
        for dataset_name in dataset_names_by_method["baseline_labels_ew"]
    }

    baseline_metrics_ud = {
        dataset_name: {metric.__name__: float("inf") for metric in metrics}
        for dataset_name in dataset_names_by_method["baseline_labels_ud"]
    }

    mlrd_per_cluster = (
        {labels_name: dict() for labels_name in methods_results.keys()}
        if return_mlrd_per_cluster
        else None
    )

    for labels_name in methods_results:
        if labels_name not in [
            "ground_truth_labels",
            "baseline_labels_ew",
            "baseline_labels_ud",
        ]:
            current_method_metrics = methods_metrics
            current_mlrd_per_cluster = dict() if return_mlrd_per_cluster else None
            for cluster_value in current_method_metrics:
                for method in current_method_metrics[cluster_value]:
                    for dataset_name in methods_results["results_labels"][
                        cluster_value
                    ][method]:
                        for metric in metrics:
                            metrics_sum = 0
                            clusterings = methods_results["results_labels"][
                                cluster_value
                            ][method][dataset_name]
                            if supervised_metrics:
                                for true_labels, pred_labels in clusterings:
                                    metrics_sum += metric(true_labels, pred_labels)
                            else:
                                for _, pred_labels in clusterings:
                                    if metric == clustering_metrics.leaky_relu_mlrd:

                                        return_value = clustering_metrics.leaky_relu_mlrd(
                                            pred_labels,
                                            euclidean_feats,
                                            k_perc=kneighbors_percentage,
                                            return_mlrd_per_cluster=return_mlrd_per_cluster,
                                        )

                                        if return_mlrd_per_cluster:
                                            current_mlrd_per_cluster.update(
                                                return_value[1]
                                            )
                                            metrics_sum += return_value[0]
                                        else:
                                            metrics_sum += return_value

                                    elif metric == clustering_metrics.dunn:
                                        # reorganize data to fit the dunn function:
                                        # k_list – A list containing a numpy array for each cluster
                                        # |c| = number of clusters
                                        # c[K] is np.array([N, p]) (N : number of samples in cluster K, p: sample dimension)
                                        k_list = [
                                            euclidean_feats.loc[
                                                pred_labels == cluster, :
                                            ].values
                                            for cluster in sorted(
                                                np.unique(pred_labels)
                                            )
                                        ]

                                        metrics_sum += metric(k_list)

                                    else:
                                        metrics_sum += metric(
                                            euclidean_feats.values, pred_labels
                                        )
                            result = metrics_sum / num_runs

                            current_method_metrics[cluster_value][method][dataset_name][
                                metric.__name__
                            ] = result

            methods_metrics = current_method_metrics
            if return_mlrd_per_cluster:
                mlrd_per_cluster[labels_name] = current_mlrd_per_cluster

        else:
            if (
                labels_name == "ground_truth_labels"
                and methods_results[labels_name] is not None
            ):
                true_labels = methods_results[labels_name]
                for dataset_name in ground_truth_metrics:
                    if not supervised_metrics:
                        for metric in metrics:
                            if metric == clustering_metrics.leaky_relu_mlrd:
                                return_value = clustering_metrics.leaky_relu_mlrd(
                                    true_labels,
                                    euclidean_feats,
                                    k_perc=kneighbors_percentage,
                                    return_mlrd_per_cluster=return_mlrd_per_cluster,
                                )

                                if return_mlrd_per_cluster:
                                    mlrd_per_cluster[labels_name].update(
                                        return_value[1]
                                    )
                                    result = return_value[0]
                                else:
                                    result = return_value

                                ground_truth_metrics[dataset_name][
                                    metric.__name__
                                ] = result

                            elif metric == clustering_metrics.dunn:
                                # reorganize data to fit the dunn function:
                                # k_list – A list containing a numpy array for each cluster
                                # |c| = number of clusters
                                # c[K] is np.array([N, p]) (N : number of samples in cluster K, p: sample dimension)
                                k_list = [
                                    euclidean_feats.loc[
                                        true_labels == cluster, :
                                    ].values
                                    for cluster in sorted(np.unique(true_labels))
                                ]

                                result = metric(k_list)
                                ground_truth_metrics[dataset_name][
                                    metric.__name__
                                ] = result
                            else:
                                result = metric(euclidean_feats.values, true_labels)
                                ground_truth_metrics[dataset_name][
                                    metric.__name__
                                ] = result

            elif labels_name == "baseline_labels_ew":
                true_labels = methods_results["baseline_labels_ew"]
                for dataset_name in baseline_metrics_ew:
                    for metric in metrics:
                        if metric == clustering_metrics.leaky_relu_mlrd:
                            return_value = clustering_metrics.leaky_relu_mlrd(
                                true_labels,
                                euclidean_feats,
                                k_perc=kneighbors_percentage,
                                return_mlrd_per_cluster=return_mlrd_per_cluster,
                            )

                            if return_mlrd_per_cluster:
                                mlrd_per_cluster[labels_name].update(return_value[1])
                                result = return_value[0]
                            else:
                                result = return_value

                            baseline_metrics_ew[dataset_name][metric.__name__] = result
                        elif metric == clustering_metrics.dunn:
                            # reorganize data to fit the dunn function:
                            # k_list – A list containing a numpy array for each cluster
                            # |c| = number of clusters
                            # c[K] is np.array([N, p]) (N : number of samples in cluster K, p: sample dimension)
                            k_list = [
                                euclidean_feats.loc[true_labels == cluster, :].values
                                for cluster in sorted(np.unique(true_labels))
                            ]

                            result = metric(k_list)
                            baseline_metrics_ew[dataset_name][metric.__name__] = result
                        else:
                            result = metric(euclidean_feats.values, true_labels)
                            baseline_metrics_ew[dataset_name][metric.__name__] = result

            elif labels_name == "baseline_labels_ud":
                true_labels = methods_results["baseline_labels_ud"]
                for dataset_name in baseline_metrics_ud:
                    for metric in metrics:
                        if metric == clustering_metrics.leaky_relu_mlrd:
                            return_value = clustering_metrics.leaky_relu_mlrd(
                                true_labels,
                                euclidean_feats,
                                k_perc=kneighbors_percentage,
                                return_mlrd_per_cluster=return_mlrd_per_cluster,
                            )

                            if return_mlrd_per_cluster:
                                mlrd_per_cluster[labels_name].update(return_value[1])
                                result = return_value[0]
                            else:
                                result = return_value

                            baseline_metrics_ud[dataset_name][metric.__name__] = result
                        elif metric == clustering_metrics.dunn:
                            # reorganize data to fit the dunn function:
                            # k_list – A list containing a numpy array for each cluster
                            # |c| = number of clusters
                            # c[K] is np.array([N, p]) (N : number of samples in cluster K, p: sample dimension)
                            k_list = [
                                euclidean_feats.loc[true_labels == cluster, :].values
                                for cluster in sorted(np.unique(true_labels))
                            ]

                            result = metric(k_list)
                            baseline_metrics_ud[dataset_name][metric.__name__] = result
                        else:
                            result = metric(euclidean_feats.values, true_labels)
                            baseline_metrics_ud[dataset_name][metric.__name__] = result

    if "baseline_labels_ew" in list(
        methods_results.keys()
    ) and "baseline_labels_ud" in list(methods_results.keys()):
        methods_metrics = {
            "ground_truth_labels": ground_truth_metrics,
            "baseline_labels_ew": baseline_metrics_ew,
            "baseline_labels_ud": baseline_metrics_ud,
            "methods_results": methods_metrics,
        }
    else:
        methods_metrics = {
            "ground_truth_labels": ground_truth_metrics,
            "methods_results": methods_metrics,
        }

    if return_mlrd_per_cluster:
        return methods_metrics, mlrd_per_cluster
    else:
        return methods_metrics


def compute_clusters(
    methods: Sequence[str],
    x_datasets: Sequence[np.ndarray],
    y_labels: Optional[Sequence[np.ndarray]],
    dataset_names: Sequence[str],
    n_lengths: Optional[Sequence[int]] = None,
    external_features: Optional[Sequence[pd.DataFrame]] = None,
    random_labels_perc: float = 0.2,
    num_runs: int = 1,
    explain: bool = False,
    n_clusters: Optional[int] = None,
    dataset_name: str = "",
    overwrite_results: bool = False,
    pfa_value: float = 0.9,
    selection_external=False,
    saved_features_extracted: str = "",
) -> Union[dict, Sequence]:
    data = prepare_data_for_clustering(
        x_datasets,
        y_labels,
        dataset_names,
        external_features,
        n_clusters,
        random_labels_perc,
    )
    jobs = max(1, get_idle_cpu_cores())
    print("Running clustering methods with", jobs, "workers.")
    if os.path.exists(dataset_name):
        if overwrite_results:
            results = run_models(
                methods,
                data,
                num_runs,
                explain,
                jobs,
                n_lengths=n_lengths,
                pfa_value=pfa_value,
                selection_external=selection_external,
                save_features_extracted=saved_features_extracted,
            )
            pickle.dump(results, open(dataset_name, "wb"))
    else:
        results = run_models(
            methods,
            data,
            num_runs,
            explain,
            jobs,
            n_lengths=n_lengths,
            pfa_value=pfa_value,
            selection_external=selection_external,
            save_features_extracted=saved_features_extracted,
        )
        pickle.dump(results, open(dataset_name, "wb"))

    results = pickle.load(open(dataset_name, "rb"))

    return_values = [results["methods_labels"]]

    if explain:
        return_values.append(results["explanation"])
    else:
        return_values.append(None)

    return return_values


def run_experiments(
    x_ew: Optional[np.ndarray] = None,
    ps_ew_2d: Optional[pd.DataFrame] = None,
    methods: Optional[list[str]] = None,
    x_ud: Optional[np.ndarray] = None,
    ps_ud_2d: Optional[pd.DataFrame] = None,
    labels: Optional[list[int]] = None,
    n_clusters: Optional[int] = None,
    supervised_metrics: bool = True,
    ext_feats: Union[list[str], pd.DataFrame] = None,
    num_runs: int = 10,
    merged_signals: bool = False,
) -> tuple[dict, dict]:
    if x_ew is None and x_ud is None:
        raise ValueError(
            "At least one dataset must be provided to run the experiments."
        )

    if not methods:
        raise ValueError("At least one method must be provided to run the experiments.")

    if isinstance(ext_feats, list):
        ext_feats = ps_ew_2d[ext_feats] if ext_feats is not None else None

    if supervised_metrics:
        if labels is not None:
            y_labels = [labels, labels] if x_ud is not None else [labels]

        else:
            if "CLUSTER" in ps_ew_2d.columns:
                y_labels = [ps_ew_2d["CLUSTER"].values, ps_ud_2d["CLUSTER"].values]
            elif "cluster" in ps_ew_2d.columns:
                y_labels = [ps_ew_2d["cluster"].values, ps_ud_2d["cluster"].values]
            else:
                raise ValueError(
                    'No cluster labels found in the DataFrame (no columns called "CLUSTER" or "cluster".'
                )
    else:
        y_labels = None

    print(
        f"Running {', '.join(methods)} for n_clusters = {n_clusters} ", sep="", end=""
    )
    print(f"(from cluster labels)" if n_clusters is None else "")
    print("Number of runs:", num_runs)
    if merged_signals:
        results_labels, explanations = compute_clusters(
            methods=methods,
            x_datasets=[x_ew],
            y_labels=y_labels,
            random_labels_perc=0.0,
            dataset_names=["EW_UD"],
            n_lengths=[5, 10, 15],
            n_clusters=n_clusters,
            external_features=[ext_feats],
            explain=True,
            num_runs=num_runs,
        )

    else:
        results_labels_ew = None
        explanations_ew = None
        results_labels_ud = None
        explanations_ud = None

        if x_ew is not None and x_ud is not None and "time2feat" in methods:
            results_labels, explanations = compute_clusters(
                methods=methods,
                x_datasets=[x_ew, x_ud],
                y_labels=y_labels,
                random_labels_perc=0.0,
                dataset_names=["EW", "UD"],
                n_lengths=[5, 10, 15],
                n_clusters=n_clusters,
                external_features=[ext_feats],
                explain=True,
                num_runs=num_runs,
            )
        else:
            if x_ew is not None:
                results_labels_ew, explanations_ew = compute_clusters(
                    methods=methods,
                    x_datasets=[x_ew],
                    y_labels=y_labels,
                    random_labels_perc=0.0,
                    dataset_names=["EW"],
                    n_lengths=[5, 10, 15],
                    n_clusters=n_clusters,
                    external_features=[ext_feats],
                    explain=True,
                    num_runs=num_runs,
                )
                results_labels_ew = {"EW": results_labels_ew}
                explanations_ew = {"EW": explanations_ew}

            if x_ud is not None:
                results_labels_ud, explanations_ud = compute_clusters(
                    methods=methods,
                    x_datasets=[x_ud],
                    y_labels=y_labels,
                    random_labels_perc=0.0,
                    dataset_names=["UD"],
                    n_lengths=[5, 10, 15],
                    n_clusters=n_clusters,
                    external_features=[ext_feats],
                    explain=True,
                    num_runs=num_runs,
                )
                results_labels_ud = {"UD": results_labels_ud}
                explanations_ud = {"UD": explanations_ud}

            if results_labels_ew is not None and results_labels_ud is not None:
                results_labels = {
                    **results_labels_ew,
                    **results_labels_ud,
                }

                explanations = dict()
                for explanation in [explanations_ew, explanations_ud]:
                    if explanation is not None:
                        for key, value in explanation.items():
                            explanations[key] = value

            else:
                results_labels = results_labels_ew or results_labels_ud
                explanations = explanations_ew or explanations_ud

    return results_labels, explanations
