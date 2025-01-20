import multiprocessing
from collections import defaultdict
import random
from typing import Optional

import numpy as np
import pandas as pd
import psutil
from tabulate import tabulate

from plotting import plot_unsupervised_metrics_range


def select_random_percent(labels, perc):
    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # Select the percentage of indexes randomly for each class
    selected_indices = {}
    for label, indices in class_indices.items():
        num_to_select = max(
            1, int(len(indices) * perc)
        )  # At least one item should be selected
        selected_indices_for_class = random.sample(indices, num_to_select)
        for idx in selected_indices_for_class:
            selected_indices[idx] = label

    return selected_indices


def gaussian(x, mu, sigma):
    return np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def realign_labels(
    ground_truth_labels, baseline_labels_ew, baseline_labels_ud, results_labels
):
    if ground_truth_labels is not None:
        if np.max(ground_truth_labels) != len(np.unique(ground_truth_labels)):
            new_cluster_labels = np.zeros(ground_truth_labels.shape, dtype=int)
            for i, cluster_label in enumerate(np.unique(ground_truth_labels), start=1):
                new_cluster_labels[ground_truth_labels == cluster_label] = i
            ground_truth_labels = new_cluster_labels

    if 0 in np.unique(baseline_labels_ew):
        baseline_labels_ew = baseline_labels_ew + 1
        baseline_labels_ud = baseline_labels_ud + 1

    for method in results_labels:
        if results_labels[method] is not None:
            for dataset in results_labels[method]:
                if results_labels[method][dataset] is not None:
                    # for dataset_name in results_labels[dataset][method]:
                    if dataset in ["time2feat", "EW + UD"]:
                        results_list = results_labels[method][dataset]
                    else:
                        results_list = results_labels[method][dataset]

                    """if isinstance(results_list, dict):
                        if dataset in ["time2feat", "EW + UD"]:
                            results_list = results_list["EW + UD"]
                        else:
                            results_list = results_list[method]"""

                    for run_num, (_, pred_labels) in enumerate(results_list):
                        if 0 in pred_labels:
                            results_labels[method][dataset][run_num][1] = (
                                np.array(pred_labels) + 1
                            )

    return (
        ground_truth_labels,
        baseline_labels_ew,
        baseline_labels_ud,
        results_labels,
    )


def get_idle_cpu_cores():
    total_cores = multiprocessing.cpu_count()
    if sys.platform == "win32":
        print("Measuring the number of idle CPU cores...")
        # first run the cpu_percent() method to update the internal CPU times
        processes_util_before = set()
        processes_util_after = dict()
        for p in psutil.process_iter(["cpu_percent"]):
            p.cpu_percent()
            processes_util_before.add(p.pid)

        # wait 5 seconds to let the processes run
        time.sleep(5)
        # update the CPU usage percentage
        for p in psutil.process_iter(["cpu_percent"]):
            processes_util_after[p.pid] = p.cpu_percent()

        # merge the processes lists so to ignore the processes which were not alive before.
        # no need to compute the difference as psutil does it with the second call to cpu_percent()
        processes_util = {
            pid: processes_util_after[pid]
            for pid in processes_util_after
            if pid in processes_util_before
        }
        # being Windows,
        # we consider cores
        # to be idle if they are used at less than 10% of their capacity
        active_processes = len([p for p in processes_util.values() if p > 10])
    else:
        active_processes = len(
            [p for p in psutil.process_iter() if p.status() == "running"]
        )
    idle_cores = max(1, total_cores - active_processes)
    return idle_cores


def show_and_save_data(
    metrics_results: dict,
    explanations_per_n_clusters: dict,
    user_n_clusters: int,
    baseline_n_clusters: int,
    ground_truth_n_clusters: Optional[int] = None,
    print_explanation: bool = False,
):
    # print metric results as text table
    # flatten the dictionary
    for labels_name in metrics_results.keys():
        flattened_data = list()
        if labels_name == "methods_results":
            current_metrics_results = metrics_results[labels_name]
            for n_clusters, methods in current_metrics_results.items():
                for method, datasets in methods.items():
                    for dataset, metrics in datasets.items():
                        for metric, value in metrics.items():
                            flattened_data.append(
                                (n_clusters, method, dataset, metric, value)
                            )
        else:
            method = (
                "baseline"
                if labels_name in ["baseline_labels_ew", "baseline_labels_ud"]
                else "ground_truth"
            )
            if method == "ground_truth" and ground_truth_n_clusters is not None:
                n_clusters = ground_truth_n_clusters
            else:
                n_clusters = (
                    user_n_clusters
                    if labels_name == "results_labels"
                    else baseline_n_clusters
                )
            for dataset, metrics in metrics_results[labels_name].items():
                for metric, value in metrics.items():
                    flattened_data.append((n_clusters, method, dataset, metric, value))

        # create a DataFrame
        res_df = pd.DataFrame(
            flattened_data,
            columns=["n_clusters", "method", "dataset", "metric", "value"],
        )

        # pivot the DataFrame
        res_df_pivot = res_df.pivot_table(
            index="metric", columns=["n_clusters", "method", "dataset"], values="value"
        )

        # TODO: for some reason time2feat is still in the table even if the only method is featts. Investigate.
        print(
            tabulate(
                res_df_pivot, headers="keys", showindex=True, tablefmt="rounded_grid"
            )
        )
        res_df_pivot.to_csv(
            f"metrics_results_{labels_name}.csv", index=True, header=True
        )
        if print_explanation:
            print("Explanation of the clustering results:")
            print(explanations_per_n_clusters)

            pd.DataFrame(
                explanations_per_n_clusters, columns=res_df_pivot.columns
            ).to_csv(f"explanations_{labels_name}.csv", index=True, header=True)

    # TODO: fix this function. The subplots axes throw an exception at some point.
    # plot_unsupervised_metrics_range()
