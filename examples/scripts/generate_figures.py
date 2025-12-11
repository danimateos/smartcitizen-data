import argparse
import gc
import math
import os
import logging
import numpy as np
import pandas as pd
import scdata as sc
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib


font = {'family' : 'Helvetica',
        'weight' : 'light',
        'size'   : 8}
matplotlib.rc('font', **font)


def heatmap_subset(subset: pd.DataFrame, metric: str, 
                   metrics_folder = "./twinair_health_metrics", 
                   output_folder = "./twinair_figures", 
                   show=True, 
                   savefig=True,
                   title_prefix: str=""):

    dfs = []
    for device_id in np.sort(subset["id"].unique()):
        filename = f"{metrics_folder}/{device_id}_{metric}.csv.gz"
        if not os.path.exists(filename):
            continue
        
        df = pd.read_csv(filename, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)

        dfs.append((device_id, df))
        logging.getLogger(__name__).info(f"Loaded {filename} - {df.shape}")

        
    # need to know max date and min date and resample().reindex,
    # or the heatmaps won't align even if we set sharey=True
    start = min([df.index.min() for (_, df) in dfs]) - pd.Timedelta("1d")
    stop = max([df.index.max() for (_, df) in dfs]) + pd.Timedelta("1d")
    
    print(start, stop)
    idx = pd.date_range(start, stop, freq="1min")
    
    n = len(dfs)
    f, axes = plt.subplots(1, n, figsize = (4*n, 14) , sharey=True)
    cbar_ax = f.add_axes([.93, .2, .03, .6])
    ix = 0
        
    for device_id, df in dfs:
        ax = axes[ix]
        
        to_plot = df.resample("1min", origin=pd.Timestamp(start)).mean().reindex(idx)
        logging.getLogger(__name__).info(f"Before: {device_id} - {df.shape}")
        logging.getLogger(__name__).info(f"After: {device_id} - {to_plot.shape}")

        sns.heatmap(to_plot, 
                    cmap="viridis", 
                    vmin=0.0, 
                    vmax=1.0, 
                    ax=ax,
                    cbar=ix == 0,
                    cbar_ax=None if ix else cbar_ax)    
        ax.set_facecolor("grey")
        ax.grid(False)
        ax.set_title(f"Device {device_id} {metric}; Location:{id_to_location.get(device_id)}\n Site:{id_to_site.get(device_id)}, Room:{id_to_room.get(device_id)}")
        
        ix += 1
        
    f.tight_layout(rect=[0, 0, .9, 1])

    target = f"twinair_figures/{title_prefix}_{metric}.png"
    if savefig and not os.path.exists(target):
        f.savefig(target)
    if show:
        plt.show()
    
    f.clf()
    plt.close(f)
    gc.collect()

    
def compare_metrics(device_id: int, title_prefix:str="", 
                    metrics_folder = "./twinair_health_metrics", 
                    output_folder = "./twinair_figures", 
                    show=True, 
                    savefig=True):
    
    dfs = []
    
    df = pd.read_csv(f"{metrics_folder}/{device_id}.csv.gz", index_col=0)
    df.index = pd.to_datetime(df.index, utc=True)
    dfs.append(("raw", df))
    
    metrics = ["nan_ratios", "top_value_ratios", "implausible_ratios", "outlier_ratios"]
    
    for metric in metrics:
        filename = f"{metrics_folder}/{device_id}_{metric}.csv.gz"
        if not os.path.exists(filename):
            continue
        
        df = pd.read_csv(filename, index_col=0)
        df.index = pd.to_datetime(df.index, utc=True)
        dfs.append((metric, df))

        
    # need to know max date and min date, or the heatmaps won't align even if they share y            
    start = min([df.index.min() for (_, df) in dfs]) - pd.Timedelta("1d")
    stop = min([df.index.max() for (_, df) in dfs]) + pd.Timedelta("1d")
    idx = pd.date_range(start, stop, freq="1min")
    
    n = len(dfs)
    f, axes = plt.subplots(1, n, figsize = (4*n, 14) , sharey=True)
    cbar_ax = f.add_axes([.93, .2, .03, .6])
    ix = 0
        
    for metric, df in dfs:
        ax = axes[ix]
        
        to_plot = df.resample("1min", origin=pd.Timestamp(start)).mean().reindex(idx)
        rescaled = to_plot / to_plot.max()
        
        sns.heatmap(rescaled, 
                    cmap="magma" if ix == 0 else "viridis", 
                    vmin=None if ix == 0 else 0.0, 
                    vmax=None if ix == 0 else 1.0, 
                    ax=ax,
                    cbar=ix == 1,
                    cbar_ax=None if ix == 0 else cbar_ax)    
        ax.set_facecolor("grey")
        ax.grid(False)
        ax.set_title(f"Device {device_id} {metric}; Location:{id_to_location.get(device_id)}\n Site:{id_to_site.get(device_id)}, Room:{id_to_room.get(device_id)}")
        
        ix += 1
        
    f.tight_layout(rect=[0, 0, .9, 1])
    
    target = f"twinair_figures/{device_id}_summary.png"
    if savefig and not os.path.exists(target):
        f.savefig(target)
    if show:
        plt.show()

    f.clf()
    plt.close(f)
    gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TwinAIR device health metric calculations.")
    parser.add_argument(
        "source",
        help="Path to the Excel file containing device info",
    )

    parser.add_argument(
        "criterion",
        help="Name of the Column to filter devices (e.g. 'Location')",
    )

    parser.add_argument(
        "value",
        help="Value used to filter devices (e.g. 'Thriassio')",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Log level (DEBUG, INFO, WARNING, ERROR)",
    )

    args = parser.parse_args()

    # configure logging from CLI
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)
    logger.info("Running scdata version %s", sc.__version__)

    criterion = args.criterion
    value = args.value


    source = args.source
    logger.info("Loading device list from file %s", source)

    criterion = args.criterion
    logger.info("Filtering by %s == %s", criterion, value)


    devices = pd.read_excel(source, sheet_name="DEVICES")
    devices["id"] = pd.to_numeric(devices["id"], errors="coerce")
    devices = devices.dropna(subset="id")
    devices["id"] = devices["id"].astype(int)

    logger.info(f"Found {len(devices['id'])} devices")


    id_to_location = dict(zip(devices["id"], devices["Location"]))
    id_to_site = dict(zip(devices["id"], devices["Site"]))
    id_to_room = dict(zip(devices["id"], devices["Room"]))

    if criterion == "None":
        these = devices
    else:
        these = devices[devices[criterion] == value]

    logger.info(f"Filtered devices, remaining {len(these['id'])}")


    for metric in ["nan_ratios", "top_value_ratios", "implausible_ratios", "outlier_ratios"]:
        logger.info(f"Generating {metric} heatmaps for  {criterion}=={value} with {len(these)} devices")
        heatmap_subset(these, metric=metric, show=False, savefig=True, title_prefix=f"{criterion}_{value}")

    for device_id in these["id"]:
        logger.info(f"Generating summary for device {device_id}")
        compare_metrics(device_id, show=False, savefig=True)