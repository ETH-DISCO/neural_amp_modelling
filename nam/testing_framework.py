from nam import data, shared_in_multi_out
from nam.data import Split as _Split, init_dataset as _init_dataset
from nam.train.lightning_module import LightningModule as _LightningModule
from torch.utils.data import DataLoader as _DataLoader
import json as _json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy as _deepcopy
import time
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def plot_metrics(metrics, metrics_path, save_path):
    metrics = metrics["validation"]["outputs"]

    # plot the distribution of MSE and ESR
    mse = [output["MSE"] for output in metrics if "MSE" in output]
    esr = [output["ESR"] for output in metrics if "ESR" in output]
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.hist(mse, bins=50, color='blue', alpha=0.7)
    plt.title("MSE Distribution")
    plt.xlabel("MSE")
    plt.ylabel("Frequency")
    for patch in plt.gca().patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)
    plt.subplot(1, 2, 2)
    plt.hist(esr, bins=np.logspace(np.log10(min(esr)), np.log10(max(esr)), 50), color='green', alpha=0.7)
    plt.xscale('log')
    mean_mse = np.mean(mse)
    plt.figtext(0.1, 0.01, f"Mean MSE: {mean_mse:.5f}", ha="left", fontsize=10)

    plt.title("ESR Distribution")
    plt.xlabel("ESR")
    plt.ylabel("Frequency") 
    for patch in plt.gca().patches:
        patch.set_edgecolor('black')
        patch.set_linewidth(0.5)

    # Calculate mean ESR excluding the top 100 highest values
    esr_sorted = sorted(esr)
    mean_esr_excluding_top_100 = np.mean(esr_sorted[:-100]) if len(esr_sorted) > 100 else np.mean(esr_sorted)
    plt.figtext(0.6, 0.01, f"Mean ESR (excluding top 100): {mean_esr_excluding_top_100:.5f}", ha="left", fontsize=10)


    # add the name of the JSON file to the plot
    plt.suptitle(f"{metrics_path.name}", fontsize=16)

    # save the figure
    plt.savefig(save_path)

def test_model(data_config_path, model_config_path, ckpt_path, metrics_path, plot_path):

    with open(data_config_path, "r") as fp:
        data_config = _json.load(fp)
        # drop data_config.train/validation.outputs.for_reference
        for s in ["train", "validation"]:
            data_config[s]["outputs"] = [
                {k: v for k, v in output.items() if k != "for_reference"}
                for output in data_config[s]["outputs"]
            ]
    with open(model_config_path, "r") as fp:
        model_config = _json.load(fp)
    model = _LightningModule.load_from_checkpoint(
                ckpt_path,
                **_LightningModule.parse_config(model_config)
            )


    print(f"Model receptive field: {model.net.receptive_field}")
    print(f"Model global condition size: {model.net.global_condition_size}")

    data_config["common"]["nx"] = model.net.receptive_field

    # do separate dataset for each input file
    data_config_subset = _deepcopy(data_config)
    assert len(data_config_subset["validation"]["outputs"]) == 964

    for i in tqdm(range(len(data_config_subset["validation"]["outputs"]))):
        data_config_subset["validation"]["outputs"] = [data_config["validation"]["outputs"][i]]

        dataset = _init_dataset(data_config_subset, _Split.VALIDATION)

        model.net.sample_rate = dataset.sample_rate
        dataloader = _DataLoader(dataset, shuffle=False, batch_size=96) # TODO jack up batch size as far as possible

        all_error_power = []
        all_target_power = []
        epsilon = 1e-8
        model.eval()
        with torch.no_grad(): 
            # Iterate through batches 
            for j, batch in enumerate(dataloader):

                try:
                    x, g, y = batch
                    x = x.to(device)
                    g = g.to(device)
                    y = y.to(device)
                    y_pred = model(x, g, pad_start=False)

                    error_power = torch.mean(torch.square(y_pred - y))
                    target_power = torch.mean(torch.square(y))

                    all_error_power.append(error_power.item())
                    all_target_power.append(target_power.item())

                except Exception as e:
                    print(f"\nError processing batch {j}: {e}")
                    raise e 
                

        avg_mse = np.mean(all_error_power)
        avg_esr = avg_mse / (np.mean(all_target_power))

        # write metrics back to data_config (weird but whatever)
        data_config["validation"]["outputs"][i]["MSE"] = avg_mse
        data_config["validation"]["outputs"][i]["ESR"] = avg_esr
        
    # write metrics to file
    with open(metrics_path, "w") as fp:
        _json.dump(data_config, fp, indent=4)
    print("Metrics written to file.")

    plot_metrics(data_config, metrics_path, plot_path)
    print("Metrics plotted.")

if __name__ == "__main__":
    DATA_CONFIG_PATH = Path("training_data/30min_val_data/ckpt_964.json")
    MODEL_CONFIG_PATH = Path("nam_full_configs_copy/models/wavenet_1e-07.json")
    CKPT_PATH = Path("")
    METRICS_PATH = Path("metrics/mode=random_datapoints=300_epochs=50_testpoints=1000_addloss=nmse-1e-07.json")
    PLOT_PATH = Path("metrics/mode=random_datapoints=300_epochs=50_testpoints=1000_addloss=nmse-1e-07.png")
    test_model(DATA_CONFIG_PATH, MODEL_CONFIG_PATH, CKPT_PATH, METRICS_PATH, PLOT_PATH)