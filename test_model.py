from nam.testing_framework import test_model
from pathlib import Path as _Path
import json as _json
from nam import data, shared_in_multi_out
import argparse as _argparse

data.register_dataset_initializer("shared_in_multi_out", shared_in_multi_out.SharedInputMultiOutputDataset.init_from_config)

parser = _argparse.ArgumentParser()

parser.add_argument(
    '--data-config-path',
    type=_Path,
    default=_Path("training_data/30min_val_data/ckpt_964.json"),
    help="Path to data config JSON"
)

parser.add_argument(
    '--model-config-path',
    type=_Path,
    default=_Path("nam_full_configs_copy/models/wavenet.json"),
    help="Path to model config JSON"
)

parser.add_argument(
    '--ckpt-path',
    type=_Path,
    help="Path to checkpoint file"
)

parser.add_argument(
    '--metrics-path',
    type=_Path,
    help="Path to output metrics JSON"
)

parser.add_argument(
    '--plot-path',
    type=_Path,
    help="_Path to output plot image"
)

args = parser.parse_args()


test_model(
    data_config_path=args.data_config_path,
    model_config_path=args.model_config_path,
    ckpt_path=args.ckpt_path,
    metrics_path=args.metrics_path,
    plot_path=args.plot_path
)