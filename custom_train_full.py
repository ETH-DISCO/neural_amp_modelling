from nam.train import full as _full
from pathlib import Path as _Path
import json as _json
from nam.util import timestamp as _timestamp
import argparse

print("script started")

def make_outdir(base_outdir: str, suffix: str) -> _Path:
    outdir = _Path(base_outdir, _timestamp() + suffix)
    outdir.mkdir(parents=True, exist_ok=False)
    return outdir

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model with custom configurations.")
    parser.add_argument(
        "--base-outdir",
        type=str,
        default="outputs",
        help="Base output directory for saving the model and logs."
    )
    parser.add_argument(
        "--outdir-suffix",
        type=str,
        default="",
        help="Suffix to append to the output directory name."
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="training_data/nam_train_data/metadata.json",
        help="Path to the data configuration file."
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="default_config_files/models/wavenet.json",
        help="Path to the model configuration file."
    )
    parser.add_argument(
        "--learning-config",
        type=str,
        default="default_config_files/learning/default.json",
        help="Path to the learning configuration file."
    )
    parser.add_argument(
        "--get-test-set-metrics",
        action="store_true",
        help="If set, will compute metrics on the test set after training. You should then also provide a test data configuration file using --test-data-config."
    )
    parser.add_argument(
        "--test-data-config",
        type=str,
        default="training_data/nam_train_data/validation_metadata.json",
        help="Path to the test data configuration file. Only used if --get-test-set-metrics is set. Only the validation split of this data will be used."
    )
    args = parser.parse_args()

    outdir = make_outdir(args.base_outdir, args.outdir_suffix)
    with open(args.model_config, "r") as fp:
        model_config = _json.load(fp)
        print("model config loaded")
    with open(args.learning_config, "r") as fp:
        learning_config = _json.load(fp)
        print("learning config loaded")
    with open(args.data_config, "r") as fp:
        data_config = _json.load(fp)
        # only keep the first 300 train data points
        data_config["train"]["outputs"] = data_config["train"]["outputs"][:300]
        # drop data_config.train/validation.outputs.for_reference
        for s in ["train", "validation"]:
            data_config[s]["outputs"] = [
                {k: v for k, v in output.items() if k != "for_reference"}
                for output in data_config[s]["outputs"]
            ]
        print("data config loaded")

    # Train
    best_ckpt_path = _Path(_full.main(data_config, model_config, learning_config, outdir))

    # Test
    if not args.get_test_set_metrics:
        print("Skipping test set metrics computation.")
        exit(0)
    print("Computing test set metrics...")
    from nam.testing_framework import test_model
    test_model(
        data_config_path=args.test_data_config,
        model_config_path=args.model_config,
        ckpt_path=best_ckpt_path,
        metrics_path=outdir / "metrics.json",
        plot_path=outdir / "plot.png"
    )
