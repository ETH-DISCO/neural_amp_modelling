# PANAMA: PArametric Neural Amp Modeling with Active learning

This is the implementation of our parametric amp modelling framework. Our code is based on [NAM](https://www.neuralampmodeler.com). Demos can be found [here](https://eth-disco.github.io/neural_amp_modelling/).

## Setup
Create a conda environment from `environments/environment_gpu.yml`.

## Config JSONs
There are three types of config JSONs:  
- **Model config**, which defines the model architecture. A default one is provided in `default_config_files/models/wavenet.json`, which assumes the global condition size (i.e. number of amp knobs) is 6. Of course, you can change it for different training scenarios.  
- **Learning config**, which provides learning hyperparams. `default_config_files/learning/default.json` shows the default learning config. We observed that using a larger batch size generally results in better results.  
- **Data config**, which supplies the paths of audio files in a dataset. We provide a mini example in `default_config_files/data/default.json` with 2 training points and 1 validation point.
    
All scripts will use the default model and learning config if none are specified via commandline args, however for training you need to provide your own data config. You can always use `--help` to see all commandline options!

## Inference
Inference can be done with `inference_w_ckpt.py`.  By default, it will use the demo checkpoint `demo_ckpt.ckpt`, which was trained on 300 datapoints for 50 epochs.   

Usage example:
```bash 
python3 inference_w_ckpt.py --input-path "my_input.wav" --g-vector 0.5 0.2 0.5 0.7 0.5 0.8
```

## Single Model Training
Training is done with `custom_train_full.py`. The results will be in a subdirectory inside of `--base-outdir`.

Usage example: 
```bash
python3 custom_train_full.py --base-outdir "outputs" --data-config "my_training_config.json"`
```

## Active Learning
`active_learner_multi_gpu.py` performs one round of our active learning method. We recommend manually making a separate output directory for it. This ***same*** directory, for example `my_active_learning_folder`, should be supplied each round along with the current round number. Round indices start at 0, where the 0-th round additionally needs a starting data config from `--starter-data-config-path`. The signal x used during optimization of cross-model disagreement with regard to g is supplied via `--x-path-for-g-opt`. By default, the ensemble size is 4, but can be adjusted via `--ensemble-size`. Make sure you have a multi-GPU setup, and that the number of GPUS is at least ensemble size + 1.

When the script terminates, it logs the g-vectors of the new datapoints to be gathered. After supplying the new audio clips in `active_learning_inputs`, rerun it with the same output directory and the new round index. We recommend using a bash script for this.   

Usage example: 
```bash
python3 active_learner_multi_gpu.py --current-round-idx 0 --x-path-for-g-opt "my_input_for_g_optimization.wav" --output-dir "my_active_learning_folder" --starter-data-config-path "my_starter_data_config.json" 
# Gather new datapoints
python3 active_learner_multi_gpu.py --current-round-idx 1 --x-path-for-g-opt "my_input_for_g_optimization.wav" --output-dir "my_active_learning_folder"
# Gather new datapoints
...
```

## Testing 
Test a model with `test_model.py`. This (ideally) requires a separate data config from the training one. The config should still be the same JSON format, but only the "validation" split of it will be used. The result metrics will be stored as a JSON file and visualized as a plot.   

Usage example: 
```bash
python3 test_model.py --data-config-path "my_test_data_config.json" --ckpt-path "model_to_be_tested.ckpt" --metrics-path "results.json" --plot-path "results.png"`
```
