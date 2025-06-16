import itertools
from nam.train.lightning_module import LightningModule as _LightningModule
import nam.data as data
import torch
import json as _json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description="Run inference with NAM model checkpoint.")
parser.add_argument("--input-path", type=Path, help="Input audio file path")
parser.add_argument("--output-dir", type=Path, default=".", help="Output directory")
parser.add_argument("--g-vector", type=float, nargs="+", default=[0.5, 0.5, 0.5, 0.5, 0.5, 0.5], help="Global condition vector")
parser.add_argument("--model-config-path", type=Path, default="default_config_files/models/wavenet.json", help="Model config path")
parser.add_argument("--ckpt-path", type=Path, default="demo_ckpt.ckpt", help="Checkpoint path")
parser.add_argument("--index", type=int, default=0, help="Index of current demo")
args = parser.parse_args()

INPUT_AUDIO_PATH = args.input_path
if not INPUT_AUDIO_PATH.exists():
    raise FileNotFoundError(f"Input audio file not found: {INPUT_AUDIO_PATH}")
OUTPUT_AUDIO_DIR = args.output_dir
G_VECTOR = args.g_vector
MODEL_CONFIG_PATH = args.model_config_path
CKPT_PATH = args.ckpt_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(MODEL_CONFIG_PATH, "r") as fp:
    model_config = _json.load(fp)
model = _LightningModule.load_from_checkpoint(
            CKPT_PATH,
            **_LightningModule.parse_config(model_config)
        )
model.eval()
model = model.to(device)

input_audio, info = data.wav_to_tensor(INPUT_AUDIO_PATH, info=True)
sample_rate = info.rate
if sample_rate != model.net.sample_rate:
    # raise ValueError(f"Sample rate mismatch: input audio {sample_rate}, model {model.net.sample_rate}")
    print(f"Warning: Sample rate mismatch: input audio {sample_rate}, model {model.net.sample_rate}. Resampling...")
    import librosa
    import numpy as np
    input_audio = np.array(input_audio)
    input_audio = librosa.resample(input_audio, orig_sr=sample_rate, target_sr=model.net.sample_rate)
    input_audio = torch.tensor(input_audio, dtype=torch.float32)
input_audio = input_audio.to(device)

print("Processing input audio...")

G_VECTOR_COMBINATIONS_PER_ELEMENT = [
    [0.3, 0.5, 0.75],
    [0.0, 0.5, 1.0],
    [0.5],
    [0.5],
    [0.5],    
    [0.0, 0.5, 1.0]
]

# Generate all possible combinations by taking one element from each sublist
g_vector_combinations = list(itertools.product(*G_VECTOR_COMBINATIONS_PER_ELEMENT))
g_vector_combinations = [list(comb) for comb in g_vector_combinations]

print(f"Total G-vector combinations: {len(g_vector_combinations)}")

def num2lohimid(num):
    if num < 0.5:
        return "low"
    elif num > 0.5:
        return "high"
    else:
        return "mid"

for i, G_VECTOR in enumerate(g_vector_combinations):
    print(f"Processing combination {i + 1}/{len(g_vector_combinations)}: {G_VECTOR}")
    gain = num2lohimid(G_VECTOR[0])
    bass = num2lohimid(G_VECTOR[1])
    presence = num2lohimid(G_VECTOR[5])
    
    name = f"demo_{args.index}_ours_gain-{gain}_tone-{bass}_pres-{presence}.wav"

    g_tensor = torch.tensor(G_VECTOR, dtype=torch.float32).unsqueeze(0).to(device)  
    print(f"Global condition tensor: {g_tensor}")

    output = model.forward(input_audio, g_tensor)

    output_audio_path = OUTPUT_AUDIO_DIR / Path(name)
    data.tensor_to_wav(output, output_audio_path, sample_rate)