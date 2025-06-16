import torch
import numpy as np
import random
from pathlib import Path
import warnings
import json as _json
from nam.train.lightning_module import LightningModule as _LightningModule
import nam.data as data

### Creates demo clips in batch

ORIGINAL_INPUT_AUDIO_PATH = Path("combined.wav") # Input fed into the model
REFERENCE_AUDIO_PATH = Path("bright.wav")   # Ground truth audio for comparison
MODEL_CONFIG_PATH = Path("../nam_full_configs_copy/models/wavenet.json") # Model architecture config
CKPT_PATH = Path("/home/lj/neural-amp-modeler/trained_models/150+30/lightning_logs/version_0/checkpoints/epoch=0088_step=463868_ESR=1.178e+02_MSE=6.041e-05.ckpt") # Model checkpoint
OUTPUT_CHUNK_DIR = Path("./eval_data/150/bright_audio_chunks") # Base directory to save the output chunks

# Model conditioning vector (example)
# dark
# G_VECTOR = [0.6, 1, 0.5, 0, 0.5, 0]

# bright
G_VECTOR = [0.2, 0.3, 0.5, 0.8, 0.5, 0.8]

# superquiet
# G_VECTOR = [0.2, 0.5, 0.5, 0.5, 0.2, 0.5]

# Evaluation parameters
NUM_SAMPLES = 20  # Number of random intervals to sample
INTERVAL_DURATION_SEC = 15 # Duration of each interval in seconds
# Enable saving of individual chunks
SAVE_AUDIO_CHUNKS = True

# --- Metric Calculation Function (Updated for ESR) ---
def calculate_metrics(reference: torch.Tensor, output: torch.Tensor, epsilon: float = 1e-9) -> tuple[float, float]:
    """
    Calculates Mean Squared Error (MSE) and Error-to-Signal Ratio (ESR)
    between two audio tensors (assumed to be single-channel).

    Args:
        reference: The ground truth audio tensor (1D).
        output: The processed audio tensor (1D).
        epsilon: A small value to prevent division by zero.

    Returns:
        A tuple containing (mse, esr). Returns (NaN, NaN) if shapes mismatch
        or if reference signal power is zero.
    """
    # Ensure tensors are float32 for calculations and on CPU
    reference = reference.to(torch.float32).cpu()
    output = output.to(torch.float32).cpu()

    # Check shapes - must be 1D and same length
    if reference.shape != output.shape or reference.ndim != 1:
        warnings.warn(f"Shape mismatch or not 1D: ref={reference.shape}, out={output.shape}. Skipping metrics for this interval.")
        return float('nan'), float('nan')

    # Calculate MSE
    mse = torch.mean((reference - output) ** 2).item()

    # Calculate ESR
    signal_power = torch.sum(reference ** 2)
    noise_power = torch.sum((reference - output) ** 2) # Also known as error power

    # Handle cases where signal power is zero (division by zero)
    if signal_power <= epsilon:
         warnings.warn("Reference signal power is near zero. ESR is undefined (returning NaN).")
         return mse, float('nan')

    # Calculate ESR = Error Power / Signal Power
    esr = (noise_power / signal_power).item()

    return mse, esr

# --- Main Script Logic ---

# 1. Load Model
print("Loading model...")
# (Error checking for paths omitted for brevity, present in previous version)
if not MODEL_CONFIG_PATH.is_file() or not CKPT_PATH.is_file():
     print("Error: Model config or checkpoint file not found. Please check paths.")
     exit(1)

try:
    with open(MODEL_CONFIG_PATH, "r") as fp:
        model_config = _json.load(fp)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = _LightningModule.load_from_checkpoint(
        CKPT_PATH,
        map_location=device,
        **_LightningModule.parse_config(model_config)
    )
    model.eval()
    model.to(device)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# 2. Load Full Audio Files (Metadata and prepare for slicing)
print("Loading audio files...")
try:
    input_audio_tensor, input_info = data.wav_to_tensor(ORIGINAL_INPUT_AUDIO_PATH, info=True)
    input_sr = input_info.rate
    print(f"Loaded input: {ORIGINAL_INPUT_AUDIO_PATH.name}, Sample Rate: {input_sr}, Shape: {input_audio_tensor.shape}")

    ref_audio_tensor, ref_info = data.wav_to_tensor(REFERENCE_AUDIO_PATH, info=True)
    ref_sr = ref_info.rate
    print(f"Loaded reference: {REFERENCE_AUDIO_PATH.name}, Sample Rate: {ref_sr}, Shape: {ref_audio_tensor.shape}")

except FileNotFoundError as e:
     print(f"Error: Audio file not found - {e}")
     exit(1)
except Exception as e:
    print(f"Error loading audio: {e}")
    exit(1)

# 3. Validation (Sample Rates, Model Compatibility)
print("Validating audio files and model...")
if input_sr != ref_sr:
    print(f"Error: Sample rates mismatch! Input: {input_sr} Hz, Reference: {ref_sr} Hz.")
    exit(1)
if hasattr(model, 'net') and hasattr(model.net, 'sample_rate') and input_sr != model.net.sample_rate:
    print(f"Warning: Input audio sample rate ({input_sr} Hz) differs from model's expected rate ({model.net.sample_rate} Hz).")
    # Consider adding resampling or exiting based on requirements

sample_rate = input_sr

# Trim full tensors to minimum length *before* loop if needed
min_len = min(input_audio_tensor.shape[-1], ref_audio_tensor.shape[-1])
if input_audio_tensor.shape[-1] != ref_audio_tensor.shape[-1]:
    print(f"Warning: Full Input and Reference audio lengths differ. Trimming both to {min_len} samples.")
    input_audio_tensor = input_audio_tensor[..., :min_len]
    ref_audio_tensor = ref_audio_tensor[..., :min_len]

# Prepare G_VECTOR tensor (once, before loop)
g_tensor = torch.tensor(G_VECTOR, dtype=torch.float32).unsqueeze(0).to(device)
print(f"Using G-Vector: {G_VECTOR}")

# --- Setup Directories for Saving Chunks ---
if SAVE_AUDIO_CHUNKS:
    ref_chunk_dir = OUTPUT_CHUNK_DIR / "reference_chunks"
    out_chunk_dir = OUTPUT_CHUNK_DIR / "output_chunks"
    try:
        ref_chunk_dir.mkdir(parents=True, exist_ok=True)
        out_chunk_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving audio chunks to: {OUTPUT_CHUNK_DIR}")
    except OSError as e:
        print(f"Error creating directories for saving chunks: {e}")
        SAVE_AUDIO_CHUNKS = False # Disable saving if directory creation fails

# 4. Interval Sampling, Processing, and Metric Calculation
total_samples_duration = input_audio_tensor.shape[-1] # Use length after potential trimming
interval_samples = int(INTERVAL_DURATION_SEC * sample_rate)

if interval_samples <= 0 or interval_samples > total_samples_duration:
    print(f"Error: Invalid interval duration ({INTERVAL_DURATION_SEC}s = {interval_samples} samples) for audio length ({total_samples_duration / sample_rate:.2f}s).")
    exit(1)

max_start_index = total_samples_duration - interval_samples
all_mse = []
all_esr = []

print(f"\nSampling and processing {NUM_SAMPLES} intervals of {INTERVAL_DURATION_SEC} seconds ({interval_samples} samples each)...")
for i in range(NUM_SAMPLES):
    start_index = random.randint(0, max_start_index)
    end_index = start_index + interval_samples

    # --- Slice the interval ---
    # Slicing assumes input/ref tensors are [N] -> results in [interval_samples]
    input_interval = input_audio_tensor[start_index:end_index].to(device)
    ref_interval_full = ref_audio_tensor[start_index:end_index] # Keep ref on CPU

    # --- Process THIS interval through the model ---
    try:
        with torch.no_grad():
            output_interval = model.forward(input_interval, g_tensor)
        output_interval = output_interval.cpu() # Move result back to CPU
    except Exception as e:
        print(f"  Sample {i+1:>{len(str(NUM_SAMPLES))}}/{NUM_SAMPLES}: Error during model processing for this interval - {e}")
        continue

    # --- Save Chunks (if enabled) ---
    if SAVE_AUDIO_CHUNKS:
        try:
            # Define filenames
            ref_fname = ref_chunk_dir / f"ref_chunk_{i+1:03d}.wav"
            out_fname = out_chunk_dir / f"out_chunk_{i+1:03d}.wav"
            # Save using tensor_to_wav (expects [C, N] tensor)
            data.tensor_to_wav(ref_interval_full, ref_fname, sample_rate)
            data.tensor_to_wav(output_interval, out_fname, sample_rate)
        except Exception as e:
            print(f"  Sample {i+1:>{len(str(NUM_SAMPLES))}}/{NUM_SAMPLES}: Error saving audio chunk - {e}")


    # --- Prepare for metrics (Ensure 1D) ---
    # Squeeze the channel dimension (assuming mono [1, N] -> [N])
    ref_interval_1d = ref_interval_full.squeeze(0)
    output_interval_1d = output_interval.squeeze(0)

    # --- Calculate metrics for THIS interval ---
    if ref_interval_1d.shape != output_interval_1d.shape:
         print(f"  Sample {i+1:>{len(str(NUM_SAMPLES))}}/{NUM_SAMPLES}: Shape mismatch between ref ({ref_interval_1d.shape}) and output ({output_interval_1d.shape}) slice. Skipping metrics.")
         # Still might have saved the chunks above if shapes were okay before squeeze
         continue

    mse, esr = calculate_metrics(ref_interval_1d, output_interval_1d)

    if np.isfinite(mse): all_mse.append(mse)
    if np.isfinite(esr): all_esr.append(esr)

    print(f"  Sample {i+1:>{len(str(NUM_SAMPLES))}}/{NUM_SAMPLES}: Start={start_index/sample_rate:7.2f}s, MSE={mse:9.4e}, ESR={esr:9.4e}")

    # Clear GPU cache periodically if memory pressure is still high (optional)
    if device == torch.device("cuda") and (i + 1) % 5 == 0:
         torch.cuda.empty_cache()


# 5. Results
print("\n--- Evaluation Summary ---")
print(f"Input Audio:     {ORIGINAL_INPUT_AUDIO_PATH.name}")
print(f"Reference Audio: {REFERENCE_AUDIO_PATH.name}")
print(f"Model:           {CKPT_PATH.name}")
print(f"Sample Rate:     {sample_rate} Hz")
print(f"Audio Length:    {total_samples_duration / sample_rate:.2f} s ({total_samples_duration} samples)")
print(f"Processed:       {NUM_SAMPLES} x {INTERVAL_DURATION_SEC} s intervals ({interval_samples} samples each)")
if SAVE_AUDIO_CHUNKS:
    print(f"Audio Chunks Saved To: {OUTPUT_CHUNK_DIR}")
print("-" * 26)

if all_mse:
    avg_mse = np.mean(all_mse)
    print(f"Average MSE: {avg_mse:.4e}  (from {len(all_mse)} valid samples)")
else:
    print("Average MSE: N/A (no valid samples)")

if all_esr:
    avg_esr = np.mean(all_esr)
    print(f"Average ESR: {avg_esr:.4e} (from {len(all_esr)} valid samples)")
else:
    print("Average ESR: N/A (no valid samples)")
print("--------------------------")

print("\nScript finished.")
