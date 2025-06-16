import nam.models.base as _base
import nam.models.wavenet as _wavenet
import json
import torch

# Specify the path to your JSON configuration file
config_path = "/home/lj/neural-amp-modeler/nam_full_configs_copy/models/wavenet.json"
# Load the JSON configuration
with open(config_path, "r") as file:
    config = json.load(file)

# Now `config` contains the loaded JSON data
config = config["net"]["config"]
# Create an instance of the Wavenet model using the loaded configuration
model = _wavenet.WaveNet(**config)
# Print the model architecture
print(model)


# --- Test Parameters ---
batch_size = 4          # How many examples to process at once
sequence_length = 20480  # Example input sequence length
# Get required input dimensions from the config or model
try:
    # Assuming input_size corresponds to channels for the first layer
    num_channels = config['layers_configs'][0]['input_size']
    # Get global condition size (should be consistent across layers)
    global_condition_size = config['layers_configs'][0]['global_condition_size']
    # Alternatively, access it from the initialized model if available:
    # global_condition_size = model._global_condition_size
except KeyError as e:
    print(f"Error accessing config key: {e}. Ensure 'layers_configs', 'input_size', and 'global_condition_size' are present.")
    exit()
except AttributeError:
     print("Error: Model does not seem to have the expected '_global_condition_size' attribute.")
     # Fallback to config if model attribute missing
     global_condition_size = config['layers_configs'][0]['global_condition_size']


print(f"Input Channels (x): {num_channels}")
print(f"Global Condition Size (g): {global_condition_size}")
print(f"Batch Size: {batch_size}")
print(f"Sequence Length: {sequence_length}")
print(f"Model Receptive Field: {model.receptive_field}")
print("-" * 30)

# --- Generate Random Input Tensors ---
print("Generating random input tensors...")
# Input signal x: (batch_size, num_channels, sequence_length)
x = torch.randn(batch_size, sequence_length)

# Global condition g: (batch_size, global_condition_size)
g = torch.randn(batch_size, global_condition_size)

print(f"Shape of input x: {x.shape}")
print(f"Shape of global condition g: {g.shape}")
print("-" * 30)

# --- Perform Forward Pass ---
print("Performing forward pass (with default padding)...")
# Use torch.no_grad() to disable gradient calculations for inference
with torch.no_grad():
    # Call the forward method with both x and g
    # Using default pad=True
    output = model(x, g)
print("Forward pass complete.")
print(f"Shape of output tensor: {output.shape}")
print(x)
print(g)
print(output)


