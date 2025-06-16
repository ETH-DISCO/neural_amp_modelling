# File: shared_input_data.py
# (Place in the same directory as data.py, e.g., nam/data/)

import logging
from pathlib import Path as _Path
from typing import ( Sequence as _Sequence, Optional as _Optional, Union as _Union, Tuple as _Tuple, List as _List )
from copy import deepcopy as _deepcopy

import numpy as _np
import torch as _torch

# Import necessary components from the main data module and core
# Adjust the import path if your structure differs slightly
from .data import (
    AbstractDataset, Dataset, DataError, XYError, wav_to_tensor, WavInfo, _REQUIRED_CHANNELS
)
from ._core import InitializableFromConfig

logger = logging.getLogger(__name__)


class SharedInputMultiOutputDataset(AbstractDataset, InitializableFromConfig):
    """
    Dataset for a single input file `x` shared across multiple output files `y`
    each paired with a global condition `g` provided directly in the config.

    Initialization is handled by the InitializableFromConfig base class,
    using the dictionary returned by `parse_config`.
    """
    # --- __init__ remains the same as before ---
    def __init__(
        self,
        x: _torch.Tensor,         # The single shared input tensor (processed)
        y_list: _Sequence[_torch.Tensor], # List of output tensors (processed)
        g_list: _Sequence[_torch.Tensor], # List of global condition tensors (loaded)
        nx: int,
        ny: _Optional[int],
        sample_rate: float,
        x_path: _Optional[_Union[str, _Path]] = None,
        **kwargs
    ):
        if not y_list or not g_list or len(y_list) != len(g_list):
            raise ValueError("y_list and g_list cannot be empty and must have the same length.")

        self._x = x
        self._y_list = y_list
        self._g_list = g_list
        self._nx = nx
        self._sample_rate = sample_rate
        self._x_path = x_path

        # Validate shapes and lengths
        self._num_pairs = len(y_list)
        if not self._num_pairs > 0:
             raise ValueError("Dataset must contain at least one (y, g) pair.")
        self._g_dim = g_list[0].shape[0]
        expected_len = len(x)
        for i in range(self._num_pairs):
            if len(y_list[i]) != expected_len:
                raise XYError(f"Length mismatch: x ({expected_len}) vs y[{i}] ({len(y_list[i])})")
            if g_list[i].ndim != 1 or g_list[i].shape[0] != self._g_dim:
                 raise ValueError(f"g[{i}] has wrong shape/dim: expected ({self._g_dim},), got {g_list[i].shape}")

        # Calculate length
        n_single = len(x)
        if nx > n_single:
             raise RuntimeError(f"Input length {n_single} < receptive field nx {nx}")
        self._ny = ny if ny is not None else n_single - nx + 1
        if self._ny <= 0:
             raise RuntimeError(f"Calculated ny <= 0. Check nx ({nx}) and input length ({n_single}).")

        single_pair_len = (n_single - nx + 1) // self._ny
        if single_pair_len == 0:
             raise RuntimeError(f"Dataset pair length is zero. Input length {n_single} might be too short for nx={nx} and ny={self._ny}.")

        self._single_pair_len = single_pair_len
        self._total_len = self._num_pairs * self._single_pair_len

        # Lookup
        self._lookup = {}
        for i in range(self._total_len):
            pair_idx = i // self._single_pair_len
            idx_within_pair = i % self._single_pair_len
            self._lookup[i] = (pair_idx, idx_within_pair)

        # y_offset
        self._y_offset = nx - 1

        logger.info(f"Initialized SharedInputMultiOutputDataset: {self._num_pairs} (g,y) pairs, "
                    f"single pair length={self._single_pair_len}, total length={self._total_len}")

    # --- __len__ and __getitem__ methods remain the same ---
    def __len__(self) -> int:
        return self._total_len

    def __getitem__(self, idx: int) -> _Tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset length {len(self)}")

        pair_idx, idx_within_pair = self._lookup[idx]

        i = idx_within_pair * self._ny
        j = i + self._y_offset

        x_segment = self._x[i : i + self._nx + self._ny - 1]
        y_segment = self._y_list[pair_idx][j : j + self._ny]
        g_tensor = self._g_list[pair_idx]

        return x_segment, g_tensor, y_segment

    # --- Properties remain the same ---
    @property
    def nx(self) -> int: return self._nx
    @property
    def ny(self) -> int: return self._ny
    @property
    def sample_rate(self) -> _Optional[float]: return self._sample_rate
    @property
    def g_dim(self) -> int: return self._g_dim

    @classmethod
    def parse_config(cls, config):
        """
        Parses the configuration dictionary, loads data, performs processing,
        and returns a dictionary containing the arguments needed by __init__.

        Expected config structure:
        {
            "x_path": "path/to/input.wav",
            "outputs": [
                {
                    "y_path": "path/to/output1.wav",
                    "g_vector": [0.1, 0.5, ...],  // <<< CHANGED from g_path
                    ...other_y_params
                },
                {
                    "y_path": "path/to/output2.wav",
                    "g_vector": [0.2, 0.6, ...],  // <<< CHANGED from g_path
                    ...other_y_params
                 },
                ...
            ],
            "nx": 1025,
            "ny": 1,
            "sample_rate": 48000, // Optional, will be inferred
            // Other common params like start/stop, delay, input_gain etc.
        }
        """
        config = _deepcopy(config) # Work on a copy

        # Pop common parameters 
        common_params = {
            k: config.pop(k) for k in [
                 "nx", "ny", "start", "stop", "start_samples", "stop_samples",
                 "start_seconds", "stop_seconds", "delay", "input_gain",
                 "require_input_pre_silence"
            ] if k in config
        }
        nx = common_params.get("nx")
        if nx is None:
             raise ValueError("Missing required parameter 'nx' in shared input dataset config")
        ny = common_params.get("ny")

        # Load shared input x 
        x_path = config.pop("x_path")
        if not x_path:
             raise ValueError("Missing required parameter 'x_path' in shared input dataset config")
        sample_rate = config.pop("sample_rate", None)
        try:
            resolved_x_path = _Path(x_path).resolve()
            x, x_wavinfo = wav_to_tensor(str(resolved_x_path), info=True, rate=sample_rate)
            sample_rate = x_wavinfo.rate
            logger.info(f"Loaded shared input: {resolved_x_path}")
        except FileNotFoundError:
             raise DataError(f"Shared input file not found: {x_path} (resolved: {resolved_x_path})")
        except Exception as e:
             raise DataError(f"Failed to load shared input {x_path}: {e}")

        # Process shared input x 
        if config.get("process_x", False):
            try:
                temp_ds_kwargs_x = common_params.copy()
                # Remove keys that will be passed explicitly to avoid duplication
                temp_ds_kwargs_x.pop('nx', None)
                temp_ds_kwargs_x.pop('ny', None)
                temp_ds_kwargs_x.pop('require_input_pre_silence', None)
                temp_ds = Dataset(
                    x=x, y=x.clone(),
                    nx=nx, ny=ny, sample_rate=sample_rate,
                    require_input_pre_silence=None, # Explicitly set to None
                    **temp_ds_kwargs_x
                )
                x_processed = temp_ds.x
                logger.info(f"Processed shared input x, final length {len(x_processed)}")
            except Exception as e:
                raise DataError(f"Failed during initial processing of shared input {x_path} with common params: {e}")
        else:
            # If check_x is False, use the original x
            x_processed = x

        y_list = []
        g_list = []
        output_configs = config.pop("outputs", [])
        if not output_configs:
             raise ValueError("Config for 'shared_input' type requires a non-empty 'outputs' list.")

        # Load each y and process g pair
        for i, out_config in enumerate(output_configs):
            y_path = out_config.get("y_path")
            g_vector = out_config.get("g_vector")
            if not y_path:
                 raise ValueError(f"Missing 'y_path' in output config entry {i}")
            if g_vector is None: # Check if g_vector exists
                 raise ValueError(f"Missing 'g_vector' in output config entry {i}")
            if not isinstance(g_vector, list): # Check if it's a list
                 raise TypeError(f"'g_vector' in output config entry {i} must be a list, got {type(g_vector)}")

            # Load y, ensuring rate match and applying relevant processing (same as before)
            try:
                current_y_params = {**common_params, **out_config}
                y_preroll = current_y_params.pop("y_preroll", None)
                y_scale = current_y_params.pop("y_scale", 1.0)
                resolved_y_path = _Path(y_path).resolve()
                y_i, y_wavinfo = wav_to_tensor(str(resolved_y_path), info=True, rate=sample_rate, preroll=y_preroll)

                if config.get("check_y", False):
                    # Apply same processing to y
                    temp_ds_kwargs_y = current_y_params.copy()
                    temp_ds_kwargs_y.pop('nx', None)
                    temp_ds_kwargs_y.pop('ny', None)
                    temp_ds_kwargs_y.pop('sample_rate', None) # sample_rate is explicit
                    temp_ds_kwargs_y.pop('g_vector', None) # Remove g_vector specific to this loop
                    temp_ds_kwargs_y.pop('y_path', None) # Remove paths
                    temp_ds_kwargs_y.pop('require_input_pre_silence', None)

                    temp_y_ds = Dataset(
                        x=y_i, y=y_i.clone(),
                        nx=nx, ny=ny, sample_rate=sample_rate,          # Pass explicitly
                        require_input_pre_silence=None, # Explicitly set to None
                        **temp_ds_kwargs_y                             # Pass rest from combined params
                    )
                    y_i_processed = temp_y_ds.x * y_scale
                else:
                    y_i_processed = y_i * y_scale

                if len(y_i_processed) != len(x_processed):
                    raise XYError(f"Post-processing length mismatch for output {i} ({resolved_y_path}): "
                                f"Expected {len(x_processed)}, got {len(y_i_processed)}")
                y_list.append(y_i_processed)
            except FileNotFoundError:
                raise DataError(f"Output file not found: {y_path} (resolved: {resolved_y_path})")
            except Exception as e:
                raise DataError(f"Failed to load or process output {y_path} for pair {i}: {e}")

            # Convert list directly to tensor
            try:
                g_i = _torch.tensor(g_vector, dtype=_torch.float32)
                if g_i.ndim != 1:
                     raise ValueError(f"Global condition 'g_vector' in output {i} is not 1D after conversion (shape={g_i.shape})")
                g_list.append(g_i)
                logger.info(f"Processed global condition vector {i} from config (dim={g_i.shape[0]})")
            except Exception as e:
                raise DataError(f"Failed to process 'g_vector' in output {i}: {e}")

        # Return dictionary matching __init__ parameters (same as before)
        init_args = {
            "x": x_processed,
            "y_list": y_list,
            "g_list": g_list,
            "nx": nx,
            "ny": ny,
            "sample_rate": sample_rate,
            "x_path": x_path,
            **config
        }
        return init_args