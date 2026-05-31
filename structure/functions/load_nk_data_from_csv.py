from pathlib import Path
from typing import Tuple
import jax.numpy as jnp
import numpy as np

def load_nk_data_from_csv(path: Path) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Load wavelength, n, k from a CSV file.

    Supports two file formats:
    1. Two‑section format (RefractiveIndex.info):
          wl,n
          <data rows>
          <empty line>
          wl,k
          <data rows>
       Each section must have the same wavelength grid (order can be swapped).
    2. Classic three‑column format:
          wavelength_um,n,k   (optional header)
          <data rows>

    Returns:
        wavelengths: 1D array (float32)
        n_values: 1D array (float32)
        k_values: 1D array (float32)
    """
    # Read all non‑empty lines
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"CSV file is empty: {path}")

    # ----- Detect the format -----
    # Look for section headers "wl,n" and "wl,k" (case‑insensitive)
    has_section_header = any(
        line.lower().replace(' ', '').startswith(('wl,n', 'wl,k'))
        for line in lines
    )

    if has_section_header:
        # ----- Two‑section format -----
        wl_n = []
        n_vals = []
        wl_k = []
        k_vals = []

        current_section = None
        for line in lines:
            # Normalise: remove spaces, lowercase
            clean = line.lower().replace(' ', '')
            if clean == 'wl,n':
                current_section = 'n'
                continue
            elif clean == 'wl,k':
                current_section = 'k'
                continue

            # Data line – split by comma
            parts = line.split(',')
            if len(parts) != 2:
                raise ValueError(f"Expected 2 columns, got {len(parts)} in line: {line}")

            try:
                wl_val = float(parts[0])
                val = float(parts[1])
            except ValueError as e:
                raise ValueError(f"Non‑numeric value in line: {line}") from e

            if current_section == 'n':
                wl_n.append(wl_val)
                n_vals.append(val)
            elif current_section == 'k':
                wl_k.append(wl_val)
                k_vals.append(val)
            else:
                # Data before any section header – skip or error?
                # We'll treat as error because it's ambiguous.
                raise ValueError(f"Data found before section header: {line}")

        if not wl_n or not wl_k:
            raise ValueError(
                "File is missing one or both sections (wl,n / wl,k)."
            )

        # Ensure wavelength grids match
        if wl_n != wl_k:
            # If grids are identical except order, we could sort them,
            # but for simplicity we require exact match.
            raise ValueError(
                "Wavelength grids in 'wl,n' and 'wl,k' sections do not match."
            )

        wavelengths = jnp.array(wl_n, dtype=jnp.float32)
        n_values = jnp.array(n_vals, dtype=jnp.float32)
        k_values = jnp.array(k_vals, dtype=jnp.float32)

    else:
        # ----- Classic three‑column format (original logic) -----
        # Try to detect if the first line is a header
        try:
            float(lines[0].split(',')[0])
            skip_header = 0
        except ValueError:
            skip_header = 1

        data = np.loadtxt(path, delimiter=',', skiprows=skip_header, ndmin=2)

        if data.shape[1] != 3:
            raise ValueError(
                f"Expected 3 columns (wavelength, n, k), got {data.shape[1]} columns."
            )

        wavelengths = jnp.array(data[:, 0], dtype=jnp.float32)
        n_values   = jnp.array(data[:, 1], dtype=jnp.float32)
        k_values   = jnp.array(data[:, 2], dtype=jnp.float32)

    return wavelengths, n_values, k_values