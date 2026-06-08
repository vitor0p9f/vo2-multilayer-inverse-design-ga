from pathlib import Path
from typing import Tuple
import jax.numpy as jnp
import numpy as np

def load_nk_data_from_csv(path: Path) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Load wavelength, n, k from a CSV file.

    Supports three file formats:
    1. Two‑section format (RefractiveIndex.info style):
          wl,n
          <data rows>
          <empty line>
          wl,k
          <data rows>
       If only the 'wl,n' section is present, k is set to zero for the same grid.
       If only the 'wl,k' section is present, n is set to one (vacuum) for the same grid.
    2. Two‑column classic format:
          wavelength_um,n   (optional header)
          <data rows>
       k is automatically set to zero.
    3. Three‑column classic format:
          wavelength_um,n,k (optional header)
          <data rows>

    Returns:
        wavelengths: 1D array (float32) – always in µm as read from file
        n_values: 1D array (float32)
        k_values: 1D array (float32)
    """
    # Read all non‑empty lines
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        raise ValueError(f"CSV file is empty: {path}")

    # ----- Detect the format -----
    # Look for section headers "wl,n" and "wl,k" (case‑insensitive, ignoring spaces)
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
            clean = line.lower().replace(' ', '')
            if clean == 'wl,n':
                current_section = 'n'
                continue
            elif clean == 'wl,k':
                current_section = 'k'
                continue

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
                raise ValueError(f"Data found before section header: {line}")

        # Build result based on which sections are present
        if wl_n and wl_k:
            # Both sections present – ensure grids match exactly (or could sort, but we require order match)
            if wl_n != wl_k:
                raise ValueError(
                    "Wavelength grids in 'wl,n' and 'wl,k' sections do not match. "
                    "They must have identical wavelength values in the same order."
                )
            wavelengths = jnp.array(wl_n, dtype=jnp.float64)
            n_values = jnp.array(n_vals, dtype=jnp.float64)
            k_values = jnp.array(k_vals, dtype=jnp.float64)
        elif wl_n:
            # Only n section present – set k to zero
            wavelengths = jnp.array(wl_n, dtype=jnp.float64)
            n_values = jnp.array(n_vals, dtype=jnp.float64)
            k_values = jnp.zeros_like(wavelengths)
        elif wl_k:
            # Only k section present – set n to one (vacuum)
            wavelengths = jnp.array(wl_k, dtype=jnp.float64)
            n_values = jnp.ones_like(wavelengths)
            k_values = jnp.array(k_vals, dtype=jnp.float64)
        else:
            raise ValueError(
                "No data found in either 'wl,n' or 'wl,k' section."
            )

    else:
        # ----- Classic format (two or three columns) -----
        # Detect if first line is a header
        try:
            float(lines[0].split(',')[0])
            skip_header = 0
        except ValueError:
            skip_header = 1

        data = np.loadtxt(path, delimiter=',', skiprows=skip_header, ndmin=2)

        if data.shape[1] == 2:
            wavelengths = jnp.array(data[:, 0], dtype=jnp.float64)
            n_values   = jnp.array(data[:, 1], dtype=jnp.float64)
            k_values   = jnp.zeros_like(wavelengths)
        elif data.shape[1] == 3:
            wavelengths = jnp.array(data[:, 0], dtype=jnp.float64)
            n_values   = jnp.array(data[:, 1], dtype=jnp.float64)
            k_values   = jnp.array(data[:, 2], dtype=jnp.float64)
        else:
            raise ValueError(
                f"Expected 2 or 3 columns (wavelength, n, [k]), got {data.shape[1]} columns."
            )

    return wavelengths, n_values, k_values