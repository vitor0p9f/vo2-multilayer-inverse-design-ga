from specifications.objects.structure import Structure
from specifications.objects.material import Material

import jax
import jax.numpy as jnp

def prepare_structure(
    wavelength_grid: jax.Array,          # (W,) – target wavelength grid, strictly increasing
    structure: Structure,                # structure containing material indices (int8 array)
    material_database: list[Material],   # list of Material objects with .wavelengths, .n, .k
    temperature: float,                  # temperature (K) for which to take the optical constants
) -> tuple[jax.Array, jax.Array]:
    """
    Interpolates the optical constants of all layers in a structure onto a
    common wavelength grid for a given temperature.

    Args:
        wavelength_grid: 1-D array of target wavelengths [m], shape (W,).
        structure: A `Structure` instance whose `materials` attribute is a
                   1-D integer array (indices into `material_database`).
        material_database: List of available materials, each supporting
                           `get_data_by_temperature(temperature)`.
        temperature: Temperature in Kelvin for which to retrieve n & k.

    Returns:
        n_stack: Real refractive index at each wavelength for each layer,
                 shape (W, L).
        k_stack: Extinction coefficient at each wavelength for each layer,
                 shape (W, L).
    """
    # Extract the material indices for each layer (length L)
    indices = structure.materials  # shape (L,) – dtype usually int8

    n_list = []  # will hold (W,) arrays
    k_list = []

    for idx in indices:
        # Fetch the corresponding material from the database
        mat = material_database[int(idx)]

        # Obtain (wl, n, k) for the desired temperature
        wl, n_vals, k_vals = mat.get_data_by_temperature(temperature)

        # Linear interpolation onto the common wavelength grid
        n_int = jnp.interp(wavelength_grid, wl, n_vals)   # (W,)
        k_int = jnp.interp(wavelength_grid, wl, k_vals)   # (W,)

        n_list.append(n_int)
        k_list.append(k_int)

    # Stack along the layer axis to form (W, L) arrays
    n_stack = jnp.stack(n_list, axis=1)
    k_stack = jnp.stack(k_list, axis=1)

    return n_stack, k_stack