from tmm.objects.optics import Absorptance, Reflectance, Transmittance
from tmm.functions.single_tmm import tmm_single

import jax

def tmm_batch(
    n: jax.Array,          # (W, L)
    k: jax.Array,          # (W, L)
    d: jax.Array,          # (L,)
    wavelengths: jax.Array, # (W,)
    angles: jax.Array,     # (A,)
) -> tuple[Reflectance, Transmittance, Absorptance]:
    """
    TMM over a grid of wavelengths and incidence angles.
    Returns containers where each field has shape (A, W).
    """

    # First vmap over wavelength (first axis of n, k, wavelengths)
    tmm_wl = jax.vmap(tmm_single, in_axes=(0, 0, None, 0, None))

    # Then vmap over angles (keep all wavelength‑related inputs fixed)
    tmm_wl_angle = jax.vmap(tmm_wl, in_axes=(None, None, None, None, 0))

    # Apply: shape (A, W, L) for n,k? No – we keep (W,L) for n,k and broadcast.
    # tmm_wl expects (W,L) for n,k and (W,) for wavelengths, returns a single result per wavelength.
    # So tmm_wl_angle expects (A,) angles, and maps over them.
    return tmm_wl_angle(n, k, d, wavelengths, angles)