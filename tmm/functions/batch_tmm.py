from tmm.objects.optics import Absorptance, Reflectance, Transmittance
from tmm.functions.single_tmm import tmm_single

import jax


def tmm_batch(
    n: jax.Array,          # (W, L)  real part of refractive index
    k: jax.Array,          # (W, L)  extinction coefficient
    d: jax.Array,          # (L,)    layer thicknesses in meters
    wavelengths: jax.Array, # (W,)   free‑space wavelengths [m]
    angles: jax.Array,     # (A,)   incidence angles [rad]
) -> tuple[Reflectance, Transmittance, Absorptance]:
    """
    Vectorised Transfer Matrix Method over wavelength and angle grids.

    This function efficiently computes the optical response of the same
    multilayer stack for multiple wavelengths and incidence angles by
    applying `tmm_single` on a 2D grid via two nested `vmap` calls.

    The stack is described by:
      - wavelength‑dependent complex refractive indices ``n + i*k`` of
        shape ``(W, L)``,
      - a single set of layer thicknesses ``d`` of shape ``(L,)``.

    The output containers contain arrays of shape ``(A, W)``, where the
    first axis indexes the angles and the second indexes the wavelengths.

    Parameters
    ----------
    n : (W, L) jax.Array
        Real part of the refractive index for each wavelength and layer.
    k : (W, L) jax.Array
        Extinction coefficient for each wavelength and layer.
    d : (L,) jax.Array
        Thickness of each layer [m].  The first and last values are
        ignored (semi‑infinite media).
    wavelengths : (W,) jax.Array
        Free‑space wavelengths [m] at which the materials are defined.
    angles : (A,) jax.Array
        Incidence angles in the entrance medium [rad].

    Returns
    -------
    (Reflectance, Transmittance, Absorptance) :
        Each object holds the s‑polarised, p‑polarised and unpolarised
        values, all of shape ``(A, W)``.
    """

    # Vectorise over wavelengths: map over first axis of n, k, wavelengths
    tmm_wl = jax.vmap(tmm_single, in_axes=(0, 0, None, 0, None))

    # Vectorise over angles: map over the last (angle) axis, keeping all
    # wavelength‑dependent inputs fixed
    tmm_wl_angle = jax.vmap(tmm_wl, in_axes=(None, None, None, None, 0))

    # Apply: shape (A, W, L) for n,k? No – we keep (W,L) for n,k and broadcast.
    # tmm_wl expects (W,L) for n,k and (W,) for wavelengths, returns a single result per wavelength.
    # So tmm_wl_angle expects (A,) angles, and maps over them.
    return tmm_wl_angle(n, k, d, wavelengths, angles)