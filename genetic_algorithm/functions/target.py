from typing import Union, Type
from specifications.types.optics import Reflectance, Transmittance, Absorptance
from specifications.types.environment import Wavelengths

import jax.numpy as jnp

def super_gaussian_target(
    wavelengths: Wavelengths,          # (W,)  wavelengths [m]
    target_cls: Union[
        Type[Reflectance],
        Type[Transmittance],
        Type[Absorptance]
    ],
    center: float,                    # λ₀ – central wavelength [m]
    width: float,                     # σ – standard deviation [m]
    order: float,                     # p – super‑Gaussian order (≥1)
    amplitude: float = 1.0,           # A – peak amplitude above baseline
    baseline: float = 0.0,            # B – baseline level
) -> Union[Reflectance, Transmittance, Absorptance]:
    """
    Super‑Gaussian spectral target (traditional Gaussian‑reverting form).

    The profile is defined as:
        f(λ) = B + A · exp( –[ (λ – λ₀)² / (2σ²) ]^p )

    When p = 1, this is exactly a Gaussian with standard deviation σ.
    For p > 1, the profile becomes increasingly flat‑topped and
    sharper‑edged.

    The target is constant with respect to angle, yielding a shape of
    (1, W) for each polarization field.

    Args:
        wavelengths: 1D array of wavelengths in meters.
        target_cls: The class to instantiate (Reflectance, Transmittance,
                    or Absorptance).
        center: λ₀ – center wavelength [m].
        width: σ – standard deviation controlling the width [m].
        order: p – exponent controlling flatness (p=1 → pure Gaussian).
        amplitude: A – peak height above baseline.
        baseline: B – baseline level.

    Returns:
        An instance of `target_cls` with both s‑ and p‑polarized fields
        filled with the same super‑Gaussian profile (shape (1, W)).
    """
    # Validate target type
    if target_cls not in (Reflectance, Transmittance, Absorptance):
        raise TypeError(
            "target_cls must be one of Reflectance, Transmittance, Absorptance"
        )

    # Compute the 1D super‑Gaussian profile (traditional form)
    wl = wavelengths
    scaled_sq = ((wl - center) ** 2) / (2 * width ** 2)   # (W,)
    profile = baseline + amplitude * jnp.exp(- (scaled_sq ** order))  # (W,)

    # Add a dummy angle dimension (fixed normal incidence)
    profile_2d = jnp.expand_dims(profile, axis=0)         # (1, W)

    # Both polarization channels receive the same profile
    return target_cls(p_polarized=profile_2d, s_polarized=profile_2d)