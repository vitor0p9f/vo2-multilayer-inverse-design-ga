from dataclasses import dataclass
from jax.tree_util import register_dataclass
from jaxtyping import Float32, Array, Bool
from .environment import Wavelengths

import jax.numpy as jnp

N_values = Float32[Array, "num_wavelengths"]
K_values = Float32[Array, "num_wavelengths"]
BooleanMask = Bool[Array, "num_wavelengths"]

@register_dataclass
@dataclass(frozen=True)
class OpticalProperty:
    p_polarized: Float32[Array, "num_wavelengths"]
    s_polarized: Float32[Array, "num_wavelengths"]
    
    @property
    def non_polarized(self) -> Float32[Array, "num_wavelengths"]:
        return (self.p_polarized + self.s_polarized) / 2

@register_dataclass
@dataclass(frozen=True)
class Absorptance(OpticalProperty):
    pass

@register_dataclass
@dataclass(frozen=True)
class Reflectance(OpticalProperty):
    pass

@register_dataclass
@dataclass(frozen=True)
class Transmittance(OpticalProperty):
    pass

@register_dataclass
@dataclass(frozen=True)
class SpectralBand:
    """
    Definition of a spectral band of interest.

    Attributes:
        start:   Lower wavelength bound [meters].
        end:     Upper wavelength bound [meters] (must be > start).
        weight:  Importance weight of this band in a multi‑objective loss.
    """
    start: Float32[Array, ""]   # scalar
    end: Float32[Array, ""]     # scalar
    weight: Float32[Array, ""]  # scalar

    def __post_init__(self):
        if not jnp.all(self.end > self.start):
            raise ValueError("Band end must be greater than start.")

    def contains(self, wavelength: Wavelengths) -> BooleanMask:
        """Return a boolean mask of shape (W,) indicating which wavelengths lie in the band."""
        return (wavelength >= self.start) & (wavelength <= self.end)