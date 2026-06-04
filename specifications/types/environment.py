from dataclasses import dataclass
from jax.tree_util import register_dataclass
import jax.numpy as jnp
from jaxtyping import Float32, Int16, Int8, Array

Wavelengths = Float32[Array, "num_wavelengths"]
Angles = Int8[Array, "num_angles"]                # if you later need floating angles, replace Int8 by Float32
Temperatures = Int16[Array, "num_temperatures"]   # use Float32 if temperatures are not integers


@register_dataclass
@dataclass(frozen=True)
class Environment:
    """
    External conditions under which optical spectra are evaluated.

    Holds the independent variables that affect a material's optical constants
    and the resulting reflectance / transmittance / absorptance.

    Attributes:
        temperatures:   1‑D array of temperatures in Kelvin.
        angles:         1‑D array of incidence angles.
        wavelengths:    1‑D array of free‑space wavelengths in meters.
    """
    temperatures: Temperatures
    angles: Angles
    wavelengths: Wavelengths

    @property
    def num_temperatures(self) -> int:
        """Number of temperature points."""
        return self.temperatures.shape[0]

    @property
    def num_angles(self) -> int:
        """Number of incidence angles."""
        return self.angles.shape[0]

    @property
    def num_wavelengths(self) -> int:
        """Number of wavelength points."""
        return self.wavelengths.shape[0]