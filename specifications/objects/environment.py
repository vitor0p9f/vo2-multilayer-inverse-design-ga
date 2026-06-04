from dataclasses import dataclass
from jax.tree_util import register_dataclass
from jaxtyping import Float32, Array

@register_dataclass
@dataclass(frozen=True)
class Environment:
    """
    External conditions under which optical properties are evaluated.

    Attributes:
        temperatures:       1‑D array of temperatures [K].
        angles:             1‑D array of incidence angles [rad].
        wavelengths:        1‑D array of free‑space wavelengths [m].
    """
    temperatures: Float32[Array, "num_temperatures"]
    angles: Float32[Array, "num_angles"]
    wavelengths: Float32[Array, "num_wavelengths"]

    @property
    def num_temperatures(self) -> int:
        return self.temperatures.shape[0]

    @property
    def num_angles(self) -> int:
        return self.angles.shape[0]

    @property
    def num_wavelengths(self) -> int:
        return self.wavelengths.shape[0]

    def single_angle(self, index: int) -> "Environment":
        """Return a new Environment with only the selected angle (keeps dims)."""
        return Environment(
            temperatures=self.temperatures,
            angles=self.angles[index:index+1],
            wavelengths=self.wavelengths,
        )

    def single_temperature(self, index: int) -> "Environment":
        """Return a new Environment with only the selected temperature."""
        return Environment(
            temperatures=self.temperatures[index:index+1],
            angles=self.angles,
            wavelengths=self.wavelengths,
        )