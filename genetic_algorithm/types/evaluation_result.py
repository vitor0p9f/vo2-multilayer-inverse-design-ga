from dataclasses import dataclass
from typing import Literal
from jax.tree_util import register_dataclass
import jax.numpy as jnp
from jaxtyping import Float32, Array
from specifications.objects.structure import Structure


@register_dataclass
@dataclass(frozen=True)
class EvaluationResult:
    """
    Result of evaluating a structure under a given environment.

    Attributes:
        structure:     The evaluated structure.
        cost:          Scalar cost value.
        reflectance:   Reflectance (T, A, W).
        transmittance: Transmittance (T, A, W).
        absorptance:   Absorptance (T, A, W).
    """
    structure: Structure
    cost: float
    reflectance: Float32[Array, "num_temperatures num_angles num_wavelengths"]
    transmittance: Float32[Array, "num_temperatures num_angles num_wavelengths"]
    absorptance: Float32[Array, "num_temperatures num_angles num_wavelengths"]

    # ------------------------------------------------------------------
    # Slicing methods
    # ------------------------------------------------------------------
    def single_temperature(self, index: int) -> "EvaluationResult":
        """
        Return a new EvaluationResult containing only the given temperature.
        The temperature dimension is kept (size 1).
        """
        return EvaluationResult(
            structure=self.structure,
            cost=self.cost,
            reflectance=self.reflectance[index:index+1, :, :],
            transmittance=self.transmittance[index:index+1, :, :],
            absorptance=self.absorptance[index:index+1, :, :],
        )

    def single_angle(self, index: int) -> "EvaluationResult":
        """
        Return a new EvaluationResult containing only the given angle.
        The angle dimension is kept (size 1).
        """
        return EvaluationResult(
            structure=self.structure,
            cost=self.cost,
            reflectance=self.reflectance[:, index:index+1, :],
            transmittance=self.transmittance[:, index:index+1, :],
            absorptance=self.absorptance[:, index:index+1, :],
        )

    def spectrum(
        self,
        temperature_index: int,
        angle_index: int,
        property: Literal["reflectance", "transmittance", "absorptance"] = "reflectance"
    ) -> Float32[Array, "num_wavelengths"]:
        """
        Return the 1D (non‑polarized) spectrum for a specific temperature
        and angle combination.

        Args:
            temperature_index: Temperature index.
            angle_index:       Angle index.
            property:          One of "reflectance", "transmittance", "absorptance".
        """
        if property == "reflectance":
            data = self.reflectance
        elif property == "transmittance":
            data = self.transmittance
        elif property == "absorptance":
            data = self.absorptance
        else:
            raise ValueError(
                "property must be 'reflectance', 'transmittance' or 'absorptance'"
            )
        return data[temperature_index, angle_index, :]

    def non_polarized_reflectance(self) -> Float32[Array, "num_temperatures num_angles num_wavelengths"]:
        """Non‑polarized reflectance (assumes stored array is already non‑polarized)."""
        return self.reflectance

    def non_polarized_transmittance(self) -> Float32[Array, "num_temperatures num_angles num_wavelengths"]:
        return self.transmittance

    def non_polarized_absorptance(self) -> Float32[Array, "num_temperatures num_angles num_wavelengths"]:
        return self.absorptance

    # ------------------------------------------------------------------
    # Summary statistics for a given (temperature, angle) and property
    # ------------------------------------------------------------------
    def summary(
        self,
        temperature_index: int,
        angle_index: int,
        property: Literal["reflectance", "transmittance", "absorptance"] = "reflectance"
    ) -> dict:
        """
        Return max, min and average of the non‑polarized spectrum over
        all wavelengths for the selected temperature, angle and optical
        property.

        Returns a dictionary with keys "max", "min", "average".
        """
        spec = self.spectrum(temperature_index, angle_index, property)
        return {
            "max": jnp.max(spec),
            "min": jnp.min(spec),
            "average": jnp.mean(spec),
        }