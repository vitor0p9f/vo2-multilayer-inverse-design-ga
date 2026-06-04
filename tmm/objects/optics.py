from specifications.types.optics import Reflectance as OpticalReflectance
from specifications.types.optics import Absorptance as OpticalAbsorptance
from specifications.types.optics import Transmittance as OpticalTransmittance
from dataclasses import dataclass
from jax.tree_util import register_dataclass

@register_dataclass
@dataclass(frozen=True)
class Absorptance:
    p_polarized: OpticalAbsorptance
    s_polarized: OpticalAbsorptance
    non_polarized: OpticalAbsorptance

@register_dataclass
@dataclass(frozen=True)
class Reflectance:
    p_polarized: OpticalReflectance
    s_polarized: OpticalReflectance
    non_polarized: OpticalReflectance

@register_dataclass
@dataclass(frozen=True)
class Transmittance:
    p_polarized: OpticalTransmittance
    s_polarized: OpticalTransmittance
    non_polarized: OpticalTransmittance