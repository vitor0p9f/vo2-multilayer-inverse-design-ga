from dataclasses import dataclass
from jax.tree_util import register_dataclass
from jaxtyping import Float32, Array

@register_dataclass
@dataclass(frozen=True)
class OpticalProperty:
    p_polarized: Float32[Array, "num_wavelengths"]
    s_polarized: Float32[Array, "num_wavelengths"]
    
    @property
    def unpolarized(self) -> Float32[Array, "num_wavelengths"]:
        return (self.p_polarized + self.s_polarized) / 2

@register_dataclass
@dataclass(frozen=True)
class Absorbance(OpticalProperty):
    pass

@register_dataclass
@dataclass(frozen=True)
class Reflectance(OpticalProperty):
    pass

@register_dataclass
@dataclass(frozen=True)
class Transmittance(OpticalProperty):
    pass