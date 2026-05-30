import enum
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Optional, Tuple, Union
from jax.tree_util import register_dataclass
from jaxtyping import Int8, Float32, Bool, Array

@register_dataclass
@dataclass(frozen=True)
class Layer:
    material_symbol: Optional[str] = None
    thickness_um: Optional[jax.Array] = None

@register_dataclass
@dataclass(frozen=True)
class Substrate(Layer):
    def __post_init__(self):
        if self.thickness_um is None:
            object.__setattr__(self, 'thickness_um', jnp.inf)

@register_dataclass
@dataclass(frozen=True)
class IncidenceMedium(Layer):
    def __post_init__(self):
        if self.thickness_um is None:
            object.__setattr__(self, 'thickness_um', jnp.inf)

@register_dataclass
@dataclass(frozen=True)
class FreeBlock:
    number: int = 0

@register_dataclass
@dataclass(frozen=True)
class StructureTemplate:
    layers: Tuple[Union[Layer, FreeBlock], ...]

class Mapping(enum.IntEnum):
    LAYER = enum.auto()
    FREE_LAYER = enum.auto()
    PARTIAL_LAYER_MISSING_MATERIAL = enum.auto()
    PARTIAL_LAYER_MISSING_THICKNESS = enum.auto()
    SUBSTRATE = enum.auto()
    PARTIAL_SUBSTRATE_MISSING_MATERIAL = enum.auto()
    INCIDENCE_MEDIUM = enum.auto()
    PARTIAL_INCIDENCE_MEDIUM_MISSING_MATERIAL = enum.auto()

Materials = Int8[Array, "n_layers"]
Thicknesses = Float32[Array, "n_layers"]
Mask = Bool[Array, "n_layers"]

@register_dataclass
@dataclass(frozen=True)
class Structure:
    materials: Materials
    thicknesses_um: Thicknesses
    active_mask: Mask
    free_thickness_mask: Mask
    free_material_mask: Mask

    @property
    def n_layers(self) -> int:
        return jnp.sum(self.active_mask).item()

    @property
    def free_layers_mask(self) -> Mask:
        return self.free_thickness_mask | self.free_material_mask