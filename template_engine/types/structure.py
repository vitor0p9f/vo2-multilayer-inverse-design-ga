import enum
import jax
import jax.numpy as jnp

from dataclasses import dataclass
from jax.tree_util import register_dataclass
from typing import Optional, Tuple

@register_dataclass
@dataclass(frozen=True)
class Layer:
    material_symbol: Optional[str]
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
    layers: Tuple[Layer | FreeBlock]

class Mapping(enum.IntEnum):
    LAYER = enum.auto()
    FREE_LAYER = enum.auto()
    PARTIAL_LAYER_MISSING_MATERIAL = enum.auto()
    PARTIAL_LAYER_MISSING_THICKNESS = enum.auto()

    SUBSTRATE = enum.auto()
    PARTIAL_SUBSTRATE_MISSING_MATERIAL = enum.auto()

    INCIDENCE_MEDIUM = enum.auto()
    PARTIAL_INCIDENCE_MEDIUM_MISSING_MATERIAL = enum.auto()