import enum
import jax
import jax.numpy as jnp

from dataclasses import dataclass
from typing import Optional, Tuple
from jax.tree_util import register_dataclass

@register_dataclass
@dataclass(frozen=True)
class Layer:
    material_symbol: Optional[str] = None
    thickness_m: Optional[jax.Array] = None

@register_dataclass
@dataclass(frozen=True)
class Substrate(Layer):
    def __post_init__(self):
        if self.thickness_m is None:
            object.__setattr__(self, 'thickness_m', jnp.inf)

@register_dataclass
@dataclass(frozen=True)
class IncidenceMedium(Layer):
    def __post_init__(self):
        if self.thickness_m is None:
            object.__setattr__(self, 'thickness_m', jnp.inf)

@register_dataclass
@dataclass(frozen=True)
class FreeBlock:
    number: int = 0

class Mapping(enum.IntEnum):
    LAYER = enum.auto()
    FREE_LAYER = enum.auto()
    PARTIAL_LAYER_MISSING_MATERIAL = enum.auto()
    PARTIAL_LAYER_MISSING_THICKNESS = enum.auto()
    SUBSTRATE = enum.auto()
    PARTIAL_SUBSTRATE_MISSING_MATERIAL = enum.auto()
    INCIDENCE_MEDIUM = enum.auto()
    PARTIAL_INCIDENCE_MEDIUM_MISSING_MATERIAL = enum.auto()

LayerMapping = Tuple[Mapping, ...]