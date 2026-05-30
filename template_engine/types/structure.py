from dataclasses import dataclass
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from jax.tree_util import register_dataclass

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
class Structure:
    layers: Tuple[Layer | FreeBlock]