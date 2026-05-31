import jax.numpy as jnp

from ..types.structure import Mask, Materials, Thicknesses
from dataclasses import dataclass
from jax.tree_util import register_dataclass

@register_dataclass
@dataclass(frozen=True)
class Structure:
    materials: Materials
    thicknesses_um: Thicknesses
    active_mask: Mask
    free_thickness_mask: Mask
    free_material_mask: Mask

    @property
    def num_layers(self) -> int:
        return jnp.sum(self.active_mask).item()

    @property
    def free_layers_mask(self) -> Mask:
        return self.free_thickness_mask | self.free_material_mask