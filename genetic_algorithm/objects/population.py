from dataclasses import dataclass
from jax.tree_util import register_dataclass

from structure.types.structure import Mask, Materials, Thicknesses

@register_dataclass
@dataclass(frozen=True)
class Population:
    """Batch of Structures with leading dimension `pop_size`."""
    materials: Materials          # shape (pop_size, n_layers)
    thicknesses_m: Thicknesses    # shape (pop_size, n_layers)
    active_mask: Mask
    free_thickness_mask: Mask
    free_material_mask: Mask

    @property
    def size(self) -> int:
        return self.materials.shape[0]