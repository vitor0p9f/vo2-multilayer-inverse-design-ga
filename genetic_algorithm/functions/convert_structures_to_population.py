from typing import List
import jax.numpy as jnp
from specifications.objects.structure import Structure
from genetic_algorithm.objects.population import Population

def structures_to_population(structs: List[Structure]) -> Population:
    """
    Convert a list of Structure objects into a Population pytree.

    Args:
        structs: List of Structure instances. They must all share the same
                 free_thickness_mask and free_material_mask (template‑derived).
                 The list may be empty.

    Returns:
        Population with stacked arrays. If the list is empty, returns a
        Population with empty (0,0) arrays.
    """
    if not structs:
        return Population(
            materials=jnp.empty((0, 0), dtype=jnp.int8),
            thicknesses_m=jnp.empty((0, 0), dtype=jnp.float32),
            active_mask=jnp.empty((0, 0), dtype=bool),
            free_thickness_mask=jnp.empty((0, 0), dtype=bool),
            free_material_mask=jnp.empty((0, 0), dtype=bool),
        )

    # Stack all fields across the list
    materials = jnp.stack([s.materials for s in structs])
    thicknesses_m = jnp.stack([s.thicknesses_m for s in structs])
    active_mask = jnp.stack([s.active_mask for s in structs])
    # Free masks are identical for all – take from first element
    free_thick = structs[0].free_thickness_mask
    free_mat = structs[0].free_material_mask
    free_thickness_mask = jnp.repeat(free_thick[None, :], len(structs), axis=0)
    free_material_mask = jnp.repeat(free_mat[None, :], len(structs), axis=0)

    return Population(
        materials=materials,
        thicknesses_m=thicknesses_m,
        active_mask=active_mask,
        free_thickness_mask=free_thickness_mask,
        free_material_mask=free_material_mask,
    )