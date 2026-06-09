from typing import List, Tuple
import jax
from genetic_algorithm.objects.population import Population
from template.objects.template import Template
from specifications.objects.material import Material
from specifications.types.structure import Thicknesses
from template.functions.build_structure import build_random_structure_from_template

def initialize_population(
    template: Template,
    key: jax.Array,
    materials: List[Material],
    thickness_options_m: Thicknesses,
    population_size: int,
    allow_air_gap: bool = False,
) -> Tuple[jax.Array, Population]:
    """
    Create a random initial population consistent with the template.

    Args:
        template:           Validated Template.
        key:                JAX PRNG key.
        materials:          List of Material objects.
        thickness_options_m:1D array of allowed thicknesses (in metres).
        population_size:    Number of individuals to generate.
        allow_air_gap:      Whether the "Air" material may appear in free layers.

    Returns:
        (next_key, population)
    """
    structures = []
    next_key = key
    for _ in range(population_size):
        next_key, struct = build_random_structure_from_template(
            template=template,
            key=next_key,
            materials=materials,
            thickness_options_m=thickness_options_m,
            allow_air_gap=allow_air_gap,
        )
        structures.append(struct)

    # Convert list of structures to batched Population arrays
    materials_list = [s.materials for s in structures]
    thicknesses_list = [s.thicknesses_m for s in structures]
    active_list = [s.active_mask for s in structures]

    # All individuals share the same free masks; take them from the first
    free_mat_mask = structures[0].free_material_mask
    free_thick_mask = structures[0].free_thickness_mask

    population = Population(
        materials=jax.numpy.stack(materials_list),
        thicknesses_m=jax.numpy.stack(thicknesses_list),
        active_mask=jax.numpy.stack(active_list),
        free_thickness_mask=jax.numpy.stack([free_thick_mask] * population_size),
        free_material_mask=jax.numpy.stack([free_mat_mask] * population_size),
    )

    return next_key, population