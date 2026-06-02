from typing import List, Tuple
from template.objects.template import Template
from structure.objects.material import Material
from structure.objects.structure import Structure
from structure.types.structure import Thicknesses
from template.functions.build_structure import build_random_structure_from_template
from genetic_algorithm.objects.population import Population

import jax

def initialize_population(
    template: Template,
    key: jax.random.PRNGKey,
    materials: List[Material],
    thickness_options_m: Thicknesses,
    population_size: int,
    allow_air_gap: bool = False,
) -> Tuple[jax.random.PRNGKey, Population]:
    """
    Creates a population of random structures from a template.

    Args:
        template: Blueprint for layer structure.
        key: JAX PRNG key (will be advanced).
        materials: Available materials (order defines index). The material
            with symbol "Air" (if present) can be excluded from free layers
            via `allow_air_gap`.
        thickness_options_m: Allowed discrete thicknesses in meters.
        population_size: Number of structures to generate.
        allow_air_gap: If False (default), the material "Air" is excluded
            from random selection for free internal layers. The incidence
            medium and substrate are never affected.

    Returns:
        (next_key, population): The advanced PRNG key and the generated
        Population object containing `population_size` individuals.
    """
    # Advance the key: one for this function, one for future use
    next_key, subkey = jax.random.split(key)

    # Split the subkey into independent seeds for each individual
    keys = jax.random.split(subkey, population_size)

    # Function that builds one Structure from one key
    def build_one(single_key: jax.random.PRNGKey) -> Structure:
        _, struct = build_random_structure_from_template(
            template=template,
            key=single_key,
            materials=materials,
            thickness_options_m=thickness_options_m,
            allow_air_gap=allow_air_gap,
        )
        return struct

    # Vectorise over the key batch -> Structure with shape (pop_size, ...)
    batched_struct = jax.vmap(build_one)(keys)

    # Wrap into dedicated Population type
    population = Population(
        materials=batched_struct.materials,
        thicknesses_m=batched_struct.thicknesses_m,
        active_mask=batched_struct.active_mask,
        free_thickness_mask=batched_struct.free_thickness_mask,
        free_material_mask=batched_struct.free_material_mask,
    )

    return next_key, population