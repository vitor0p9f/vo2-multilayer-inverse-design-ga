from typing import List, Tuple
from ..types.evaluation_result import EvaluationResult
from genetic_algorithm.objects.population import Population
from specifications.objects.structure import Structure
import jax.numpy as jnp


def replace_population(
    parents: List[EvaluationResult],
    offspring: List[EvaluationResult],
    elite_frac: float = 0.1,
) -> Tuple[Population, List[Structure]]:
    """
    Build the next generation by keeping the best elites from the
    current population and filling the remaining slots with the best
    offspring. Both lists are sorted by cost (ascending).

    Args:
        parents:   Evaluated results of the current population.
        offspring: Evaluated results of the offspring.
        elite_frac: Fraction of the population to keep as elites.

    Returns:
        population:   Population object with stacked arrays.
        new_structures: List of Structure objects that make up the
                        new population (same length as parents).
    """
    pop_size = len(parents)
    if pop_size == 0:
        empty_pop = Population(
            materials=jnp.empty((0, 0), dtype=jnp.int8),
            thicknesses_m=jnp.empty((0, 0), dtype=jnp.float64),
            active_mask=jnp.empty((0, 0), dtype=bool),
            free_thickness_mask=jnp.empty((0, 0), dtype=bool),
            free_material_mask=jnp.empty((0, 0), dtype=bool),
        )
        return empty_pop, []

    elite_count = int(elite_frac * pop_size)
    elite_count = max(0, min(elite_count, pop_size))

    # Sort both lists by cost (ascending)
    sorted_parents = sorted(parents, key=lambda r: r.cost)
    sorted_offspring = sorted(offspring, key=lambda r: r.cost)

    # Elites from the current generation
    elites = sorted_parents[:elite_count]

    # Fill the remaining slots with the best offspring
    needed = pop_size - elite_count
    best_offspring = sorted_offspring[:needed]

    # Combine structures
    new_structures = [e.structure for e in elites] + [o.structure for o in best_offspring]

    # Safety pad (should not be needed in a correct GA pipeline)
    while len(new_structures) < pop_size:
        new_structures.append(elites[0].structure)

    # Stack into Population arrays
    materials = jnp.stack([s.materials for s in new_structures])
    thicknesses_m = jnp.stack([s.thicknesses_m for s in new_structures])
    active_mask = jnp.stack([s.active_mask for s in new_structures])

    # Free masks are identical across structures; take from the first
    free_thick = new_structures[0].free_thickness_mask
    free_mat = new_structures[0].free_material_mask
    free_thickness_mask = jnp.repeat(free_thick[None, :], pop_size, axis=0)
    free_material_mask = jnp.repeat(free_mat[None, :], pop_size, axis=0)

    pop = Population(
        materials=materials,
        thicknesses_m=thicknesses_m,
        active_mask=active_mask,
        free_thickness_mask=free_thickness_mask,
        free_material_mask=free_material_mask,
    )
    return pop, new_structures