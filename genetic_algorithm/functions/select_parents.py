from typing import List, Tuple
from ..types.evaluation_result import EvaluationResult
from specifications.objects.structure import Structure

import jax
import jax.numpy as jnp

def select_parent_pairs(
    results: List[EvaluationResult],
    key: jax.random.PRNGKey,
    elite_frac: float = 0.1,
    tournament_size: int = 3,
) -> Tuple[jax.random.PRNGKey, List[Tuple[Structure, Structure]]]:
    """
    Tournament selection of parent pairs with replacement.

    For each required parent, `tournament_size` individuals are sampled
    uniformly with replacement, and the one with the smallest cost wins.

    Enough pairs are returned so that, together with the preserved elites,
    the total population size is maintained. Each pair is expected to
    generate exactly two children via crossover.

    Args:
        results:          Evaluated population (current generation).
        key:              JAX PRNG key.
        elite_frac:       Fraction of the population to keep as elites
                          (elites are not subjected to crossover).
        tournament_size:  Number of participants in each tournament.

    Returns:
        (next_key, parent_pairs)
        parent_pairs: List of (parent1, parent2) Structure tuples.
    """
    pop_size = len(results)
    if pop_size == 0:
        return key, []

    # Number of elites to preserve
    elite_count = int(elite_frac * pop_size)
    elite_count = max(0, min(elite_count, pop_size))
    offspring_needed = pop_size - elite_count

    if offspring_needed <= 0:
        return key, []   # elites already fill the population

    # Each pair yields 2 children → ceil(offspring_needed / 2) pairs
    num_pairs = (offspring_needed + 1) // 2
    num_parents = 2 * num_pairs

    # Fitness vector (lower is better)
    costs = jnp.array([r.cost for r in results], dtype=jnp.float32)

    # Split the PRNG key: one part for the random indices, one for future use
    next_key, subkey = jax.random.split(key)

    # Generate random participant indices: shape (num_parents, tournament_size)
    participant_idx = jax.random.randint(
        subkey,
        shape=(num_parents, tournament_size),
        minval=0,
        maxval=pop_size,
        dtype=jnp.int32,
    )

    # For each row, select the participant with the smallest cost
    # costs[participant_idx] → (num_parents, tournament_size)
    best_local = jnp.argmin(costs[participant_idx], axis=-1)      # (num_parents,)
    winner_indices = participant_idx[jnp.arange(num_parents), best_local]

    # Reshape into pairs (num_pairs, 2)
    pair_indices = winner_indices.reshape(num_pairs, 2)

    # Build the list of Structure tuples
    parent_pairs = []
    for i in range(num_pairs):
        idx1 = int(pair_indices[i, 0])
        idx2 = int(pair_indices[i, 1])
        parent_pairs.append((results[idx1].structure, results[idx2].structure))

    return next_key, parent_pairs